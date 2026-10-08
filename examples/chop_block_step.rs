//! Times one forward and backward pass through RustingChop's seven
//! convolution blocks (conv 3x3 without bias, GroupNorm, SiLU, then spatial
//! softmax and global average pooling) on two paths: the per-layer one, where
//! every convolution uploads its input and downloads its output and the rest
//! runs on the host, and the device-resident one, where only the input goes
//! up.
//!
//! ```bash
//! cargo run --release --features cuda --example chop_block_step -- 128
//! ```
//!
//! The head after the pooled features is a few small matrices and is left out
//! of both paths; the loss gradient is ones over every feature.

use rand::SeedableRng;
use rand::rngs::StdRng;
use rusting_brain::conv::{
    self, ImageBatch, TrainableConv2d, TrainableGroupNorm, global_avg_pool,
    global_avg_pool_backward, spatial_softmax, spatial_softmax_backward,
};
use rusting_brain::ffn::{silu, silu_derivative};
use rusting_brain::matrix::Matrix;
use rusting_brain::param::CudaDevice;
use std::time::Instant;

const BLOCKS: [(usize, usize); 7] = [
    (32, 2),
    (32, 1),
    (64, 2),
    (64, 1),
    (96, 2),
    (96, 1),
    (128, 2),
];
const STEPS: usize = 5;

fn median(mut times: Vec<f32>) -> f32 {
    times.sort_by(f32::total_cmp);
    times[times.len() / 2]
}

fn per_layer_step(
    blocks: &mut [(TrainableConv2d, TrainableGroupNorm)],
    input: &ImageBatch,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut x = input.clone();
    let mut caches = Vec::new();
    for (conv, norm) in blocks.iter() {
        let (y, conv_cache) = conv.forward(&x)?;
        let (y, norm_cache) = norm.forward(&y)?;
        let mut tokens = y.tokens.clone();
        tokens.data.iter_mut().for_each(|v| *v = silu(*v));
        x = ImageBatch::from_tokens(y.batch, y.height, y.width, tokens);
        caches.push((conv_cache, norm_cache, y.tokens));
    }
    let (keypoints, softmax_cache) = spatial_softmax(&x);
    let pooled = global_avg_pool(&x);
    let mut dx = spatial_softmax_backward(
        &softmax_cache,
        &Matrix::from_vec(
            keypoints.rows,
            keypoints.cols,
            vec![1.0; keypoints.data.len()],
        ),
    );
    let d_pool = global_avg_pool_backward(
        &Matrix::from_vec(pooled.rows, pooled.cols, vec![1.0; pooled.data.len()]),
        x.height,
        x.width,
    );
    dx.tokens
        .data
        .iter_mut()
        .zip(&d_pool.tokens.data)
        .for_each(|(a, b)| *a += b);
    for ((conv, norm), (conv_cache, norm_cache, pre)) in blocks.iter_mut().zip(caches).rev() {
        dx.tokens
            .data
            .iter_mut()
            .zip(&pre.data)
            .for_each(|(g, p)| *g *= silu_derivative(*p));
        let d_conv = norm.backward(&norm_cache, &dx);
        dx = conv.backward(&conv_cache, &d_conv)?;
    }
    Ok(())
}

fn device_step(
    device: &CudaDevice,
    blocks: &mut [(TrainableConv2d, TrainableGroupNorm)],
    input: &ImageBatch,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut x = device.upload(input)?;
    let mut caches = Vec::new();
    for (conv, norm) in blocks.iter() {
        let (y, conv_cache) = conv.forward_device(&x)?;
        let (y, norm_cache) = norm.forward_device(&y)?;
        x = conv::silu_device(&y)?;
        caches.push((conv_cache, norm_cache, y));
    }
    let (keypoints, softmax_cache) = conv::spatial_softmax_device(&x)?;
    let pooled = conv::global_avg_pool_device(&x)?;
    let ones = |like: &conv::DeviceImageBatch| {
        let mut host = ImageBatch::new(like.batch, like.height, like.width, like.channels());
        host.tokens.data.fill(1.0);
        device.upload(&host)
    };
    let mut dx = conv::spatial_softmax_backward_device(&softmax_cache, &ones(&keypoints)?)?;
    let d_pool = conv::global_avg_pool_backward_device(&ones(&pooled)?, x.height, x.width)?;
    // NOTE: the sum goes through the host; a device add would save one
    // round trip of the last map, which is 6x12x128 per sample.
    let mut sum = device.download(&dx)?;
    let d_pool = device.download(&d_pool)?;
    sum.tokens
        .data
        .iter_mut()
        .zip(&d_pool.tokens.data)
        .for_each(|(a, b)| *a += b);
    dx = device.upload(&sum)?;
    for ((conv, norm), (conv_cache, norm_cache, pre)) in blocks.iter_mut().zip(caches).rev() {
        let d_pre = conv::silu_backward_device(&pre, &dx)?;
        let d_conv = norm.backward_device(&norm_cache, &d_pre)?;
        dx = conv.backward_device(&conv_cache, &d_conv)?;
    }
    // Pooling the input gradient and reading it back waits for the stream at
    // the cost of a few kilobytes.
    device.download(&conv::global_avg_pool_device(&dx)?)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let batch: usize = std::env::args().nth(1).map_or(Ok(128), |v| v.parse())?;
    let device = CudaDevice::new(0, 8192)?;
    let mut rng = StdRng::seed_from_u64(7);
    let mut blocks = Vec::new();
    let mut channels = 15;
    for (out, stride) in BLOCKS {
        let mut conv = TrainableConv2d::new(channels, out, 3, stride, &mut rng)?.without_bias();
        conv.to_cuda_on(&device)?;
        blocks.push((conv, TrainableGroupNorm::new(out.min(8), out, 1e-3)?));
        channels = out;
    }
    let mut input = ImageBatch::new(batch, 96, 192, 15);
    use rand::Rng;
    input
        .tokens
        .data
        .iter_mut()
        .for_each(|v| *v = rng.r#gen::<f32>() * 2.0 - 1.0);

    let mut per_layer = Vec::new();
    for step in 0..=STEPS {
        let start = Instant::now();
        per_layer_step(&mut blocks, &input)?;
        if step > 0 {
            per_layer.push(start.elapsed().as_secs_f32() * 1000.0);
        }
    }
    for (_, norm) in &mut blocks {
        norm.to_cuda_on(&device)?;
    }
    let mut resident = Vec::new();
    for step in 0..=STEPS {
        let start = Instant::now();
        device_step(&device, &mut blocks, &input)?;
        if step > 0 {
            resident.push(start.elapsed().as_secs_f32() * 1000.0);
        }
    }
    let (per_layer, resident) = (median(per_layer), median(resident));
    println!(
        "batch {batch}: per-layer {per_layer:.1} ms, device-resident {resident:.1} ms, {:.2}x",
        per_layer / resident
    );
    Ok(())
}
