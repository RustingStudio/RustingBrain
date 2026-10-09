//! Validates a YOLO11s checkpoint on a YOLO-format dataset's `val` split the
//! way ultralytics' validator does (rect batches of 16 at imgsz 1280, stride
//! 32, pad 0.5; conf 0.001, NMS IoU 0.7, at most 300 boxes) and prints mAP50
//! and mAP50-95. Images run one at a time on the device.
//!
//! ```bash
//! cargo run --release --features cuda,images --example yolo_val -- \
//!     /home/vasylt/cs2/fixtures/k13d_player_v1.safetensors /home/vasylt/cs2/dataset 1
//! ```

use rand::SeedableRng;
use rand::rngs::StdRng;
use rusting_brain::conv::ImageBatch;
use rusting_brain::detection::{
    DetectionImage, mean_average_precision, non_max_suppression, read_yolo_split, rect_shape,
};
use rusting_brain::matrix::Matrix;
use rusting_brain::param::CudaDevice;
use rusting_brain::safetensors::SafeTensors;
use rusting_brain::yolo::{Yolo11, Yolo11Scale, decode};
use std::path::Path;
use std::time::{Duration, Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let weights = args.get(1).map_or(
        "/home/vasylt/cs2/fixtures/k13d_player_v1.safetensors",
        |a| a,
    );
    let root = Path::new(args.get(2).map_or("/home/vasylt/cs2/dataset", |a| a));
    let classes: usize = args.get(3).map_or(Ok(1), |a| a.parse())?;
    let device = CudaDevice::new(0, 8192)?;
    let mut model = Yolo11::new(Yolo11Scale::S, classes, &mut StdRng::seed_from_u64(0))?;
    model.load_ultralytics(&mut SafeTensors::open(weights)?)?;
    model.to_cuda_on(&device)?;

    let mut images = read_yolo_split(root, "val")?
        .into_iter()
        .map(|(path, labels)| DetectionImage::open(&path, &labels, 1280))
        .collect::<Result<Vec<_>, _>>()?;
    images.sort_by(|a, b| {
        (a.height as f32 / a.width as f32).total_cmp(&(b.height as f32 / b.width as f32))
    });
    let (mut results, mut model_time, mut total_time) =
        (Vec::new(), Duration::ZERO, Duration::ZERO);
    for batch in images.chunks(16) {
        let aspects: Vec<f32> = batch
            .iter()
            .map(|i| i.height as f32 / i.width as f32)
            .collect();
        let [height, width] = rect_shape(&aspects, 1280, 32, 0.5);
        for image in batch {
            let (image, _, _) = image.letterbox(height, width, false);
            let pixels = image.rgb.iter().map(|&v| v as f32 / 255.0).collect();
            let input = ImageBatch::from_tokens(
                1,
                height,
                width,
                Matrix::from_vec(height * width, 3, pixels),
            );
            let start = Instant::now();
            let tape = model.forward_device(&device.upload(&input)?, false)?;
            let maps = model
                .outputs_device(&tape)
                .map(|m| m.download())
                .into_iter()
                .collect::<Result<Vec<_>, _>>()?;
            let model_done = Instant::now();
            let (boxes, scores) = decode([&maps[0], &maps[1], &maps[2]], classes).remove(0);
            let detections = non_max_suppression(&boxes, &scores, classes, 0.001, 0.7, 300);
            model_time += model_done - start;
            total_time += start.elapsed();
            let labels = image
                .classes
                .iter()
                .copied()
                .zip(image.boxes.iter().copied())
                .collect();
            results.push((detections, labels));
        }
    }
    let [map50, map50_95] = mean_average_precision(&results);
    let n = results.len() as f64;
    println!(
        "{} images: mAP50 {map50:.4}, mAP50-95 {map50_95:.4}; {:.1} ms/image (model {:.1} ms, unfused, caches kept)",
        results.len(),
        total_time.as_secs_f64() * 1e3 / n,
        model_time.as_secs_f64() * 1e3 / n,
    );
    Ok(())
}
