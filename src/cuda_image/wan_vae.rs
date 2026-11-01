//! Wan 2.2's video decoder on the device, FP16 like the rest of this module.
//!
//! The same chunked, causal walk as [`crate::wan_vae::WanVaeDecoder`] on the
//! host, with one [`Map`] per frame. A three-frame convolution is one
//! [`DeviceConv`] per tap: the newest tap writes the output (and adds the
//! bias), the older ones accumulate into it, and a tap that would read before
//! the start of the clip is skipped, which is the zero padding.
//!
//! NOTE: the older taps carry a zero bias that `conv_into` still adds,
//! one extra pass over each output; a bias-free epilogue flag on
//! `DeviceConv` removes it if it ever shows in a profile.

use super::*;
use std::sync::atomic::Ordering;
use crate::wan_vae::{CausalConv3d, Streams, WanAttention, WanResnet, WanVaeDecoder, unpatchify};

const WAN_KERNELS: &str = r#"
typedef unsigned short half_t;
__device__ __forceinline__ float h2f(half_t h){
  float f; asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h)); return f;
}
__device__ __forceinline__ half_t f2h(float v){
  half_t h; asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(v)); return h;
}

// `x / max(|x|, 1e-12) * gamma` over the channels of each pixel, gamma with
// sqrt(C) already folded in, then SiLU if asked. One thread a pixel, so each
// channel's read is coalesced across the block.
extern "C" __global__ void wan_rms(half_t*__restrict__ x,const float*__restrict__ gamma,
    int channels,long long pixels,int silu){
  long long p=(long long)blockIdx.x*blockDim.x+threadIdx.x;
  if(p>=pixels)return;
  float total=0.f;
  for(int c=0;c<channels;c++){float v=h2f(x[c*pixels+p]);total+=v*v;}
  float inv=1.f/fmaxf(sqrtf(total),1e-12f);
  for(int c=0;c<channels;c++){
    float v=h2f(x[c*pixels+p])*inv*gamma[c];
    if(silu)v=v/(1.f+expf(-v));
    x[c*pixels+p]=f2h(v);
  }
}

// One output frame of `DupUp3D`, added into `out` ([out_c, h*fs, w*fs]):
// channel j = o*factor + dt*fs*fs + dy*fs + dx of the unfolded input reads
// input channel j / repeats.
extern "C" __global__ void wan_dup_up_add(half_t*__restrict__ out,const half_t*__restrict__ in,
    int h,int w,int fs,int factor,int repeats,int dt,long long n){
  long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=n)return;
  long long ow=(long long)w*fs,plane=ow*h*fs;
  long long o=i/plane,p=i%plane,oy=p/ow,ox=p%ow;
  int dy=(int)(oy%fs),dx=(int)(ox%fs);
  long long source=(o*factor+dt*fs*fs+dy*fs+dx)/repeats;
  out[i]=f2h(h2f(out[i])+h2f(in[(source*h+oy/fs)*w+ox/fs]));
}
"#;

fn wan_ptx() -> Result<&'static Ptx, NetworkError> {
    static PTX: OnceLock<Result<Ptx, String>> = OnceLock::new();
    PTX.get_or_init(|| compile_ptx(WAN_KERNELS).map_err(|error| error.to_string()))
        .as_ref()
        .map_err(|error| NetworkError::Cuda(format!("Wan VAE kernel compilation failed: {error}")))
}

/// A three-frame convolution, one convolution a tap, oldest first. Only the
/// newest carries the bias.
struct DeviceCausal {
    taps: Vec<DeviceConv>,
}

struct DeviceWanResnet {
    norm1: CudaSlice<f32>,
    conv1: DeviceCausal,
    norm2: CudaSlice<f32>,
    conv2: DeviceCausal,
    shortcut: Option<DeviceConv>,
}

struct DeviceWanAttention {
    norm: CudaSlice<f32>,
    query: DeviceDense,
    key: DeviceDense,
    value: DeviceDense,
    proj: DeviceDense,
}

struct DeviceWanUpBlock {
    resnets: Vec<DeviceWanResnet>,
    /// The time convolution's two halves, one per frame each input frame
    /// becomes, then the 3x3 convolution after the nearest doubling.
    upsample: Option<(Option<[DeviceCausal; 2]>, DeviceConv)>,
    shortcut: Option<crate::wan_vae::DupUp>,
}

/// The device and this module's kernels, with the layers that only need
/// those.
struct WanKernels {
    /// Set while the clip's last latent frame goes through.
    ending: std::sync::atomic::AtomicBool,
    gpu: Arc<ImageGpu>,
    rms: CudaFunction,
    dup_up_add: CudaFunction,
}

/// A [`WanVaeDecoder`] resident on the device.
pub struct DeviceWanDecoder {
    kernels: WanKernels,
    patch_size: usize,
    post_quant: DeviceConv,
    conv_in: DeviceCausal,
    mid: (DeviceWanResnet, DeviceWanAttention, DeviceWanResnet),
    up_blocks: Vec<DeviceWanUpBlock>,
    norm_out: CudaSlice<f32>,
    conv_out: DeviceCausal,
}

impl std::fmt::Debug for DeviceWanDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceWanDecoder").finish_non_exhaustive()
    }
}

/// Output rows `rows` of tap `tap`, with the bias only on the newest tap.
fn upload_tap(
    gpu: &ImageGpu,
    conv: &CausalConv3d,
    tap: usize,
    rows: std::ops::Range<usize>,
) -> Result<DeviceConv, NetworkError> {
    let host = &conv.conv;
    let width = host.weight.cols / conv.taps;
    let mut weight = Vec::with_capacity(rows.len() * width);
    for row in rows.clone() {
        weight.extend_from_slice(&host.weight.row(row)[tap * width..(tap + 1) * width]);
    }
    let bias = host
        .bias
        .as_ref()
        .filter(|_| tap + 1 == conv.taps)
        .map(|bias| bias[rows.clone()].to_vec());
    let tap = Conv2d::new(
        Matrix::from_vec(rows.len(), width, weight),
        bias,
        host.in_channels / conv.taps,
        host.kernel,
        host.stride,
        host.padding,
    )?;
    upload_conv(gpu, &tap)
}

/// The channels each tap of `conv` reads, once its weight is checked to
/// split evenly into `taps` per-frame kernels.
fn tap_channels(conv: &CausalConv3d) -> Result<usize, NetworkError> {
    let host = &conv.conv;
    let area = host.kernel * host.kernel;
    if conv.taps == 0
        || !host.in_channels.is_multiple_of(conv.taps)
        || host.weight.cols != host.in_channels * area
        || host
            .bias
            .as_ref()
            .is_some_and(|bias| bias.len() != host.weight.rows)
    {
        return Err(NetworkError::InvalidConfig(format!(
            "a {}-tap convolution over {} channels with a {}x{} weight and {:?} biases does \
             not split into taps",
            conv.taps,
            host.in_channels,
            host.weight.rows,
            host.weight.cols,
            host.bias.as_ref().map(Vec::len)
        )));
    }
    Ok(host.in_channels / conv.taps)
}

/// An RMS norm's scales, refused unless there is one a channel.
fn upload_gamma(
    gpu: &ImageGpu,
    gamma: &[f32],
    channels: usize,
) -> Result<CudaSlice<f32>, NetworkError> {
    if gamma.len() != channels {
        return Err(NetworkError::InvalidConfig(format!(
            "an RMS norm over {channels} channels has {} scales",
            gamma.len()
        )));
    }
    gpu.upload_f32(gamma)
}

fn upload_causal(
    gpu: &ImageGpu,
    conv: &CausalConv3d,
    rows: std::ops::Range<usize>,
) -> Result<DeviceCausal, NetworkError> {
    tap_channels(conv)?;
    if rows.end > conv.conv.weight.rows {
        return Err(NetworkError::InvalidConfig(format!(
            "rows {rows:?} of a {}-row convolution",
            conv.conv.weight.rows
        )));
    }
    Ok(DeviceCausal {
        taps: (0..conv.taps)
            .map(|tap| upload_tap(gpu, conv, tap, rows.clone()))
            .collect::<Result<_, _>>()?,
    })
}

fn upload_all(gpu: &ImageGpu, conv: &CausalConv3d) -> Result<DeviceCausal, NetworkError> {
    upload_causal(gpu, conv, 0..conv.conv.weight.rows)
}

fn upload_wan_resnet(gpu: &ImageGpu, resnet: &WanResnet) -> Result<DeviceWanResnet, NetworkError> {
    Ok(DeviceWanResnet {
        norm1: upload_gamma(gpu, &resnet.norm1, tap_channels(&resnet.conv1)?)?,
        conv1: upload_all(gpu, &resnet.conv1)?,
        norm2: upload_gamma(gpu, &resnet.norm2, tap_channels(&resnet.conv2)?)?,
        conv2: upload_all(gpu, &resnet.conv2)?,
        shortcut: resnet
            .shortcut
            .as_ref()
            .map(|conv| upload_conv(gpu, conv))
            .transpose()?,
    })
}

fn upload_wan_attention(
    gpu: &ImageGpu,
    attention: &WanAttention,
) -> Result<DeviceWanAttention, NetworkError> {
    Ok(DeviceWanAttention {
        norm: upload_gamma(gpu, &attention.norm, attention.query.in_dim())?,
        query: upload_dense(gpu, &attention.query)?,
        key: upload_dense(gpu, &attention.key)?,
        value: upload_dense(gpu, &attention.value)?,
        proj: upload_dense(gpu, &attention.proj)?,
    })
}

/// The frames a causal layer has seen, followed by `chunk`, and where the
/// chunk starts.
fn extend(history: &mut Option<Vec<Map>>, chunk: Vec<Map>) -> (Vec<Map>, usize) {
    let mut stream = history.take().unwrap_or_default();
    let past = stream.len();
    stream.extend(chunk);
    (stream, past)
}

/// Keeps the last two frames of `stream` as the layer's history.
fn retire(history: &mut Option<Vec<Map>>, mut stream: Vec<Map>) {
    let keep = stream.len().saturating_sub(crate::wan_vae::CACHE_FRAMES);
    stream.drain(..keep);
    *history = Some(stream);
}

impl DeviceWanDecoder {
    pub fn upload(gpu: &Arc<ImageGpu>, decoder: &WanVaeDecoder) -> Result<Self, NetworkError> {
        let up_blocks = decoder
            .up_blocks
            .iter()
            .map(|block| -> Result<DeviceWanUpBlock, NetworkError> {
                let upsample = block
                    .upsample
                    .as_ref()
                    .map(|upsample| -> Result<_, NetworkError> {
                        let time = upsample
                            .time
                            .as_ref()
                            .map(|time| -> Result<[DeviceCausal; 2], NetworkError> {
                                let rows = time.conv.weight.rows;
                                if !rows.is_multiple_of(2) {
                                    return Err(NetworkError::InvalidConfig(format!(
                                        "an upsampling time convolution makes two frames of \
                                         {rows} channels"
                                    )));
                                }
                                let half = rows / 2;
                                Ok([
                                    upload_causal(gpu, time, 0..half)?,
                                    upload_causal(gpu, time, half..2 * half)?,
                                ])
                            })
                            .transpose()?;
                        Ok((time, upload_conv(gpu, &upsample.conv)?))
                    })
                    .transpose()?;
                Ok(DeviceWanUpBlock {
                    resnets: block
                        .resnets
                        .iter()
                        .map(|resnet| upload_wan_resnet(gpu, resnet))
                        .collect::<Result<_, _>>()?,
                    upsample,
                    shortcut: block.shortcut,
                })
            })
            .collect::<Result<_, _>>()?;
        Ok(Self {
            kernels: WanKernels::new(gpu)?,
            patch_size: decoder.config.patch_size,
            post_quant: upload_conv(gpu, &decoder.post_quant)?,
            conv_in: upload_all(gpu, &decoder.conv_in)?,
            mid: (
                upload_wan_resnet(gpu, &decoder.mid.first)?,
                upload_wan_attention(gpu, &decoder.mid.attention)?,
                upload_wan_resnet(gpu, &decoder.mid.last)?,
            ),
            up_blocks,
            norm_out: upload_gamma(gpu, &decoder.norm_out, tap_channels(&decoder.conv_out)?)?,
            conv_out: upload_all(gpu, &decoder.conv_out)?,
        })
    }

    /// The host decoder's [`WanVaeDecoder::decode`], on the device.
    pub fn decode(&self, latent: &[FeatureMap]) -> Result<Vec<FeatureMap>, NetworkError> {
        let gpu = &*self.kernels.gpu;
        let mut streams = Streams::new();
        let mut video = Vec::new();
        for (index, frame) in latent.iter().enumerate() {
            streams.rewind();
            self.kernels
                .ending
                .store(index + 1 == latent.len(), Ordering::Relaxed);
            let chunk = vec![gpu.conv(&self.post_quant, &upload_map(gpu, frame)?)?];
            for frame in self.decode_chunk(&mut streams, chunk, index == 0)? {
                let mut frame = unpatchify(&download_map(gpu, &frame)?, self.patch_size);
                // An FP16 overflow above becomes infinities, and the next
                // norm turns them into NaN; say so rather than clamp it away.
                if frame.data.iter().any(|value| !value.is_finite()) {
                    return Err(NetworkError::Cuda(
                        "Wan VAE decode overflowed FP16 on the device; decode on the host".into(),
                    ));
                }
                for value in &mut frame.data {
                    *value = value.clamp(-1.0, 1.0);
                }
                video.push(frame);
            }
        }
        Ok(video)
    }

    /// One up block over `hidden`.
    fn up_block(
        &self,
        block: &DeviceWanUpBlock,
        streams: &mut Streams<Map>,
        mut hidden: Vec<Map>,
        first: bool,
    ) -> Result<Vec<Map>, NetworkError> {
        let (k, gpu) = (&self.kernels, &*self.kernels.gpu);
        let copy = match block.shortcut {
            Some(_) => Some(
                hidden
                    .iter()
                    .map(|frame| clone_map(gpu, frame))
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            None => None,
        };
        for resnet in &block.resnets {
            hidden = k.resnet(resnet, streams, hidden)?;
        }
        if let Some((time, conv)) = &block.upsample {
            if let Some([early, late]) = time {
                let history = streams.next();
                if history.is_none() {
                    // As on the host: the first chunk keeps its frames.
                    *history = Some(Vec::new());
                } else {
                    let (stream, past) = extend(history, hidden);
                    let early = k.causal_over(early, &stream, past, None)?;
                    let late = k.causal_over(late, &stream, past, None)?;
                    retire(history, stream);
                    hidden = early
                        .into_iter()
                        .zip(late)
                        .flat_map(|(early, late)| [early, late])
                        .collect();
                }
            }
            hidden = hidden
                .into_iter()
                .map(|frame| {
                    let large = gpu.upsample_nearest(&frame, 2)?;
                    drop(frame);
                    gpu.conv(conv, &large)
                })
                .collect::<Result<_, _>>()?;
        }
        if let (Some(shortcut), Some(copy)) = (block.shortcut, copy) {
            k.dup_up_add(shortcut, &copy, &mut hidden, first)?;
        }
        Ok(hidden)
    }

    fn decode_chunk(
        &self,
        streams: &mut Streams<Map>,
        chunk: Vec<Map>,
        first: bool,
    ) -> Result<Vec<Map>, NetworkError> {
        let k = &self.kernels;
        let mut hidden = k.causal(&self.conv_in, streams.next(), chunk, None)?;
        hidden = k.resnet(&self.mid.0, streams, hidden)?;
        hidden = hidden
            .iter()
            .map(|frame| k.attention(&self.mid.1, frame))
            .collect::<Result<_, _>>()?;
        hidden = k.resnet(&self.mid.2, streams, hidden)?;
        // The blocks after the last one that adds frames take them one at a
        // time. The causal caches make that the same computation, and the
        // full-size frames, where the memory goes, are never held together.
        let tail = self
            .up_blocks
            .iter()
            .rposition(|block| {
                let time = block.upsample.as_ref().is_some_and(|up| up.0.is_some());
                time || block.shortcut.is_some_and(|shortcut| shortcut.time > 1)
            })
            .map_or(0, |last| last + 1);
        for block in &self.up_blocks[..tail] {
            hidden = self.up_block(block, streams, hidden, first)?;
        }
        let mark = streams.position();
        let mut video = Vec::with_capacity(hidden.len());
        let ending = k.ending.load(Ordering::Relaxed);
        let count = hidden.len();
        for (index, frame) in hidden.into_iter().enumerate() {
            // Only the clip's very last frame feeds no later output.
            k.ending.store(ending && index + 1 == count, Ordering::Relaxed);
            streams.seek(mark);
            let mut frame = vec![frame];
            for block in &self.up_blocks[tail..] {
                frame = self.up_block(block, streams, frame, first)?;
            }
            k.rms(&mut frame[0], &self.norm_out, true)?;
            video.extend(k.causal(&self.conv_out, streams.next(), frame, None)?);
        }
        Ok(video)
    }
}

impl WanKernels {
    fn new(gpu: &Arc<ImageGpu>) -> Result<Self, NetworkError> {
        let module = gpu
            .context
            .stream
            .context()
            .load_module(wan_ptx()?.clone())
            .map_err(cuda_err("Wan VAE module loading"))?;
        let get = |name: &str| {
            module
                .load_function(name)
                .map_err(cuda_err("Wan VAE kernel lookup"))
        };
        Ok(Self {
            ending: Default::default(),
            rms: get("wan_rms")?,
            dup_up_add: get("wan_dup_up_add")?,
            gpu: Arc::clone(gpu),
        })
    }

    /// A causal convolution over `chunk` after `history`, which it updates.
    ///
    /// A spatial one keeps in `history` not the frames it has seen but what
    /// they have already contributed to the next outputs, so each frame is
    /// gathered once and every tap's product goes straight into the output it
    /// belongs to. The gather moves as many bytes as the GEMMs do, and the
    /// old order, one output at a time, did it once a tap.
    fn causal(
        &self,
        conv: &DeviceCausal,
        history: &mut Option<Vec<Map>>,
        chunk: Vec<Map>,
        targets: Option<Vec<Map>>,
    ) -> Result<Vec<Map>, NetworkError> {
        let newest = conv.taps.len() - 1;
        if newest > 0 && conv.taps.iter().all(|tap| tap.kernel > 1 && tap.stride == 1) {
            return self.causal_fan(conv, history, chunk, targets);
        }
        let (stream, past) = extend(history, chunk);
        let outputs = self.causal_over(conv, &stream, past, targets)?;
        retire(history, stream);
        Ok(outputs)
    }

    /// [`Self::causal`] for a spatial convolution: `history` holds the partial
    /// sums of the next `taps - 1` outputs.
    fn causal_fan(
        &self,
        conv: &DeviceCausal,
        history: &mut Option<Vec<Map>>,
        chunk: Vec<Map>,
        targets: Option<Vec<Map>>,
    ) -> Result<Vec<Map>, NetworkError> {
        let gpu = &*self.gpu;
        let newest = conv.taps.len() - 1;
        let ending = self.ending.load(Ordering::Relaxed);
        let mut partial = history.take().unwrap_or_default();
        let mut targets = targets.map(Vec::into_iter);
        let count = chunk.len();
        let mut outputs = Vec::with_capacity(count);
        for (index, frame) in chunk.iter().enumerate() {
            let front = (!partial.is_empty()).then(|| partial.remove(0));
            let (mut output, beta) = match (targets.as_mut().and_then(Iterator::next), front) {
                (Some(mut target), Some(front)) => {
                    gpu.add(&mut target.tensor, &front.tensor)?;
                    (target, 1.0)
                }
                (Some(target), None) => (target, 1.0),
                (None, Some(front)) => (front, 1.0),
                (None, None) => {
                    let out = &conv.taps[newest];
                    let fresh = gpu.uninit(out.out_channels, frame.pixels())?;
                    let map = Map {
                        tensor: fresh,
                        height: frame.height,
                        width: frame.width,
                    };
                    (map, 0.0)
                }
            };
            // The last frame of the clip feeds no later output.
            let reach = if ending { newest.min(count - 1 - index) } else { newest };
            let have = partial.len();
            for back in have..reach {
                let out = &conv.taps[newest - 1 - back];
                partial.push(Map {
                    tensor: gpu.uninit(out.out_channels, frame.pixels())?,
                    height: frame.height,
                    width: frame.width,
                });
            }
            let mut jobs: Vec<(&DeviceConv, &mut Map, f32)> =
                vec![(&conv.taps[newest], &mut output, beta)];
            for (back, sum) in partial.iter_mut().enumerate().take(reach) {
                let beta = if back < have { 1.0 } else { 0.0 };
                jobs.push((&conv.taps[newest - 1 - back], sum, beta));
            }
            gpu.conv_fan(frame, &mut jobs, COLUMN_BUDGET)?;
            outputs.push(output);
        }
        *history = Some(partial);
        Ok(outputs)
    }

    /// The outputs for `stream[past..]`, each added onto its `targets` entry
    /// when there are targets.
    fn causal_over(
        &self,
        conv: &DeviceCausal,
        stream: &[Map],
        past: usize,
        targets: Option<Vec<Map>>,
    ) -> Result<Vec<Map>, NetworkError> {
        let gpu = &*self.gpu;
        let newest = conv.taps.len() - 1;
        let mut targets = targets.map(Vec::into_iter);
        (past..stream.len())
            .map(|end| {
                let mut output = match targets.as_mut().and_then(Iterator::next) {
                    Some(mut target) => {
                        gpu.conv_add(&conv.taps[newest], &stream[end], &mut target)?;
                        target
                    }
                    None => gpu.conv(&conv.taps[newest], &stream[end])?,
                };
                for (tap, weights) in conv.taps[..newest].iter().enumerate() {
                    let back = newest - tap;
                    if back <= end {
                        gpu.conv_add(weights, &stream[end - back], &mut output)?;
                    }
                }
                Ok(output)
            })
            .collect()
    }

    fn resnet(
        &self,
        resnet: &DeviceWanResnet,
        streams: &mut Streams<Map>,
        input: Vec<Map>,
    ) -> Result<Vec<Map>, NetworkError> {
        let gpu = &*self.gpu;
        let skip = input
            .iter()
            .map(|frame| match &resnet.shortcut {
                Some(conv) => gpu.conv(conv, frame),
                None => clone_map(gpu, frame),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut hidden = input;
        for frame in &mut hidden {
            self.rms(frame, &resnet.norm1, true)?;
        }
        hidden = self.causal(&resnet.conv1, streams.next(), hidden, None)?;
        for frame in &mut hidden {
            self.rms(frame, &resnet.norm2, true)?;
        }
        self.causal(&resnet.conv2, streams.next(), hidden, Some(skip))
    }

    fn attention(&self, attention: &DeviceWanAttention, input: &Map) -> Result<Map, NetworkError> {
        let gpu = &*self.gpu;
        let mut normed = clone_map(gpu, input)?;
        self.rms(&mut normed, &attention.norm, false)?;
        let tokens = gpu.to_tokens(&normed)?;
        drop(normed);
        let queries = gpu.dense(&attention.query, &tokens)?;
        let keys = gpu.dense(&attention.key, &tokens)?;
        let values = gpu.dense(&attention.value, &tokens)?;
        drop(tokens);
        let attended = gpu.attention(queries.view(), keys.view(), values.view(), 1, false)?;
        let projected = gpu.dense(&attention.proj, &attended)?;
        let mut output = gpu.to_map(&projected, input.height, input.width)?;
        gpu.add(&mut output.tensor, &input.tensor)?;
        Ok(output)
    }

    fn rms(&self, map: &mut Map, gamma: &CudaSlice<f32>, silu: bool) -> Result<(), NetworkError> {
        if gamma.len() != map.channels() {
            return Err(NetworkError::InvalidConfig(format!(
                "an RMS norm with {} scales over {} channels",
                gamma.len(),
                map.channels()
            )));
        }
        let (channels, pixels) = (map.channels() as i32, map.pixels() as i64);
        let launch = cfg(map.pixels());
        unsafe {
            self.gpu
                .context
                .stream
                .launch_builder(&self.rms)
                .arg(&mut map.tensor.data)
                .arg(gamma)
                .arg(&channels)
                .arg(&pixels)
                .arg(&i32::from(silu))
                .launch(launch)
                .map_err(cuda_err("Wan RMS norm kernel"))?;
        }
        Ok(())
    }

    /// Adds `DupUp3D` of `input` onto `output`, frame by frame.
    fn dup_up_add(
        &self,
        shortcut: crate::wan_vae::DupUp,
        input: &[Map],
        output: &mut [Map],
        first: bool,
    ) -> Result<(), NetworkError> {
        let (fs, time) = (shortcut.side, shortcut.time);
        if fs == 0 || time == 0 {
            return Err(NetworkError::InvalidConfig(format!(
                "a DupUp shortcut of {time} frames by {fs}x{fs}"
            )));
        }
        let factor = time * fs * fs;
        let offset = if first { time - 1 } else { 0 };
        if (input.len() * time).checked_sub(offset) != Some(output.len()) {
            return Err(NetworkError::InvalidConfig(format!(
                "a {time}-frame shortcut over {} frames does not make {}",
                input.len(),
                output.len()
            )));
        }
        for (index, target) in output.iter_mut().enumerate() {
            let (frame, dt) = ((index + offset) / time, (index + offset) % time);
            let source = &input[frame];
            let wanted = shortcut.out_channels * factor;
            let repeats = wanted.checked_div(source.channels()).unwrap_or(0);
            // The kernel reads input channel `j / repeats` for every `j` below
            // `wanted` and writes every element of `target`, so both have to
            // be exactly these sizes.
            if repeats == 0
                || source.channels() * repeats != wanted
                || target.channels() != shortcut.out_channels
                || target.height != source.height * fs
                || target.width != source.width * fs
            {
                return Err(NetworkError::InvalidConfig(format!(
                    "a DupUp from {}x{}x{} to {}x{}x{} at {time} frames by {fs}x{fs}",
                    source.channels(),
                    source.height,
                    source.width,
                    target.channels(),
                    target.height,
                    target.width
                )));
            }
            let n = target.tensor.len();
            unsafe {
                self.gpu
                    .context
                    .stream
                    .launch_builder(&self.dup_up_add)
                    .arg(&mut target.tensor.data)
                    .arg(&source.tensor.data)
                    .arg(&(source.height as i32))
                    .arg(&(source.width as i32))
                    .arg(&(fs as i32))
                    .arg(&(factor as i32))
                    .arg(&(repeats as i32))
                    .arg(&(dt as i32))
                    .arg(&(n as i64))
                    .launch(cfg(n))
                    .map_err(cuda_err("Wan DupUp kernel"))?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wan_vae::{DupUp, rms_norm};

    fn values(count: usize, seed: u32) -> Vec<f32> {
        (0..count as u32)
            .map(|index| {
                ((index.wrapping_mul(2654435761).wrapping_add(seed) >> 8) % 1000) as f32 / 500.0
                    - 1.0
            })
            .collect()
    }

    fn map(channels: usize, height: usize, width: usize, seed: u32) -> FeatureMap {
        FeatureMap::from_vec(
            channels,
            height,
            width,
            values(channels * height * width, seed),
        )
        .unwrap()
    }

    fn check(gpu: &ImageGpu, got: &[Map], want: &[FeatureMap], limit: f32) {
        assert_eq!(got.len(), want.len());
        for (got, want) in got.iter().zip(want) {
            let got = download_map(gpu, got).unwrap();
            let error = crate::wan_vae::tests::relative(&got.data, &want.data);
            assert!(error < limit, "relative L2 {error:.3e}");
        }
    }

    /// The RMS norm and DupUp kernels and the per-tap causal convolution
    /// against the host layers, the convolution over three chunks with a
    /// skip added on the second.
    #[test]
    fn device_wan_layers_match_the_host() {
        let gpu = ImageGpu::new(0).unwrap();
        let kernels = WanKernels::new(&gpu).unwrap();

        for silu in [false, true] {
            let mut want = map(24, 5, 7, 1);
            let mut got = upload_map(&gpu, &want).unwrap();
            let gamma: Vec<f32> = values(24, 2).iter().map(|value| value * 4.9).collect();
            rms_norm(&mut want, &gamma, silu);
            kernels
                .rms(&mut got, &gpu.upload_f32(&gamma).unwrap(), silu)
                .unwrap();
            check(&gpu, &[got], &[want], 2e-3);
        }

        let shortcut = DupUp {
            out_channels: 4,
            time: 2,
            side: 2,
        };
        let input = [map(16, 3, 4, 3), map(16, 3, 4, 4)];
        let device: Vec<_> = input
            .iter()
            .map(|frame| upload_map(&gpu, frame).unwrap())
            .collect();
        for first in [true, false] {
            let mut want = shortcut.forward(&input, first);
            let base: Vec<_> = (0..want.len() as u32)
                .map(|seed| map(4, 6, 8, 10 + seed))
                .collect();
            let mut got: Vec<_> = base
                .iter()
                .map(|frame| upload_map(&gpu, frame).unwrap())
                .collect();
            for (frame, base) in want.iter_mut().zip(&base) {
                for (value, base) in frame.data.iter_mut().zip(&base.data) {
                    *value += base;
                }
            }
            kernels
                .dup_up_add(shortcut, &device, &mut got, first)
                .unwrap();
            check(&gpu, &got, &want, 2e-3);
        }

        let (channels, out, taps) = (6, 5, 3);
        let conv = CausalConv3d {
            conv: Conv2d::new(
                Matrix::from_vec(
                    out,
                    taps * channels * 9,
                    values(out * taps * channels * 9, 5)
                        .iter()
                        .map(|value| value * 0.1)
                        .collect(),
                ),
                Some(values(out, 6)),
                taps * channels,
                3,
                1,
                1,
            )
            .unwrap(),
            taps,
        };
        let weights = upload_all(&gpu, &conv).unwrap();
        let (mut host, mut device) = (None, None);
        for (chunk, frames) in [1, 2, 2].into_iter().enumerate() {
            let input: Vec<_> = (0..frames as u32)
                .map(|seed| map(channels, 4, 5, 20 + chunk as u32 * 4 + seed))
                .collect();
            let skip: Vec<_> = (0..frames as u32)
                .map(|seed| map(out, 4, 5, 40 + seed))
                .collect();
            let mut want = conv.forward(&mut host, input.clone()).unwrap();
            let targets = (chunk == 1).then(|| {
                for (frame, skip) in want.iter_mut().zip(&skip) {
                    for (value, skip) in frame.data.iter_mut().zip(&skip.data) {
                        *value += skip;
                    }
                }
                skip.iter()
                    .map(|frame| upload_map(&gpu, frame).unwrap())
                    .collect()
            });
            let input = input
                .iter()
                .map(|frame| upload_map(&gpu, frame).unwrap())
                .collect();
            let got = kernels
                .causal(&weights, &mut device, input, targets)
                .unwrap();
            check(&gpu, &got, &want, 2e-3);
        }
    }

    /// Shapes a malformed checkpoint could bring are refused before any
    /// upload or launch.
    #[test]
    fn malformed_wan_shapes_are_refused() {
        let gpu = ImageGpu::new(0).unwrap();
        let kernels = WanKernels::new(&gpu).unwrap();
        let refused = |result: Result<(), NetworkError>| {
            assert!(
                matches!(result, Err(NetworkError::InvalidConfig(_))),
                "{result:?}"
            )
        };

        let mut frame = upload_map(&gpu, &map(8, 3, 4, 1)).unwrap();
        let short = gpu.upload_f32(&[1.0; 7]).unwrap();
        refused(kernels.rms(&mut frame, &short, true));

        let shortcut = DupUp {
            out_channels: 4,
            time: 2,
            side: 2,
        };
        let input = [upload_map(&gpu, &map(16, 3, 4, 2)).unwrap()];
        for (channels, height, width) in [(5, 6, 8), (4, 6, 9), (4, 7, 8)] {
            let mut output = [
                upload_map(&gpu, &FeatureMap::new(channels, height, width)).unwrap(),
                upload_map(&gpu, &FeatureMap::new(channels, height, width)).unwrap(),
            ];
            refused(kernels.dup_up_add(shortcut, &input, &mut output, false));
        }
        // 12 channels do not repeat into 4 x 8.
        let input = [upload_map(&gpu, &map(12, 3, 4, 3)).unwrap()];
        let mut output = [
            upload_map(&gpu, &FeatureMap::new(4, 6, 8)).unwrap(),
            upload_map(&gpu, &FeatureMap::new(4, 6, 8)).unwrap(),
        ];
        refused(kernels.dup_up_add(shortcut, &input, &mut output, false));

        let conv = |in_channels: usize, taps: usize| CausalConv3d {
            conv: Conv2d::new(Matrix::new(2, in_channels * 9), None, in_channels, 3, 1, 1).unwrap(),
            taps,
        };
        refused(upload_all(&gpu, &conv(7, 3)).map(drop));
        refused(upload_all(&gpu, &conv(6, 0)).map(drop));
        // Three taps over 6 channels read 2 a frame, so each norm needs 2.
        let mut resnet = WanResnet {
            norm1: vec![1.0; 2],
            conv1: conv(6, 3),
            norm2: vec![1.0; 2],
            conv2: conv(6, 3),
            shortcut: None,
        };
        assert!(upload_wan_resnet(&gpu, &resnet).is_ok());
        resnet.norm2.push(1.0);
        refused(upload_wan_resnet(&gpu, &resnet).map(drop));
    }
}
