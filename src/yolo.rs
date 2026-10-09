//! YOLO11, ultralytics' `yolo11.yaml`, as a graph over the layers in
//! [`crate::conv`]: trainable on the host and on a device, with every tensor
//! named as in an ultralytics state dict so trained weights load directly.
//!
//! The graph ends at the raw Detect maps, one per stride 8, 16 and 32, each
//! the 64 box-distribution logits then the class logits per pixel: what
//! [`crate::detection`]'s losses and decoding take.

use crate::conv::{
    BatchNormCache, ConvCache, DepthwiseCache, DepthwiseConv2d, ImageBatch, MaxPoolCache,
    PixelAttentionCache, PixelAttentionShape, TrainableBatchNorm2d, TrainableConv2d,
    concat_channels, max_pool_same, max_pool_same_backward, pixel_attention,
    pixel_attention_backward, split_channels, upsample_nearest_backward, upsample_nearest_batch,
};
use crate::ffn::{silu, silu_derivative};
use crate::network::NetworkError;
use crate::param::Param;
use crate::safetensors::SafeTensors;
use rand::rngs::StdRng;

/// The box branch's distribution bins per side.
pub const REG_MAX: usize = 16;

/// A row of `yolo11.yaml`'s `scales`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Yolo11Scale {
    pub depth: f32,
    pub width: f32,
    pub max_channels: usize,
}

impl Yolo11Scale {
    pub const S: Self = Self {
        depth: 0.5,
        width: 0.5,
        max_channels: 1024,
    };
}

/// The convolution inside ultralytics' `Conv` or `DWConv`.
// A graph holds a few hundred nodes; boxing saves nothing that matters.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug)]
pub enum Kernel {
    Dense(TrainableConv2d),
    Depthwise(DepthwiseConv2d),
}

// A graph holds a few hundred nodes; boxing saves nothing that matters.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug)]
pub enum Op {
    Input,
    /// ultralytics' `Conv` or `DWConv`: a convolution without bias, batch
    /// norm, then SiLU under `act`.
    Conv {
        kernel: Kernel,
        norm: TrainableBatchNorm2d,
        act: bool,
    },
    /// A plain `nn.Conv2d` with bias, the Detect branches' last layer.
    Plain(TrainableConv2d),
    /// A dense [`Op::Conv`] with its evaluating norm folded into the weights
    /// and bias, from [`Yolo11::fuse`]. Inference only. `narrow` caches the
    /// weight as bf16 for a mixed-precision device, made on the first device
    /// forward pass.
    // NOTE: the cache is not invalidated, which holds while a fused model's weights never change.
    Fused {
        conv: TrainableConv2d,
        act: bool,
        #[cfg(feature = "cuda")]
        narrow: Option<cudarc::driver::CudaSlice<u8>>,
    },
    Add,
    Concat,
    /// `width` channels from `offset`.
    Slice {
        offset: usize,
        width: usize,
    },
    /// Nearest, by two.
    Upsample,
    /// Stride 1, same padding.
    MaxPool(usize),
    /// The attended map then the values, `2 * shape.dim()` channels.
    Attention(PixelAttentionShape),
}

#[derive(Clone, Debug)]
pub struct Node {
    /// The ultralytics module path, `model.10.m.0.attn.qkv`; empty for nodes
    /// without parameters.
    pub name: String,
    pub op: Op,
    pub inputs: Vec<usize>,
    pub channels: usize,
}

/// YOLO11 with its Detect head. Node 0 is the input image.
#[derive(Clone, Debug)]
pub struct Yolo11 {
    pub nodes: Vec<Node>,
    /// The raw Detect maps at strides 8, 16 and 32.
    pub outputs: [usize; 3],
    pub classes: usize,
}

struct Builder<'a> {
    nodes: Vec<Node>,
    rng: &'a mut StdRng,
}

type Built = Result<usize, NetworkError>;

impl Builder<'_> {
    fn push(&mut self, name: String, op: Op, inputs: Vec<usize>, channels: usize) -> usize {
        self.nodes.push(Node {
            name,
            op,
            inputs,
            channels,
        });
        self.nodes.len() - 1
    }

    fn ch(&self, node: usize) -> usize {
        self.nodes[node].channels
    }

    fn conv(
        &mut self,
        name: String,
        from: usize,
        c2: usize,
        k: usize,
        s: usize,
        act: bool,
    ) -> Built {
        let conv = TrainableConv2d::new(self.ch(from), c2, k, s, self.rng)?.without_bias();
        let op = Op::Conv {
            kernel: Kernel::Dense(conv),
            norm: TrainableBatchNorm2d::yolo(c2),
            act,
        };
        Ok(self.push(name, op, vec![from], c2))
    }

    fn dwconv(&mut self, name: String, from: usize, act: bool) -> usize {
        let c = self.ch(from);
        let op = Op::Conv {
            kernel: Kernel::Depthwise(DepthwiseConv2d::new(c, 3, 1, self.rng)),
            norm: TrainableBatchNorm2d::yolo(c),
            act,
        };
        self.push(name, op, vec![from], c)
    }

    fn plain(&mut self, name: String, from: usize, c2: usize) -> Built {
        let conv = TrainableConv2d::new(self.ch(from), c2, 1, 1, self.rng)?;
        Ok(self.push(name, Op::Plain(conv), vec![from], c2))
    }

    fn op(&mut self, op: Op, inputs: Vec<usize>, channels: usize) -> usize {
        self.push(String::new(), op, inputs, channels)
    }

    fn add(&mut self, a: usize, b: usize) -> usize {
        self.op(Op::Add, vec![a, b], self.ch(a))
    }

    fn concat(&mut self, parts: Vec<usize>) -> usize {
        let channels = parts.iter().map(|&p| self.ch(p)).sum();
        self.op(Op::Concat, parts, channels)
    }

    fn halves(&mut self, from: usize) -> [usize; 2] {
        let width = self.ch(from) / 2;
        [0, width].map(|offset| self.op(Op::Slice { offset, width }, vec![from], width))
    }

    fn bottleneck(
        &mut self,
        name: &str,
        from: usize,
        c2: usize,
        shortcut: bool,
        k: [usize; 2],
        e: f32,
    ) -> Built {
        let hidden = (c2 as f32 * e) as usize;
        let a = self.conv(format!("{name}.cv1"), from, hidden, k[0], 1, true)?;
        let b = self.conv(format!("{name}.cv2"), a, c2, k[1], 1, true)?;
        Ok(match shortcut && self.ch(from) == c2 {
            true => self.add(from, b),
            false => b,
        })
    }

    fn c3k(&mut self, name: &str, from: usize, c2: usize, n: usize, shortcut: bool) -> Built {
        let hidden = c2 / 2;
        let mut a = self.conv(format!("{name}.cv1"), from, hidden, 1, 1, true)?;
        for i in 0..n {
            a = self.bottleneck(&format!("{name}.m.{i}"), a, hidden, shortcut, [3, 3], 1.0)?;
        }
        let b = self.conv(format!("{name}.cv2"), from, hidden, 1, 1, true)?;
        let joined = self.concat(vec![a, b]);
        self.conv(format!("{name}.cv3"), joined, c2, 1, 1, true)
    }

    fn c3k2(&mut self, name: &str, from: usize, c2: usize, n: usize, c3k: bool, e: f32) -> Built {
        let c = (c2 as f32 * e) as usize;
        let y = self.conv(format!("{name}.cv1"), from, 2 * c, 1, 1, true)?;
        let mut parts = self.halves(y).to_vec();
        for i in 0..n {
            let last = parts[parts.len() - 1];
            let m = format!("{name}.m.{i}");
            parts.push(match c3k {
                true => self.c3k(&m, last, c, 2, true)?,
                false => self.bottleneck(&m, last, c, true, [3, 3], 0.5)?,
            });
        }
        let joined = self.concat(parts);
        self.conv(format!("{name}.cv2"), joined, c2, 1, 1, true)
    }

    /// The YOLO11 checkpoints' SPPF, whose first convolution keeps its SiLU.
    fn sppf(&mut self, name: &str, from: usize, c2: usize) -> Built {
        let a = self.conv(format!("{name}.cv1"), from, self.ch(from) / 2, 1, 1, true)?;
        let mut parts = vec![a];
        for _ in 0..3 {
            let last = parts[parts.len() - 1];
            parts.push(self.op(Op::MaxPool(5), vec![last], self.ch(a)));
        }
        let joined = self.concat(parts);
        self.conv(format!("{name}.cv2"), joined, c2, 1, 1, true)
    }

    fn attention(&mut self, name: &str, from: usize, heads: usize) -> Built {
        let dim = self.ch(from);
        let shape = PixelAttentionShape::yolo(dim, heads);
        let width = dim + 2 * shape.key_dim * heads;
        let qkv = self.conv(format!("{name}.qkv"), from, width, 1, 1, false)?;
        let both = self.op(Op::Attention(shape), vec![qkv], 2 * dim);
        let [attended, values] = self.halves(both);
        let pe = self.dwconv(format!("{name}.pe"), values, false);
        let sum = self.add(attended, pe);
        self.conv(format!("{name}.proj"), sum, dim, 1, 1, false)
    }

    fn psa_block(&mut self, name: &str, from: usize, heads: usize) -> Built {
        let c = self.ch(from);
        let a = self.attention(&format!("{name}.attn"), from, heads)?;
        let x = self.add(from, a);
        let f = self.conv(format!("{name}.ffn.0"), x, 2 * c, 1, 1, true)?;
        let f = self.conv(format!("{name}.ffn.1"), f, c, 1, 1, false)?;
        Ok(self.add(x, f))
    }

    fn c2psa(&mut self, name: &str, from: usize, n: usize) -> Built {
        let c1 = self.ch(from);
        let y = self.conv(format!("{name}.cv1"), from, c1, 1, 1, true)?;
        let [a, mut b] = self.halves(y);
        for i in 0..n {
            b = self.psa_block(&format!("{name}.m.{i}"), b, (c1 / 2 / 64).max(1))?;
        }
        let joined = self.concat(vec![a, b]);
        self.conv(format!("{name}.cv2"), joined, c1, 1, 1, true)
    }

    fn detect(
        &mut self,
        name: &str,
        feats: [usize; 3],
        classes: usize,
    ) -> Result<[usize; 3], NetworkError> {
        let first = self.ch(feats[0]);
        let c2 = 16.max(first / 4).max(4 * REG_MAX);
        let c3 = first.max(classes.min(100));
        let mut outputs = [0; 3];
        for (i, &f) in feats.iter().enumerate() {
            let b = self.conv(format!("{name}.cv2.{i}.0"), f, c2, 3, 1, true)?;
            let b = self.conv(format!("{name}.cv2.{i}.1"), b, c2, 3, 1, true)?;
            let b = self.plain(format!("{name}.cv2.{i}.2"), b, 4 * REG_MAX)?;
            let c = self.dwconv(format!("{name}.cv3.{i}.0.0"), f, true);
            let c = self.conv(format!("{name}.cv3.{i}.0.1"), c, c3, 1, 1, true)?;
            let c = self.dwconv(format!("{name}.cv3.{i}.1.0"), c, true);
            let c = self.conv(format!("{name}.cv3.{i}.1.1"), c, c3, 1, 1, true)?;
            let c = self.plain(format!("{name}.cv3.{i}.2"), c, classes)?;
            // Detect.bias_init: boxes start at one, classes at a prior of
            // five objects per 640 image.
            let prior = (5.0 / classes as f32 / (640.0 / STRIDES[i]).powi(2)).ln();
            for (node, value) in [(b, 1.0), (c, prior)] {
                if let Op::Plain(conv) = &mut self.nodes[node].op {
                    conv.bias.value.data.fill(value);
                }
            }
            outputs[i] = self.concat(vec![b, c]);
        }
        Ok(outputs)
    }
}

/// What [`Yolo11::backward`] needs from [`Yolo11::forward`]: every node's
/// output and cache.
pub struct Tape {
    pub values: Vec<ImageBatch>,
    caches: Vec<Cache>,
}

enum Cache {
    None,
    Conv {
        kernel: KernelCache,
        norm: BatchNormCache,
        /// The normalized map before SiLU, under `act`.
        pre: Option<ImageBatch>,
    },
    Plain(ConvCache),
    MaxPool(MaxPoolCache),
    Attention(PixelAttentionCache),
}

enum KernelCache {
    Dense(ConvCache),
    Depthwise(DepthwiseCache),
}

/// The `width` channels from `offset` of a map with `channels` channels, as
/// the widths [`split_channels`] takes, and which part is the slice.
fn slice_widths(channels: usize, offset: usize, width: usize) -> (Vec<usize>, usize) {
    let widths: Vec<usize> = [offset, width, channels - offset - width]
        .into_iter()
        .filter(|&w| w > 0)
        .collect();
    (widths, usize::from(offset > 0))
}

/// A PyTorch tensor in this crate's layout: convolution weights go from
/// `[out, in, ky, kx]` to the `[out, ky, kx, in]` that
/// [`TrainableConv2d`]'s columns take; everything else is unchanged.
fn torch_layout(data: Vec<f32>, shape: &[usize]) -> Vec<f32> {
    let &[_, inputs, kh, kw] = shape else {
        return data;
    };
    let taps = kh * kw;
    let mut moved = vec![0.0; data.len()];
    for (index, v) in data.into_iter().enumerate() {
        let (out, input, tap) = (index / (inputs * taps), index / taps % inputs, index % taps);
        moved[(out * taps + tap) * inputs + input] = v;
    }
    moved
}

/// The inverse of [`torch_layout`].
fn torch_order(data: &[f32], shape: &[usize]) -> Vec<f32> {
    let &[_, inputs, kh, kw] = shape else {
        return data.to_vec();
    };
    let taps = kh * kw;
    (0..data.len())
        .map(|index| {
            let (out, input, tap) = (index / (inputs * taps), index / taps % inputs, index % taps);
            data[(out * taps + tap) * inputs + input]
        })
        .collect()
}

impl Yolo11 {
    /// `yolo11.yaml` at `scale` with `classes` classes, initialised at random.
    pub fn new(scale: Yolo11Scale, classes: usize, rng: &mut StdRng) -> Result<Self, NetworkError> {
        let ch =
            |c: usize| ((c.min(scale.max_channels) as f32 * scale.width / 8.0).ceil() as usize) * 8;
        let n = ((2.0 * scale.depth).round() as usize).max(1);
        // NOTE: the m, l and x scales force C3k inside every C3k2;
        // only n and s are built here.
        let mut g = Builder {
            nodes: Vec::new(),
            rng,
        };
        let input = g.op(Op::Input, Vec::new(), 3);
        let l0 = g.conv("model.0".into(), input, ch(64), 3, 2, true)?;
        let l1 = g.conv("model.1".into(), l0, ch(128), 3, 2, true)?;
        let l2 = g.c3k2("model.2", l1, ch(256), n, false, 0.25)?;
        let l3 = g.conv("model.3".into(), l2, ch(256), 3, 2, true)?;
        let l4 = g.c3k2("model.4", l3, ch(512), n, false, 0.25)?;
        let l5 = g.conv("model.5".into(), l4, ch(512), 3, 2, true)?;
        let l6 = g.c3k2("model.6", l5, ch(512), n, true, 0.5)?;
        let l7 = g.conv("model.7".into(), l6, ch(1024), 3, 2, true)?;
        let l8 = g.c3k2("model.8", l7, ch(1024), n, true, 0.5)?;
        let l9 = g.sppf("model.9", l8, ch(1024))?;
        let l10 = g.c2psa("model.10", l9, n)?;
        let l11 = g.op(Op::Upsample, vec![l10], g.ch(l10));
        let l12 = g.concat(vec![l11, l6]);
        let l13 = g.c3k2("model.13", l12, ch(512), n, false, 0.5)?;
        let l14 = g.op(Op::Upsample, vec![l13], g.ch(l13));
        let l15 = g.concat(vec![l14, l4]);
        let l16 = g.c3k2("model.16", l15, ch(256), n, false, 0.5)?;
        let l17 = g.conv("model.17".into(), l16, ch(256), 3, 2, true)?;
        let l18 = g.concat(vec![l17, l13]);
        let l19 = g.c3k2("model.19", l18, ch(512), n, false, 0.5)?;
        let l20 = g.conv("model.20".into(), l19, ch(512), 3, 2, true)?;
        let l21 = g.concat(vec![l20, l10]);
        let l22 = g.c3k2("model.22", l21, ch(1024), n, true, 0.5)?;
        let outputs = g.detect("model.23", [l16, l19, l22], classes)?;
        Ok(Self {
            nodes: g.nodes,
            outputs,
            classes,
        })
    }

    /// Every parameter by its state-dict name. A batch norm's running
    /// statistics come as `<module>.bn.running`, both rows in one.
    pub fn state_mut(&mut self) -> Vec<(String, &mut Param)> {
        let mut state = Vec::new();
        for node in &mut self.nodes {
            let name = &node.name;
            match &mut node.op {
                Op::Conv { kernel, norm, .. } => {
                    let weight = match kernel {
                        Kernel::Dense(conv) => &mut conv.weight.weight,
                        Kernel::Depthwise(conv) => &mut conv.weight,
                    };
                    state.push((format!("{name}.conv.weight"), weight));
                    state.push((format!("{name}.bn.weight"), &mut norm.gamma));
                    state.push((format!("{name}.bn.bias"), &mut norm.beta));
                    state.push((format!("{name}.bn.running"), &mut norm.running));
                }
                Op::Plain(conv) => {
                    state.push((format!("{name}.weight"), &mut conv.weight.weight));
                    state.push((format!("{name}.bias"), &mut conv.bias));
                }
                Op::Fused { conv, .. } => {
                    state.push((format!("{name}.conv.weight"), &mut conv.weight.weight));
                    state.push((format!("{name}.conv.bias"), &mut conv.bias));
                }
                _ => {}
            }
        }
        state
    }

    /// Has every dense convolution keep its input rather than its `im2col`
    /// columns for the backward pass, which rebuilds them: at a 1280 input
    /// the columns are most of a training step's memory.
    pub fn set_recompute_columns(&mut self, recompute: bool) {
        for node in &mut self.nodes {
            match &mut node.op {
                Op::Conv {
                    kernel: Kernel::Dense(conv),
                    ..
                }
                | Op::Plain(conv) => conv.set_recompute_columns(recompute),
                _ => {}
            }
        }
    }

    /// This model for inference: every dense convolution with its batch
    /// norm's running statistics folded in, as ultralytics' `fuse()`. Needs
    /// the parameters on the host; the result only infers.
    pub fn fuse(&self) -> Self {
        let mut fused = self.clone();
        for node in &mut fused.nodes {
            if let Op::Conv {
                kernel: Kernel::Dense(conv),
                norm,
                act,
            } = &node.op
            {
                node.op = Op::Fused {
                    conv: norm.fuse_into(conv),
                    act: *act,
                    #[cfg(feature = "cuda")]
                    narrow: None,
                };
            }
        }
        fused
    }

    /// The parameters an optimizer steps.
    pub fn params_mut(&mut self) -> Vec<&mut Param> {
        self.state_mut()
            .into_iter()
            .filter(|(name, _)| !name.ends_with(".running"))
            .map(|(_, param)| param)
            .collect()
    }

    /// Reads an ultralytics state dict saved as safetensors, onto the host.
    pub fn load_ultralytics(&mut self, tensors: &mut SafeTensors) -> Result<(), NetworkError> {
        for (name, param) in self.state_mut() {
            let data = match name.strip_suffix(".running") {
                Some(bn) => {
                    let (mut mean, _) = tensors.tensor(&format!("{bn}.running_mean"))?;
                    mean.extend(tensors.tensor(&format!("{bn}.running_var"))?.0);
                    mean
                }
                None => {
                    let (data, shape) = tensors.tensor(&name)?;
                    torch_layout(data, &shape)
                }
            };
            if data.len() != param.value.data.len() {
                return Err(NetworkError::InvalidConfig(format!(
                    "{name} holds {} values where this graph has {}",
                    data.len(),
                    param.value.data.len()
                )));
            }
            param.value.data.copy_from_slice(&data);
        }
        Ok(())
    }

    /// Writes the model as an ultralytics state dict in safetensors, which
    /// [`Yolo11::load_ultralytics`] reads back. Needs the parameters on the
    /// host.
    pub fn save_ultralytics(&self, path: impl AsRef<std::path::Path>) -> Result<(), NetworkError> {
        let mut tensors: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
        for node in &self.nodes {
            let (name, c) = (&node.name, node.channels);
            // A weight's square kernel from its size and input channels.
            let mut conv = |key: String, weight: &Param, inputs: usize| {
                let k = ((weight.value.data.len() / (c * inputs)) as f64).sqrt() as usize;
                let shape = vec![c, inputs, k, k];
                tensors.push((key, shape.clone(), torch_order(&weight.value.data, &shape)));
            };
            match &node.op {
                Op::Conv { kernel, .. } => match kernel {
                    Kernel::Dense(dense) => conv(
                        format!("{name}.conv.weight"),
                        &dense.weight.weight,
                        dense.in_channels(),
                    ),
                    Kernel::Depthwise(depthwise) => {
                        conv(format!("{name}.conv.weight"), &depthwise.weight, 1)
                    }
                },
                Op::Plain(dense) => conv(
                    format!("{name}.weight"),
                    &dense.weight.weight,
                    dense.in_channels(),
                ),
                Op::Fused { conv: dense, .. } => conv(
                    format!("{name}.conv.weight"),
                    &dense.weight.weight,
                    dense.in_channels(),
                ),
                _ => {}
            }
            let mut vector =
                |key: String, values: &[f32]| tensors.push((key, vec![c], values.to_vec()));
            match &node.op {
                Op::Conv { norm, .. } => {
                    let running = &norm.running.value.data;
                    vector(format!("{name}.bn.weight"), &norm.gamma.value.data);
                    vector(format!("{name}.bn.bias"), &norm.beta.value.data);
                    vector(format!("{name}.bn.running_mean"), &running[..c]);
                    vector(format!("{name}.bn.running_var"), &running[c..]);
                }
                Op::Plain(dense) => vector(format!("{name}.bias"), &dense.bias.value.data),
                Op::Fused { conv: dense, .. } => {
                    vector(format!("{name}.conv.bias"), &dense.bias.value.data)
                }
                _ => {}
            }
        }
        let refs: Vec<(&str, &[usize], &[f32])> = tensors
            .iter()
            .map(|(name, shape, data)| (name.as_str(), &shape[..], &data[..]))
            .collect();
        crate::safetensors::write(path, &refs, &Default::default())
    }

    /// As [`Yolo11::load_ultralytics`] for fine-tuning from a checkpoint
    /// with other classes: tensors missing or of another size are left at
    /// their initialisation, as ultralytics' `intersect_dicts` leaves them.
    /// Returns their names.
    pub fn transfer_ultralytics(
        &mut self,
        tensors: &mut SafeTensors,
    ) -> Result<Vec<String>, NetworkError> {
        let mut skipped = Vec::new();
        for (name, param) in self.state_mut() {
            let data = match name.strip_suffix(".running") {
                Some(bn) => {
                    tensors
                        .tensor(&format!("{bn}.running_mean"))
                        .and_then(|(mut mean, _)| {
                            mean.extend(tensors.tensor(&format!("{bn}.running_var"))?.0);
                            Ok(mean)
                        })
                }
                None => tensors
                    .tensor(&name)
                    .map(|(data, shape)| torch_layout(data, &shape)),
            };
            match data {
                Ok(data) if data.len() == param.value.data.len() => {
                    param.value.data.copy_from_slice(&data)
                }
                _ => skipped.push(name),
            }
        }
        Ok(skipped)
    }

    /// Runs the graph, batch norms on batch statistics under `training`.
    pub fn forward(&mut self, input: &ImageBatch, training: bool) -> Result<Tape, NetworkError> {
        let mut values: Vec<ImageBatch> = Vec::with_capacity(self.nodes.len());
        let mut caches = Vec::with_capacity(self.nodes.len());
        for node in &mut self.nodes {
            let x = node.inputs.first().map(|&i| &values[i]);
            let (out, cache) = match &mut node.op {
                Op::Input => (input.clone(), Cache::None),
                Op::Conv { kernel, norm, act } => {
                    let x = x.unwrap();
                    let (y, kernel) = match kernel {
                        Kernel::Dense(conv) => {
                            let (y, c) = conv.forward(x)?;
                            (y, KernelCache::Dense(c))
                        }
                        Kernel::Depthwise(conv) => {
                            let (y, c) = conv.forward(x)?;
                            (y, KernelCache::Depthwise(c))
                        }
                    };
                    let (y, norm) = norm.forward(&y, training)?;
                    match act {
                        true => {
                            let mut out = y.clone();
                            out.tokens.data.iter_mut().for_each(|v| *v = silu(*v));
                            (
                                out,
                                Cache::Conv {
                                    kernel,
                                    norm,
                                    pre: Some(y),
                                },
                            )
                        }
                        false => (
                            y,
                            Cache::Conv {
                                kernel,
                                norm,
                                pre: None,
                            },
                        ),
                    }
                }
                Op::Plain(conv) => {
                    let (y, c) = conv.forward(x.unwrap())?;
                    (y, Cache::Plain(c))
                }
                Op::Fused { conv, act, .. } => {
                    let (mut y, _) = conv.forward(x.unwrap())?;
                    if *act {
                        y.tokens.data.iter_mut().for_each(|v| *v = silu(*v));
                    }
                    (y, Cache::None)
                }
                Op::Add => {
                    let mut y = x.unwrap().clone();
                    y.add(&values[node.inputs[1]]);
                    (y, Cache::None)
                }
                Op::Concat => {
                    let parts: Vec<&ImageBatch> = node.inputs.iter().map(|&i| &values[i]).collect();
                    (concat_channels(&parts), Cache::None)
                }
                Op::Slice { offset, width } => {
                    let x = x.unwrap();
                    let (widths, part) = slice_widths(x.channels(), *offset, *width);
                    (split_channels(x, &widths).swap_remove(part), Cache::None)
                }
                Op::Upsample => (upsample_nearest_batch(x.unwrap(), 2), Cache::None),
                Op::MaxPool(k) => {
                    let (y, c) = max_pool_same(x.unwrap(), *k);
                    (y, Cache::MaxPool(c))
                }
                Op::Attention(shape) => {
                    let (attended, values, c) = pixel_attention(x.unwrap(), *shape)?;
                    (concat_channels(&[&attended, &values]), Cache::Attention(c))
                }
            };
            values.push(out);
            caches.push(cache);
        }
        Ok(Tape { values, caches })
    }

    /// The raw Detect maps of a forward pass.
    pub fn outputs<'a>(&self, tape: &'a Tape) -> [&'a ImageBatch; 3] {
        self.outputs.map(|i| &tape.values[i])
    }

    /// Accumulates every parameter's gradient from the Detect maps'
    /// gradients and returns the input's.
    pub fn backward(
        &mut self,
        tape: Tape,
        grads: [&ImageBatch; 3],
    ) -> Result<ImageBatch, NetworkError> {
        let Tape { values, caches } = tape;
        let mut grad: Vec<Option<ImageBatch>> = vec![None; self.nodes.len()];
        for (&i, g) in self.outputs.iter().zip(grads) {
            grad[i] = Some(g.clone());
        }
        for (i, cache) in caches.into_iter().enumerate().skip(1).rev() {
            let Some(g) = grad[i].take() else {
                continue;
            };
            let node = &mut self.nodes[i];
            let input = |k: usize| &values[node.inputs[k]];
            let input_grads = match (&mut node.op, cache) {
                (
                    Op::Conv { kernel, norm, .. },
                    Cache::Conv {
                        kernel: kc,
                        norm: nc,
                        pre,
                    },
                ) => {
                    let mut g = g;
                    if let Some(pre) = pre {
                        g.tokens
                            .data
                            .iter_mut()
                            .zip(&pre.tokens.data)
                            .for_each(|(g, p)| *g *= silu_derivative(*p));
                    }
                    let g = norm.backward(&nc, &g);
                    vec![match (kernel, kc) {
                        (Kernel::Dense(conv), KernelCache::Dense(c)) => conv.backward(&c, &g)?,
                        (Kernel::Depthwise(conv), KernelCache::Depthwise(c)) => {
                            conv.backward(&c, &g)
                        }
                        _ => unreachable!("a cache from another kernel"),
                    }]
                }
                (Op::Plain(conv), Cache::Plain(c)) => vec![conv.backward(&c, &g)?],
                (Op::Add, _) => vec![g.clone(), g],
                (Op::Concat, _) => {
                    let widths: Vec<usize> = (0..node.inputs.len())
                        .map(|k| input(k).channels())
                        .collect();
                    split_channels(&g, &widths)
                }
                (Op::Slice { offset, .. }, _) => {
                    let channels = input(0).channels();
                    let before = g.like(*offset);
                    let after = g.like(channels - *offset - g.channels());
                    let parts: Vec<&ImageBatch> = [&before, &g, &after]
                        .into_iter()
                        .filter(|p| p.channels() > 0)
                        .collect();
                    vec![concat_channels(&parts)]
                }
                (Op::Upsample, _) => vec![upsample_nearest_backward(&g, 2)],
                (Op::MaxPool(_), Cache::MaxPool(c)) => vec![max_pool_same_backward(&c, &g)],
                (Op::Attention(shape), Cache::Attention(c)) => {
                    let halves = split_channels(&g, &[shape.dim(), shape.dim()]);
                    vec![pixel_attention_backward(&c, &halves[0], &halves[1])]
                }
                (Op::Fused { .. }, _) => {
                    return Err(NetworkError::InvalidConfig(
                        "a fused model only infers".into(),
                    ));
                }
                _ => unreachable!("a cache from another op"),
            };
            for (&j, g) in node.inputs.iter().zip(input_grads) {
                match &mut grad[j] {
                    Some(sum) => sum.add(&g),
                    slot => *slot = Some(g),
                }
            }
        }
        Ok(grad[0].take().expect("the input feeds the graph"))
    }
}

/// The Detect maps' strides.
pub const STRIDES: [f32; 3] = [8.0, 16.0, 32.0];

/// Every anchor's box, centre x, centre y, width and height in input pixels,
/// and its `classes` sigmoid scores, image by image: ultralytics' Detect
/// inference on the raw maps of [`Yolo11::outputs`], ready for
/// [`crate::detection::non_max_suppression`].
pub fn decode(maps: [&ImageBatch; 3], classes: usize) -> Vec<(Vec<[f32; 4]>, Vec<f32>)> {
    let sizes: Vec<(usize, usize)> = maps.iter().map(|m| (m.height, m.width)).collect();
    let (points, strides) = crate::detection::make_anchors(&sizes, &STRIDES);
    (0..maps[0].batch)
        .map(|image| {
            let rows = maps.iter().flat_map(|m| {
                let pixels = m.height * m.width;
                (image * pixels..(image + 1) * pixels).map(|r| m.tokens.row(r))
            });
            let mut boxes = Vec::with_capacity(points.len());
            let mut scores = Vec::with_capacity(points.len() * classes);
            for ((row, &[px, py]), &stride) in rows.zip(&points).zip(&strides) {
                let [l, t, r, b] = std::array::from_fn(|side| {
                    let bins = &row[side * REG_MAX..(side + 1) * REG_MAX];
                    let max = bins.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let (mut sum, mut mean) = (0.0, 0.0);
                    for (k, &v) in bins.iter().enumerate() {
                        let e = (v - max).exp();
                        sum += e;
                        mean += e * k as f32;
                    }
                    mean / sum * stride
                });
                boxes.push([px + (r - l) / 2.0, py + (b - t) / 2.0, l + r, t + b]);
                scores.extend(row[4 * REG_MAX..].iter().map(|&v| 1.0 / (1.0 + (-v).exp())));
            }
            (boxes, scores)
        })
        .collect()
}

/// ultralytics' default loss gains: box, class, distribution focal.
pub const LOSS_GAINS: [f32; 3] = [7.5, 0.5, 1.5];

/// One labelled box: its class and `[x1, y1, x2, y2]` in input pixels.
pub type Target = (usize, [f32; 4]);

/// ultralytics' `v8DetectionLoss` on the raw maps of [`Yolo11::outputs`].
/// `targets` holds each image's labelled boxes. Returns the box, class and
/// distribution focal terms after [`LOSS_GAINS`] (what ultralytics logs) and
/// the gradient of their sum times the batch size (what it backpropagates)
/// with respect to each map.
pub fn detection_loss(
    maps: [&ImageBatch; 3],
    classes: usize,
    targets: &[Vec<Target>],
) -> ([f32; 3], [ImageBatch; 3]) {
    use crate::detection::{
        Overlap, TaskAlignedAssigner, box_iou, box_to_ltrb, distribution_focal_loss, make_anchors,
    };
    use rayon::prelude::*;
    let batch = maps[0].batch;
    assert_eq!(targets.len(), batch, "one target list per image");
    let sizes: Vec<(usize, usize)> = maps.iter().map(|m| (m.height, m.width)).collect();
    let (points, strides) = make_anchors(&sizes, &STRIDES);
    let anchors = points.len();
    let channels = 4 * REG_MAX + classes;
    // Anchor `a` of an image: its map and the pixel in it, maps in order.
    let ends: Vec<usize> = sizes
        .iter()
        .scan(0, |end, (h, w)| {
            *end += h * w;
            Some(*end)
        })
        .collect();
    let row = |image: usize, a: usize| -> &[f32] {
        let i = ends.iter().position(|&end| a < end).expect("an anchor");
        let pixels = sizes[i].0 * sizes[i].1;
        maps[i].tokens.row(image * pixels + a + pixels - ends[i])
    };
    // The predicted distances in grid units, each side's softmax, and the box
    // they make in grid units.
    let decode = |a: usize, row: &[f32]| -> ([f32; 4], [f32; 4], [f32; 4 * REG_MAX]) {
        let mut probs = [0.0f32; 4 * REG_MAX];
        let ltrb: [f32; 4] = std::array::from_fn(|side| {
            let bins = &row[side * REG_MAX..(side + 1) * REG_MAX];
            let max = bins.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = bins.iter().map(|&v| (v - max).exp()).sum();
            let mut mean = 0.0;
            for (k, &v) in bins.iter().enumerate() {
                let p = (v - max).exp() / sum;
                probs[side * REG_MAX + k] = p;
                mean += p * k as f32;
            }
            mean
        });
        let s = strides[a];
        let [px, py] = [points[a][0] / s, points[a][1] / s];
        let grid = [px - ltrb[0], py - ltrb[1], px + ltrb[2], py + ltrb[3]];
        (grid, ltrb, probs)
    };

    let assigner = TaskAlignedAssigner::yolo(&STRIDES);
    // Each image decodes its anchors in parallel and assigns on its own.
    let assignments: Vec<_> = targets
        .par_iter()
        .enumerate()
        .map(|(image, labelled)| {
            let mut scores = vec![0.0f32; anchors * classes];
            let mut pixel_boxes = vec![[0.0f32; 4]; anchors];
            scores
                .par_chunks_mut(classes)
                .zip(pixel_boxes.par_iter_mut())
                .enumerate()
                .for_each(|(a, (scores, pixel_box))| {
                    let row = row(image, a);
                    *pixel_box = decode(a, row).0.map(|v| v * strides[a]);
                    for (score, &v) in scores.iter_mut().zip(&row[4 * REG_MAX..]) {
                        *score = 1.0 / (1.0 + (-v).exp());
                    }
                });
            let gt_classes: Vec<usize> = labelled.iter().map(|t| t.0).collect();
            let gt_boxes: Vec<[f32; 4]> = labelled.iter().map(|t| t.1).collect();
            assigner.assign(
                &scores,
                classes,
                &pixel_boxes,
                &points,
                &gt_classes,
                &gt_boxes,
            )
        })
        .collect();
    let score_sum: f32 = assignments
        .iter()
        .flat_map(|a| &a.scores)
        .sum::<f32>()
        .max(1.0);

    let scale = batch as f32 / score_sum;
    // Each map's gradient, one block an image, so an image's anchors write
    // their own three blocks.
    let mut grads: Vec<Vec<f32>> = sizes
        .iter()
        .map(|(h, w)| vec![0.0f32; batch * h * w * channels])
        .collect();
    let [g0, g1, g2] = &mut grads[..] else {
        unreachable!("three maps")
    };
    let block = |i: usize| sizes[i].0 * sizes[i].1 * channels;
    let per_image: Vec<[f32; 3]> = g0
        .par_chunks_mut(block(0))
        .zip(g1.par_chunks_mut(block(1)))
        .zip(g2.par_chunks_mut(block(2)))
        .zip(&assignments)
        .enumerate()
        .map(|(image, (((b0, b1), b2), assignment))| {
            let image_grads = [b0, b1, b2];
            let mut loss = [0.0f32; 3];
            for a in 0..anchors {
                let row = row(image, a);
                let i = ends.iter().position(|&end| a < end).expect("an anchor");
                let local = (a + sizes[i].0 * sizes[i].1 - ends[i]) * channels;
                let grad = &mut image_grads[i][local..local + channels];
                let targets = &assignment.scores[a * classes..(a + 1) * classes];
                for (c, &t) in targets.iter().enumerate() {
                    let x = row[4 * REG_MAX + c];
                    loss[1] += x.max(0.0) - x * t + (-x.abs()).exp().ln_1p();
                    grad[4 * REG_MAX + c] = LOSS_GAINS[1] * scale * (1.0 / (1.0 + (-x).exp()) - t);
                }
                if !assignment.foreground[a] {
                    continue;
                }
                let weight: f32 = targets.iter().sum();
                let s = strides[a];
                let target = assignment.boxes[a].map(|v| v / s);
                let (grid, ltrb, probs) = decode(a, row);
                let (iou, d_box) = box_iou(grid, target, Overlap::Ciou);
                loss[0] += (1.0 - iou) * weight;
                // d(1 - iou) through x1 = px - l, y1 = py - t, x2 = px + r, y2 = py + b.
                let d_ltrb = [d_box[0], d_box[1], -d_box[2], -d_box[3]]
                    .map(|d| LOSS_GAINS[0] * scale * weight * d);
                let point = [points[a][0] / s, points[a][1] / s];
                let target_ltrb = box_to_ltrb(point, target, (REG_MAX - 1) as f32);
                for side in 0..4 {
                    let bins = &row[side * REG_MAX..(side + 1) * REG_MAX];
                    let (dfl, d_dfl) = distribution_focal_loss(bins, target_ltrb[side]);
                    loss[2] += dfl * weight / 4.0;
                    for k in 0..REG_MAX {
                        let p = probs[side * REG_MAX + k];
                        grad[side * REG_MAX + k] += d_ltrb[side] * p * (k as f32 - ltrb[side])
                            + LOSS_GAINS[2] * scale * weight / 4.0 * d_dfl[k];
                    }
                }
            }
            loss
        })
        .collect();
    let loss: [f32; 3] = std::array::from_fn(|i| per_image.iter().map(|l| l[i]).sum());
    let loss = std::array::from_fn(|i| loss[i] / score_sum * LOSS_GAINS[i]);

    let mut grads = grads.into_iter();
    let grads = std::array::from_fn(|i| {
        let (h, w) = sizes[i];
        let data = grads.next().expect("three maps");
        ImageBatch::from_tokens(
            batch,
            h,
            w,
            crate::matrix::Matrix::from_vec(batch * h * w, channels, data),
        )
    });
    (loss, grads)
}

/// What [`validate`] measured.
#[derive(Clone, Copy, Debug)]
pub struct Validation {
    pub map50: f32,
    pub map50_95: f32,
    /// Per image: the upload, device forward pass and download, and everything.
    pub model_ms: f64,
    pub total_ms: f64,
    /// Per image by stage: upload, forward, download, decode, NMS. Each stage
    /// waits for the device before its clock stops.
    pub stages_ms: [f64; 5],
}

/// ultralytics' validator: images by aspect in rect batches of 16 at
/// `size` (stride 32, pad 0.5), each letterboxed without upscaling and run on
/// its own; conf 0.001, NMS IoU 0.7, at most 300 boxes. `images` come at a
/// long side of `size`, as [`crate::detection::DetectionImage::open`] gives
/// them.
#[cfg(feature = "cuda")]
pub fn validate(
    model: &mut Yolo11,
    device: &crate::param::CudaDevice,
    images: &[crate::detection::DetectionImage],
    size: usize,
) -> Result<Validation, NetworkError> {
    use crate::detection::{mean_average_precision, non_max_suppression, rect_shape};
    use std::time::{Duration, Instant};
    let aspect = |i: &crate::detection::DetectionImage| i.height as f32 / i.width as f32;
    let mut order: Vec<&_> = images.iter().collect();
    order.sort_by(|a, b| aspect(a).total_cmp(&aspect(b)));
    let mut results = Vec::new();
    let mut stages = [Duration::ZERO; 5];
    let context = device.context();
    for batch in order.chunks(16) {
        let aspects: Vec<f32> = batch.iter().map(|i| aspect(i)).collect();
        let [height, width] = rect_shape(&aspects, size, 32, 0.5);
        for image in batch {
            let (image, _, _) = image.letterbox(height, width, false);
            let input = input_batch(&[&image]);
            let mut clock = Instant::now();
            let mut lap = |stage: usize| {
                stages[stage] += clock.elapsed();
                clock = Instant::now();
            };
            let input = device.upload(&input)?;
            context.synchronize()?;
            lap(0);
            let tape = model.forward_device(&input, false)?;
            context.synchronize()?;
            lap(1);
            let maps = model
                .outputs_device(&tape)
                .map(|m| m.download())
                .into_iter()
                .collect::<Result<Vec<_>, _>>()?;
            lap(2);
            let (boxes, scores) = decode([&maps[0], &maps[1], &maps[2]], model.classes).remove(0);
            lap(3);
            let detections = non_max_suppression(&boxes, &scores, model.classes, 0.001, 0.7, 300);
            lap(4);
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
    let n = results.len().max(1) as f64;
    let stages_ms = stages.map(|d| d.as_secs_f64() * 1e3 / n);
    Ok(Validation {
        map50,
        map50_95,
        model_ms: stages_ms[..3].iter().sum(),
        total_ms: stages_ms.iter().sum(),
        stages_ms,
    })
}

/// Same-sized images as one channels-last batch scaled to `0..1`.
pub fn input_batch(images: &[&crate::detection::DetectionImage]) -> ImageBatch {
    let (height, width) = (images[0].height, images[0].width);
    let pixels = images
        .iter()
        .flat_map(|i| {
            assert_eq!((i.height, i.width), (height, width), "one size per batch");
            i.rgb.iter().map(|&v| v as f32 / 255.0)
        })
        .collect();
    ImageBatch::from_tokens(
        images.len(),
        height,
        width,
        crate::matrix::Matrix::from_vec(images.len() * height * width, 3, pixels),
    )
}

/// [`Tape`] on a device, from [`Yolo11::forward_device`].
#[cfg(feature = "cuda")]
pub struct DeviceTape {
    pub values: Vec<crate::conv::DeviceImageBatch>,
    caches: Vec<DeviceCache>,
}

#[cfg(feature = "cuda")]
enum DeviceCache {
    None,
    /// A dense kernel's cache; a depthwise one reads its input back from
    /// the tape.
    Conv {
        dense: Option<ConvCache>,
        /// SiLU, when the convolution has it, ran inside the batch norm.
        norm: crate::conv::DeviceBatchNormCache,
    },
    Plain(ConvCache),
    MaxPool(crate::conv::DeviceMaxPoolCache),
    Attention(crate::conv::DevicePixelAttentionCache),
}

#[cfg(feature = "cuda")]
impl Yolo11 {
    /// Moves every parameter and running statistic onto `device`.
    pub fn to_cuda_on(&mut self, device: &crate::param::CudaDevice) -> Result<(), NetworkError> {
        self.state_mut()
            .into_iter()
            .try_for_each(|(_, param)| param.to_cuda_on(device))
    }

    /// Brings every parameter, gradient and running statistic back.
    pub fn to_cpu(&mut self) -> Result<(), NetworkError> {
        self.state_mut()
            .into_iter()
            .try_for_each(|(_, param)| param.to_cpu())
    }

    /// [`Yolo11::forward`] on the device the parameters live on.
    pub fn forward_device(
        &mut self,
        input: &crate::conv::DeviceImageBatch,
        training: bool,
    ) -> Result<DeviceTape, NetworkError> {
        use crate::conv::*;
        let mut values: Vec<DeviceImageBatch> = Vec::with_capacity(self.nodes.len());
        let mut caches = Vec::with_capacity(self.nodes.len());
        for node in &mut self.nodes {
            let x = node.inputs.first().map(|&i| &values[i]);
            let (out, cache) = match &mut node.op {
                Op::Input => (input.clone(), DeviceCache::None),
                Op::Conv { kernel, norm, act } => {
                    let x = x.unwrap();
                    let (y, dense) = match kernel {
                        Kernel::Dense(conv) => {
                            let (y, c) = conv.forward_device_keeping(x, false)?;
                            (y, Some(c))
                        }
                        Kernel::Depthwise(conv) => (conv.forward_device(x)?, None),
                    };
                    let (out, norm) = norm.forward_device_with(y, training, *act)?;
                    (out, DeviceCache::Conv { dense, norm })
                }
                Op::Plain(conv) => {
                    let (y, c) = conv.forward_device_keeping(x.unwrap(), false)?;
                    (y, DeviceCache::Plain(c))
                }
                Op::Fused { conv, act, narrow } => {
                    if narrow.is_none() {
                        *narrow = conv.bf16_weight()?;
                    }
                    (
                        conv.infer_device_with(x.unwrap(), *act, narrow.as_ref())?,
                        DeviceCache::None,
                    )
                }
                Op::Add => (
                    add_device(x.unwrap(), &values[node.inputs[1]])?,
                    DeviceCache::None,
                ),
                Op::Concat => {
                    let parts: Vec<&DeviceImageBatch> =
                        node.inputs.iter().map(|&i| &values[i]).collect();
                    (concat_channels_device(&parts)?, DeviceCache::None)
                }
                Op::Slice { offset, width } => (
                    slice_channels_device(x.unwrap(), *offset, *width)?,
                    DeviceCache::None,
                ),
                Op::Upsample => (upsample_nearest_device(x.unwrap(), 2)?, DeviceCache::None),
                Op::MaxPool(k) => {
                    let (y, c) = max_pool_same_device(x.unwrap(), *k)?;
                    (y, DeviceCache::MaxPool(c))
                }
                Op::Attention(shape) => {
                    let (attended, values, c) = pixel_attention_device(x.unwrap(), *shape)?;
                    (
                        concat_channels_device(&[&attended, &values])?,
                        DeviceCache::Attention(c),
                    )
                }
            };
            values.push(out);
            caches.push(cache);
        }
        Ok(DeviceTape { values, caches })
    }

    /// The raw Detect maps of a device forward pass.
    pub fn outputs_device<'a>(
        &self,
        tape: &'a DeviceTape,
    ) -> [&'a crate::conv::DeviceImageBatch; 3] {
        self.outputs.map(|i| &tape.values[i])
    }

    /// [`Yolo11::backward`] on the device.
    pub fn backward_device(
        &mut self,
        tape: DeviceTape,
        grads: [&crate::conv::DeviceImageBatch; 3],
    ) -> Result<crate::conv::DeviceImageBatch, NetworkError> {
        use crate::conv::*;
        let DeviceTape { values, caches } = tape;
        let mut grad: Vec<Option<DeviceImageBatch>> = (0..self.nodes.len()).map(|_| None).collect();
        for (&i, g) in self.outputs.iter().zip(grads) {
            grad[i] = Some(g.clone());
        }
        for (i, cache) in caches.into_iter().enumerate().skip(1).rev() {
            let Some(g) = grad[i].take() else {
                continue;
            };
            let node = &mut self.nodes[i];
            let input = |k: usize| &values[node.inputs[k]];
            let input_grads = match (&mut node.op, cache) {
                (Op::Conv { kernel, norm, .. }, DeviceCache::Conv { dense, norm: nc }) => {
                    let g = norm.backward_device(&nc, &g)?;
                    vec![match (kernel, dense) {
                        (Kernel::Dense(conv), Some(c)) => {
                            conv.backward_device_from(&c, Some(input(0)), &g)?
                        }
                        (Kernel::Depthwise(conv), None) => conv.backward_device(input(0), &g)?,
                        _ => unreachable!("a cache from another kernel"),
                    }]
                }
                (Op::Plain(conv), DeviceCache::Plain(c)) => {
                    vec![conv.backward_device_from(&c, Some(input(0)), &g)?]
                }
                (Op::Add, _) => vec![g.clone(), g],
                (Op::Concat, _) => {
                    let widths: Vec<usize> = (0..node.inputs.len())
                        .map(|k| input(k).channels())
                        .collect();
                    split_channels_device(&g, &widths)?
                }
                (Op::Slice { offset, .. }, _) => {
                    let channels = input(0).channels();
                    let before = g.like(*offset)?;
                    let after = g.like(channels - *offset - g.channels())?;
                    let parts: Vec<&DeviceImageBatch> = [&before, &g, &after]
                        .into_iter()
                        .filter(|p| p.channels() > 0)
                        .collect();
                    vec![concat_channels_device(&parts)?]
                }
                (Op::Upsample, _) => vec![upsample_nearest_backward_device(&g, 2)?],
                (Op::MaxPool(_), DeviceCache::MaxPool(c)) => {
                    vec![max_pool_same_backward_device(&c, &g)?]
                }
                (Op::Attention(shape), DeviceCache::Attention(c)) => {
                    let halves = split_channels_device(&g, &[shape.dim(), shape.dim()])?;
                    vec![pixel_attention_backward_device(&c, &halves[0], &halves[1])?]
                }
                (Op::Fused { .. }, _) => {
                    return Err(NetworkError::InvalidConfig(
                        "a fused model only infers".into(),
                    ));
                }
                _ => unreachable!("a cache from another op"),
            };
            for (&j, g) in node.inputs.iter().zip(input_grads) {
                grad[j] = Some(match grad[j].take() {
                    Some(sum) => add_device(&sum, &g)?,
                    None => g,
                });
            }
        }
        Ok(grad[0].take().expect("the input feeds the graph"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    const TINY: Yolo11Scale = Yolo11Scale {
        depth: 1.0,
        width: 0.0625,
        max_channels: 1024,
    };

    fn map(tensors: &mut SafeTensors, name: &str) -> ImageBatch {
        let (data, shape) = tensors.tensor(name).unwrap();
        let rows = shape[0] * shape[1] * shape[2];
        ImageBatch::from_tokens(
            shape[0],
            shape[1],
            shape[2],
            crate::matrix::Matrix::from_vec(rows, shape[3], data),
        )
    }

    fn assert_close(got: &[f32], expected: &[f32], tolerance: f32, what: &str) {
        assert_eq!(got.len(), expected.len(), "{what}");
        let scale = expected.iter().fold(0f32, |m, v| m.max(v.abs()));
        let gap = got
            .iter()
            .zip(expected)
            .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(
            gap <= tolerance * scale + 1e-5,
            "{what}: gap {gap} at scale {scale}"
        );
    }

    #[test]
    fn detection_loss_matches_ultralytics() {
        let fixture = "/home/vasylt/cs2/fixtures/k13e_loss.safetensors";
        let Ok(mut tensors) = SafeTensors::open(fixture) else {
            eprintln!("skipped: no {fixture}");
            return;
        };
        let maps: [ImageBatch; 3] =
            std::array::from_fn(|i| map(&mut tensors, &format!("maps.{i}")));
        let (rows, _) = tensors.tensor("targets").unwrap();
        let mut targets = vec![Vec::new(); 2];
        for row in rows.chunks(6) {
            targets[row[0] as usize].push((row[1] as usize, [row[2], row[3], row[4], row[5]]));
        }
        let (loss, grads) = detection_loss([&maps[0], &maps[1], &maps[2]], 2, &targets);
        let (expected, _) = tensors.tensor("loss").unwrap();
        assert_close(&loss, &expected, 1e-5, "loss terms");
        for (i, grad) in grads.iter().enumerate() {
            let (expected, _) = tensors.tensor(&format!("grad.{i}")).unwrap();
            assert_close(&grad.tokens.data, &expected, 1e-4, &format!("grad.{i}"));
        }
    }

    #[test]
    fn yolo11_matches_ultralytics() {
        let fixture = "/home/vasylt/cs2/fixtures/k13a_yolo11_tiny.safetensors";
        let Ok(mut tensors) = SafeTensors::open(fixture) else {
            eprintln!("skipped: no {fixture}");
            return;
        };
        let mut model = Yolo11::new(TINY, 1, &mut StdRng::seed_from_u64(0)).unwrap();
        model.load_ultralytics(&mut tensors).unwrap();
        let input = map(&mut tensors, "input");
        let tape = model.clone().forward(&input, false).unwrap();
        for (i, out) in model.outputs(&tape).into_iter().enumerate() {
            assert_close(
                &out.tokens.data,
                &map(&mut tensors, &format!("eval.{i}")).tokens.data,
                1e-4,
                "eval",
            );
        }
        let tape = model.forward(&input, true).unwrap();
        let weights: Vec<ImageBatch> = (0..3)
            .map(|i| map(&mut tensors, &format!("weight.{i}")))
            .collect();
        for (i, out) in model.outputs(&tape).into_iter().enumerate() {
            assert_close(
                &out.tokens.data,
                &map(&mut tensors, &format!("train.{i}")).tokens.data,
                1e-4,
                "train",
            );
        }
        let grad = model
            .backward(tape, [&weights[0], &weights[1], &weights[2]])
            .unwrap();
        assert_close(
            &grad.tokens.data,
            &map(&mut tensors, "grad.input").tokens.data,
            1e-4,
            "input grad",
        );
        for (name, param) in model.state_mut() {
            match name.strip_suffix(".running") {
                Some(bn) => {
                    let (mean, _) = tensors.tensor(&format!("after.{bn}.running_mean")).unwrap();
                    let (var, _) = tensors.tensor(&format!("after.{bn}.running_var")).unwrap();
                    assert_close(&param.value.data, &[mean, var].concat(), 1e-4, &name);
                }
                None => {
                    let (expected, shape) = tensors.tensor(&format!("grad.{name}")).unwrap();
                    assert_close(
                        &param.grad.data,
                        &torch_layout(expected, &shape),
                        // Small BN gamma grads sum thousands of terms; one reaches 1.2e-3 in f32.
                        2e-3,
                        &name,
                    );
                }
            }
        }
    }

    #[test]
    fn a_saved_model_loads_back_exactly() {
        let mut model = Yolo11::new(TINY, 2, &mut StdRng::seed_from_u64(1)).unwrap();
        for (_, param) in model.state_mut() {
            for (i, v) in param.value.data.iter_mut().enumerate() {
                *v += (i % 7) as f32 * 0.01;
            }
        }
        let path =
            std::env::temp_dir().join(format!("yolo_save_{}.safetensors", std::process::id()));
        model.save_ultralytics(&path).unwrap();
        let mut loaded = Yolo11::new(TINY, 2, &mut StdRng::seed_from_u64(2)).unwrap();
        loaded
            .load_ultralytics(&mut SafeTensors::open(&path).unwrap())
            .unwrap();
        std::fs::remove_file(&path).unwrap();
        let fused = model.fuse();
        fused.save_ultralytics(&path).unwrap();
        let mut fused_loaded = loaded.fuse();
        fused_loaded
            .load_ultralytics(&mut SafeTensors::open(&path).unwrap())
            .unwrap();
        std::fs::remove_file(&path).unwrap();
        for (mut a, mut b) in [(model, loaded), (fused, fused_loaded)] {
            for ((name, x), (_, y)) in a.state_mut().into_iter().zip(b.state_mut()) {
                assert_eq!(x.value.data, y.value.data, "{name}");
            }
        }

        // Saved from a real checkpoint, every tensor comes back as ultralytics
        // wrote it, shape included.
        let Ok(mut original) =
            SafeTensors::open("/home/vasylt/cs2/fixtures/k13d_player_v1.safetensors")
        else {
            return;
        };
        let mut player = Yolo11::new(Yolo11Scale::S, 1, &mut StdRng::seed_from_u64(0)).unwrap();
        player.load_ultralytics(&mut original).unwrap();
        player.save_ultralytics(&path).unwrap();
        let mut saved = SafeTensors::open(&path).unwrap();
        let names: Vec<String> = saved.names().map(str::to_owned).collect();
        assert!(names.len() > 300, "{} tensors", names.len());
        for name in names {
            assert_eq!(
                saved.tensor(&name).unwrap(),
                original.tensor(&name).unwrap(),
                "{name}"
            );
        }
        std::fs::remove_file(&path).unwrap();
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn device_yolo11_matches_the_host() {
        use rand::Rng;
        let device = match crate::cuda_training::cuda_doctor(0, 4096) {
            Ok(_) => crate::param::CudaDevice::new(0, 4096).unwrap(),
            Err(NetworkError::Cuda(message)) if message.contains("NO_DEVICE") => {
                crate::cuda_training::skip_without_cuda();
                return;
            }
            Err(error) => panic!("CUDA is present but the CUDA doctor failed: {error}"),
        };
        let rng = &mut StdRng::seed_from_u64(1);
        let mut host = Yolo11::new(TINY, 2, rng).unwrap();
        let mut noise = |shape: &ImageBatch| {
            let mut out = shape.clone();
            out.tokens
                .data
                .iter_mut()
                .for_each(|v| *v = rng.r#gen::<f32>() - 0.5);
            out
        };
        let input = noise(&ImageBatch::from_tokens(
            2,
            64,
            96,
            crate::matrix::Matrix::new(2 * 64 * 96, 3),
        ));
        let mut model = host.clone();
        model.to_cuda_on(&device).unwrap();
        let tape = host.forward(&input, true).unwrap();
        let grads: Vec<ImageBatch> = host.outputs(&tape).into_iter().map(&mut noise).collect();
        let on_tape = model
            .forward_device(&device.upload(&input).unwrap(), true)
            .unwrap();
        for (got, expected) in model
            .outputs_device(&on_tape)
            .into_iter()
            .zip(host.outputs(&tape))
        {
            assert_close(
                &got.download().unwrap().tokens.data,
                &expected.tokens.data,
                // Batch statistics over 23 layers: summation order alone
                // moves the train-mode outputs by 2e-4 relative and the
                // input gradient by 3e-4.
                1e-3,
                "output",
            );
        }
        let on_grads: Vec<_> = grads.iter().map(|g| device.upload(g).unwrap()).collect();
        let input_grad = host
            .backward(tape, [&grads[0], &grads[1], &grads[2]])
            .unwrap();
        let on_input_grad = model
            .backward_device(on_tape, [&on_grads[0], &on_grads[1], &on_grads[2]])
            .unwrap();
        assert_close(
            &on_input_grad.download().unwrap().tokens.data,
            &input_grad.tokens.data,
            1e-3,
            "input grad",
        );
        model.to_cpu().unwrap();
        for ((name, got), (_, expected)) in model.state_mut().into_iter().zip(host.state_mut()) {
            match name.ends_with(".running") {
                true => assert_close(&got.value.data, &expected.value.data, 1e-4, &name),
                false => assert_close(&got.grad.data, &expected.grad.data, 2e-3, &name),
            }
        }
        // Folding the norms in changes nothing but rounding, on either side.
        let eval = host.forward(&input, false).unwrap();
        let mut fused = host.fuse();
        let host_fused = fused.forward(&input, false).unwrap();
        fused.to_cuda_on(&device).unwrap();
        let on_fused = fused
            .forward_device(&device.upload(&input).unwrap(), false)
            .unwrap();
        for ((on_host, on_device), expected) in fused
            .outputs(&host_fused)
            .into_iter()
            .zip(fused.outputs_device(&on_fused))
            .zip(host.outputs(&eval))
        {
            assert_close(
                &on_host.tokens.data,
                &expected.tokens.data,
                1e-4,
                "fused host",
            );
            assert_close(
                &on_device.download().unwrap().tokens.data,
                &expected.tokens.data,
                1e-4,
                "fused device",
            );
        }
        // Mixed precision runs the im2col convolutions as bf16 GEMMs; bf16
        // keeps 8 bits of mantissa, so the maps stay within a few percent.
        let mixed = crate::param::CudaDevice::with_mixed_precision(0, 4096).unwrap();
        let mut fused = host.fuse();
        fused.to_cuda_on(&mixed).unwrap();
        let on_mixed = fused
            .forward_device(&mixed.upload(&input).unwrap(), false)
            .unwrap();
        for (on_device, expected) in fused
            .outputs_device(&on_mixed)
            .into_iter()
            .zip(host.outputs(&eval))
        {
            assert_close(
                &on_device.download().unwrap().tokens.data,
                &expected.tokens.data,
                5e-2,
                "fused mixed device",
            );
        }
    }

    /// Gate 1, first half: the trained player model's raw Detect maps on
    /// ultralytics' first validation batch, image by image on the device.
    #[cfg(feature = "cuda")]
    #[test]
    fn player_model_matches_ultralytics_on_sixteen_val_images() {
        let fixtures = "/home/vasylt/cs2/fixtures";
        let (Ok(mut weights), Ok(mut reference)) = (
            SafeTensors::open(format!("{fixtures}/k13d_player_v1.safetensors")),
            SafeTensors::open(format!("{fixtures}/k13d_player_val16.safetensors")),
        ) else {
            eprintln!("skipped: no K13d fixtures");
            return;
        };
        let device = match crate::cuda_training::cuda_doctor(0, 8192) {
            Ok(_) => crate::param::CudaDevice::new(0, 8192).unwrap(),
            Err(NetworkError::Cuda(message)) if message.contains("NO_DEVICE") => {
                crate::cuda_training::skip_without_cuda();
                return;
            }
            Err(error) => panic!("CUDA is present but the CUDA doctor failed: {error}"),
        };
        let mut model = Yolo11::new(Yolo11Scale::S, 1, &mut StdRng::seed_from_u64(0)).unwrap();
        model.load_ultralytics(&mut weights).unwrap();
        model.to_cuda_on(&device).unwrap();
        let (pixels, shape) = reference.tensor("input").unwrap();
        let &[images, height, width, _] = shape.as_slice() else {
            panic!("input is {shape:?}")
        };
        let raw: Vec<Vec<f32>> = (0..3)
            .map(|i| reference.tensor(&format!("raw.{i}")).unwrap().0)
            .collect();
        let per_image = height * width * 3;
        let mut worst = 0.0f32;
        for image in 0..images {
            let input = ImageBatch::from_tokens(
                1,
                height,
                width,
                crate::matrix::Matrix::from_vec(
                    height * width,
                    3,
                    pixels[image * per_image..(image + 1) * per_image]
                        .iter()
                        .map(|v| v / 255.0)
                        .collect(),
                ),
            );
            let tape = model
                .forward_device(&device.upload(&input).unwrap(), false)
                .unwrap();
            for (out, expected) in model.outputs_device(&tape).into_iter().zip(&raw) {
                let got = out.download().unwrap().tokens.data;
                let expected = &expected[image * got.len()..(image + 1) * got.len()];
                let scale = expected.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                let gap = got
                    .iter()
                    .zip(expected)
                    .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
                worst = worst.max(gap / scale);
            }
        }
        eprintln!("worst raw-map gap {worst:e} of the map's largest value over {images} images");
        assert!(worst < 1e-3, "worst relative gap {worst}");
    }
}
