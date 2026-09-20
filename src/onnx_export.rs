//! Writing a [`Network`] or a [`TransformerLm`] out as an ONNX graph.
//!
//! A model trained here can otherwise only be run here. ONNX is what every
//! other runtime reads. A dense network is a short graph — one `Gemm` per layer
//! and one activation node after it. A transformer is a long one: RMSNorm is
//! six nodes, rotary positions are nine, and a block is about forty-five.
//!
//! NOTE: the protobuf is written by hand, because the only ONNX dependency
//! in the crate (`tract-onnx`, behind `--features onnx`) reads and does not
//! write, and the schema this needs is nine message types. Encoding is the
//! easy direction: every field below is a varint, a string or a length-
//! delimited submessage. What a real protobuf crate would buy is the reverse
//! direction and the operators this does not emit, not these hundred lines.

use crate::activations::Activation;
use crate::attention::MultiHeadAttention;
use crate::matrix::Matrix;
use crate::network::{Network, NetworkError};
use crate::norm::RmsNorm;
use crate::param::Linear;
use crate::rope::Rope;
use crate::transformer::TransformerLm;
use crate::transformer_block::FeedForward;
use std::path::Path;

/// Appends a base-128 varint, protobuf's integer encoding.
fn varint(mut value: u64, out: &mut Vec<u8>) {
    while value >= 0x80 {
        out.push((value as u8) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

/// Field number and wire type: 0 for a varint, 2 for anything length-delimited.
fn tag(field: u32, wire: u32, out: &mut Vec<u8>) {
    varint((u64::from(field) << 3) | u64::from(wire), out);
}

fn varint_field(field: u32, value: u64, out: &mut Vec<u8>) {
    tag(field, 0, out);
    varint(value, out);
}

fn bytes_field(field: u32, data: &[u8], out: &mut Vec<u8>) {
    tag(field, 2, out);
    varint(data.len() as u64, out);
    out.extend_from_slice(data);
}

fn string_field(field: u32, value: &str, out: &mut Vec<u8>) {
    bytes_field(field, value.as_bytes(), out);
}

/// A `TensorProto` holding f32 weights: dims, `elem_type` 1 (FLOAT), a name,
/// and the values as little-endian `raw_data`, which is what the format
/// requires whatever the host is.
fn tensor(name: &str, dims: &[usize], values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4 + 32);
    for &dim in dims {
        varint_field(1, dim as u64, &mut out);
    }
    varint_field(2, 1, &mut out);
    string_field(8, name, &mut out);

    let raw: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    bytes_field(9, &raw, &mut out);
    out
}

/// A `ValueInfoProto` for a `[batch, width]` float tensor. The first dimension
/// is a `dim_param` rather than a number, so the graph serves one row or a
/// thousand.
fn value_info(name: &str, width: usize) -> Vec<u8> {
    let mut batch = Vec::new();
    string_field(2, "batch", &mut batch);
    let mut fixed = Vec::new();
    varint_field(1, width as u64, &mut fixed);

    let mut shape = Vec::new();
    bytes_field(1, &batch, &mut shape);
    bytes_field(1, &fixed, &mut shape);

    let mut tensor_type = Vec::new();
    varint_field(1, 1, &mut tensor_type);
    bytes_field(2, &shape, &mut tensor_type);

    let mut type_proto = Vec::new();
    bytes_field(1, &tensor_type, &mut type_proto);

    let mut out = Vec::new();
    string_field(1, name, &mut out);
    bytes_field(2, &type_proto, &mut out);
    out
}

/// What a node attribute can hold. `Ints` covers `axes` and `perm`, which are
/// the only repeated attributes anything here needs.
enum Attr<'a> {
    Int(i64),
    Ints(&'a [i64]),
}

/// A `NodeProto` and its attributes — `transB` on a `Gemm`, `axis` on a
/// `Softmax`, `perm` on a `Transpose`.
fn node(op: &str, inputs: &[&str], output: &str, attributes: &[(&str, Attr<'_>)]) -> Vec<u8> {
    let mut out = Vec::new();
    for input in inputs {
        string_field(1, input, &mut out);
    }
    string_field(2, output, &mut out);
    string_field(3, output, &mut out);
    string_field(4, op, &mut out);

    for (name, value) in attributes {
        let mut attr = Vec::new();
        string_field(1, name, &mut attr);
        match value {
            Attr::Int(value) => {
                varint_field(3, *value as u64, &mut attr);
                varint_field(20, 2, &mut attr); // AttributeProto.INT
            }
            Attr::Ints(values) => {
                for value in *values {
                    varint_field(8, *value as u64, &mut attr);
                }
                varint_field(20, 7, &mut attr); // AttributeProto.INTS
            }
        }
        bytes_field(5, &attr, &mut out);
    }
    out
}

/// Writes `network` to `path` as an ONNX model at opset 13.
pub fn export(network: &Network, path: impl AsRef<Path>) -> Result<(), NetworkError> {
    let mut graph = Vec::new();
    let mut initializers = Vec::new();
    let mut current = "input".to_string();

    for (index, layer) in network.layers().iter().enumerate() {
        let (weights, biases) = (format!("W{index}"), format!("B{index}"));
        initializers.push(tensor(
            &weights,
            &[layer.weights.rows, layer.weights.cols],
            &layer.weights.data,
        ));
        initializers.push(tensor(&biases, &[layer.biases.rows], &layer.biases.data));

        // Gemm with transB: y = x * W^T + b, which is the row-vector form of
        // the `y = W x + b` the forward pass computes.
        let gemm = format!("gemm{index}");
        bytes_field(
            1,
            &node(
                "Gemm",
                &[&current, &weights, &biases],
                &gemm,
                &[("transB", Attr::Int(1))],
            ),
            &mut graph,
        );
        current = gemm;

        let op = match layer.activation {
            Activation::Relu => "Relu",
            Activation::Sigmoid => "Sigmoid",
            Activation::Tanh => "Tanh",
            Activation::Softmax => "Softmax",
            // Nothing to apply: the Gemm output is the layer output.
            Activation::Linear => continue,
        };
        let activated = format!("act{index}");
        bytes_field(1, &node(op, &[&current], &activated, &[]), &mut graph);
        current = activated;
    }

    string_field(2, "rusting_brain", &mut graph);
    for initializer in &initializers {
        bytes_field(5, initializer, &mut graph);
    }
    bytes_field(11, &value_info("input", network.input_size()), &mut graph);
    bytes_field(12, &value_info(&current, network.output_size()), &mut graph);

    let mut opset = Vec::new();
    varint_field(2, 13, &mut opset);

    let mut model = Vec::new();
    varint_field(1, 8, &mut model); // ir_version 8, which is ONNX 1.12
    string_field(2, "rusting_brain", &mut model);
    bytes_field(7, &graph, &mut model);
    bytes_field(8, &opset, &mut model);

    std::fs::write(path, model)?;
    Ok(())
}

/// A `TensorProto` holding `int64` values, which is what `Reshape`, `Slice`
/// and `Expand` take their shapes as.
fn tensor_i64(name: &str, dims: &[usize], values: &[i64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 8 + 32);
    for &dim in dims {
        varint_field(1, dim as u64, &mut out);
    }
    varint_field(2, 7, &mut out); // TensorProto.INT64
    string_field(8, name, &mut out);

    let raw: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    bytes_field(9, &raw, &mut out);
    out
}

/// A `ValueInfoProto` whose dimensions are all fixed numbers.
///
/// The dense export leaves its batch dimension symbolic. A transformer graph
/// cannot: the rotary tables and the causal mask are baked in at one sequence
/// length, so the shape is part of what was exported.
fn value_info_fixed(name: &str, elem_type: u64, dims: &[usize]) -> Vec<u8> {
    let mut shape = Vec::new();
    for &dim in dims {
        let mut fixed = Vec::new();
        varint_field(1, dim as u64, &mut fixed);
        bytes_field(1, &fixed, &mut shape);
    }

    let mut tensor_type = Vec::new();
    varint_field(1, elem_type, &mut tensor_type);
    bytes_field(2, &shape, &mut tensor_type);

    let mut type_proto = Vec::new();
    bytes_field(1, &tensor_type, &mut type_proto);

    let mut out = Vec::new();
    string_field(1, name, &mut out);
    bytes_field(2, &type_proto, &mut out);
    out
}

/// Nodes and initializers under construction, and the counter that keeps
/// every name in the graph distinct.
struct Graph {
    nodes: Vec<Vec<u8>>,
    initializers: Vec<Vec<u8>>,
    counter: usize,
}

impl Graph {
    fn new() -> Self {
        Self {
            nodes: Vec::new(),
            initializers: Vec::new(),
            counter: 0,
        }
    }

    fn fresh(&mut self, hint: &str) -> String {
        self.counter += 1;
        format!("{hint}_{}", self.counter)
    }

    /// Adds a node and returns the name of its single output, which is what
    /// the next node reads.
    fn op(
        &mut self,
        op: &str,
        inputs: &[&str],
        hint: &str,
        attributes: &[(&str, Attr<'_>)],
    ) -> String {
        let output = self.fresh(hint);
        self.nodes.push(node(op, inputs, &output, attributes));
        output
    }

    fn constant(&mut self, hint: &str, dims: &[usize], values: &[f32]) -> String {
        let name = self.fresh(hint);
        self.initializers.push(tensor(&name, dims, values));
        name
    }

    /// A 1-D `int64` initializer: a shape for `Reshape` or `Expand`, or the
    /// bounds of a `Slice`.
    fn shape(&mut self, values: &[i64]) -> String {
        let name = self.fresh("shape");
        self.initializers
            .push(tensor_i64(&name, &[values.len()], values));
        name
    }

    fn reshape(&mut self, input: &str, dims: &[usize]) -> String {
        let shape: Vec<i64> = dims.iter().map(|&dim| dim as i64).collect();
        let shape = self.shape(&shape);
        self.op("Reshape", &[input, &shape], "reshaped", &[])
    }

    /// `input * W^T`, the projection a [`Linear`] computes.
    ///
    /// The weights are transposed here rather than by a `Gemm` attribute,
    /// because everything they feed is rank 3 and `Gemm` is rank 2 only.
    fn project(&mut self, hint: &str, linear: &Linear, input: &str) -> String {
        let weight = &linear.weight.value;
        let name = self.constant(hint, &[weight.cols, weight.rows], &transposed(weight));
        self.op("MatMul", &[input, &name], hint, &[])
    }
}

/// `[rows, cols]` read out as `[cols, rows]`.
fn transposed(value: &Matrix) -> Vec<f32> {
    let mut out = vec![0.0; value.data.len()];
    for row in 0..value.rows {
        for col in 0..value.cols {
            out[col * value.rows + row] = value.data[row * value.cols + col];
        }
    }
    out
}

/// A rotary table widened from `[positions, head_dim / 2]` to
/// `[positions, 1, head_dim]`.
///
/// Rotate-half pairs channel `j` with channel `j + head_dim / 2`, and both
/// carry the same angle, so each row is its half repeated. The middle
/// dimension is 1 so one table broadcasts across every head.
fn widened(table: &[f32], positions: usize, half: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(positions * half * 2);
    for position in 0..positions {
        let row = &table[position * half..position * half + half];
        out.extend_from_slice(row);
        out.extend_from_slice(row);
    }
    out
}

/// An additive causal mask: zero where a query may read a key, and a number
/// large enough that the softmax returns exactly zero everywhere else.
fn causal_mask(seq_len: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(seq_len * seq_len);
    for query in 0..seq_len {
        for key in 0..seq_len {
            out.push(if key <= query { 0.0 } else { -1e30 });
        }
    }
    out
}

/// `x * inverse_rms(x) * gain`, six nodes, the shape
/// [`RmsNorm::forward`](crate::norm::RmsNorm::forward) computes.
fn rms_norm(graph: &mut Graph, norm: &RmsNorm, input: &str) -> String {
    let square = graph.op("Mul", &[input, input], "square", &[]);
    let mean = graph.op(
        "ReduceMean",
        &[&square],
        "mean_square",
        &[("axes", Attr::Ints(&[-1])), ("keepdims", Attr::Int(1))],
    );
    let epsilon = graph.constant("epsilon", &[], &[norm.eps]);
    let shifted = graph.op("Add", &[&mean, &epsilon], "shifted", &[]);
    let rms = graph.op("Sqrt", &[&shifted], "rms", &[]);
    let normalized = graph.op("Div", &[input, &rms], "normalized", &[]);
    let gain = graph.constant("gain", &[norm.d_model()], &norm.weight.value.data);
    graph.op("Mul", &[&normalized, &gain], "normed", &[])
}

/// Rotary positions over a `[seq_len, heads * head_dim]` projection, returning
/// it as `[seq_len, heads, head_dim]`.
fn rotary(graph: &mut Graph, rope: &Rope, heads: usize, seq_len: usize, input: &str) -> String {
    let head_dim = rope.head_dim();
    let half = head_dim / 2;
    let split = graph.reshape(input, &[seq_len, heads, head_dim]);

    let axis = graph.shape(&[2]);
    let start = graph.shape(&[0]);
    let middle = graph.shape(&[half as i64]);
    let end = graph.shape(&[head_dim as i64]);
    let low = graph.op("Slice", &[&split, &start, &middle, &axis], "low", &[]);
    let high = graph.op("Slice", &[&split, &middle, &end, &axis], "high", &[]);
    let negated = graph.op("Neg", &[&high], "negated", &[]);
    let turned = graph.op(
        "Concat",
        &[&negated, &low],
        "turned",
        &[("axis", Attr::Int(2))],
    );

    let dims = [seq_len, 1, head_dim];
    let cos = graph.constant("cos", &dims, &widened(rope.cos(), seq_len, half));
    let sin = graph.constant("sin", &dims, &widened(rope.sin(), seq_len, half));
    let straight = graph.op("Mul", &[&split, &cos], "straight", &[]);
    let rotated = graph.op("Mul", &[&turned, &sin], "rotated", &[]);
    graph.op("Add", &[&straight, &rotated], "positioned", &[])
}

/// Repeats each key/value head `group` times, which is what grouped-query
/// attention does when it reads key head `head / group`.
fn repeat_kv(
    graph: &mut Graph,
    input: &str,
    seq_len: usize,
    kv_heads: usize,
    group: usize,
    head_dim: usize,
) -> String {
    if group == 1 {
        return input.to_string();
    }
    let spread = graph.reshape(input, &[seq_len, kv_heads, 1, head_dim]);
    let shape = graph.shape(&[
        seq_len as i64,
        kv_heads as i64,
        group as i64,
        head_dim as i64,
    ]);
    let expanded = graph.op("Expand", &[&spread, &shape], "expanded", &[]);
    graph.reshape(&expanded, &[seq_len, kv_heads * group, head_dim])
}

/// One attention sub-layer: projections, rotary positions, the masked softmax
/// and the output projection.
fn attention(
    graph: &mut Graph,
    attention: &MultiHeadAttention,
    seq_len: usize,
    input: &str,
) -> String {
    let heads = attention.num_heads();
    let kv_heads = attention.num_kv_heads();
    let head_dim = attention.head_dim();
    let group = heads / kv_heads;

    // The `1 / sqrt(head_dim)` scale rides on the query weights rather than on
    // a `Mul` node: rotary positions are a rotation, so a scalar factor passes
    // through them, and the graph loses a node tract's optimizer stumbles over.
    let scale = (head_dim as f32).sqrt().recip();
    let mut scaled_query = attention.query.clone();
    for value in &mut scaled_query.weight.value.data {
        *value *= scale;
    }
    let queries = graph.project("queries", &scaled_query, input);
    let keys = graph.project("keys", &attention.key, input);
    let values = graph.project("values", &attention.value, input);

    let queries = rotary(graph, &attention.rope, heads, seq_len, &queries);
    let keys = rotary(graph, &attention.rope, kv_heads, seq_len, &keys);
    let values = graph.reshape(&values, &[seq_len, kv_heads, head_dim]);

    let keys = repeat_kv(graph, &keys, seq_len, kv_heads, group, head_dim);
    let values = repeat_kv(graph, &values, seq_len, kv_heads, group, head_dim);

    // Heads first, so one batched `MatMul` covers all of them.
    let queries = graph.op(
        "Transpose",
        &[&queries],
        "queries_by_head",
        &[("perm", Attr::Ints(&[1, 0, 2]))],
    );
    let keys = graph.op(
        "Transpose",
        &[&keys],
        "keys_by_head",
        &[("perm", Attr::Ints(&[1, 2, 0]))],
    );
    let values = graph.op(
        "Transpose",
        &[&values],
        "values_by_head",
        &[("perm", Attr::Ints(&[1, 0, 2]))],
    );

    let scaled = graph.op("MatMul", &[&queries, &keys], "scores", &[]);
    // A bidirectional layer reads the whole sequence, so it carries no mask at
    // all rather than a square initializer of zeros.
    let masked = if attention.is_causal() {
        let mask = graph.constant("mask", &[1, seq_len, seq_len], &causal_mask(seq_len));
        graph.op("Add", &[&scaled, &mask], "masked", &[])
    } else {
        scaled
    };
    let probabilities = graph.op(
        "Softmax",
        &[&masked],
        "probabilities",
        &[("axis", Attr::Int(-1))],
    );

    let context = graph.op("MatMul", &[&probabilities, &values], "context", &[]);
    let context = graph.op(
        "Transpose",
        &[&context],
        "context_by_token",
        &[("perm", Attr::Ints(&[1, 0, 2]))],
    );
    let merged = graph.reshape(&context, &[seq_len, heads * head_dim]);
    graph.project("attended", &attention.output, &merged)
}

/// `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`, the same
/// approximation [`crate::ffn::gelu`] uses. Opset 13 has no `Gelu` node.
fn gelu(graph: &mut Graph, input: &str) -> String {
    let square = graph.op("Mul", &[input, input], "square", &[]);
    let cube = graph.op("Mul", &[&square, input], "cube", &[]);
    let lambda = graph.constant("lambda", &[], &[0.044_715]);
    let scaled = graph.op("Mul", &[&cube, &lambda], "scaled_cube", &[]);
    let sum = graph.op("Add", &[input, &scaled], "inner_sum", &[]);
    let coefficient = graph.constant("coefficient", &[], &[0.797_884_6]);
    let inner = graph.op("Mul", &[&sum, &coefficient], "inner", &[]);
    let tanh = graph.op("Tanh", &[&inner], "tanh", &[]);
    let one = graph.constant("one", &[], &[1.0]);
    let shifted = graph.op("Add", &[&tanh, &one], "shifted", &[]);
    let half = graph.constant("half", &[], &[0.5]);
    let halved = graph.op("Mul", &[&shifted, &half], "halved", &[]);
    graph.op("Mul", &[input, &halved], "gelu", &[])
}

fn feed_forward(
    graph: &mut Graph,
    feed_forward: &FeedForward,
    input: &str,
) -> Result<String, NetworkError> {
    match feed_forward {
        FeedForward::SwiGlu(ffn) => {
            let gate = graph.project("gate", &ffn.gate, input);
            let up = graph.project("up", &ffn.up, input);
            let sigmoid = graph.op("Sigmoid", &[&gate], "sigmoid", &[]);
            let silu = graph.op("Mul", &[&gate, &sigmoid], "silu", &[]);
            let hidden = graph.op("Mul", &[&silu, &up], "hidden", &[]);
            Ok(graph.project("down", &ffn.down, &hidden))
        }
        FeedForward::Gelu(ffn) => {
            let hidden = graph.project("up", &ffn.up, input);
            let activated = gelu(graph, &hidden);
            Ok(graph.project("down", &ffn.down, &activated))
        }
        FeedForward::Moe(_) => Err(NetworkError::InvalidConfig(
            "a mixture-of-experts layer has no ONNX representation: routing is data-dependent \
             control flow, not a graph of tensor ops"
                .into(),
        )),
    }
}

/// Writes `model` to `path` as an ONNX model at opset 13, for sequences of
/// exactly `seq_len` tokens.
pub fn export_transformer(
    model: &TransformerLm,
    seq_len: usize,
    path: impl AsRef<Path>,
) -> Result<(), NetworkError> {
    let mut graph = Graph::new();

    let table = graph.constant(
        "embedding",
        &[
            model.embedding.weight.value.rows,
            model.embedding.weight.value.cols,
        ],
        &model.embedding.weight.value.data,
    );
    let mut hidden = graph.op(
        "Gather",
        &[&table, "ids"],
        "hidden",
        &[("axis", Attr::Int(0))],
    );

    for block in &model.blocks {
        let normed = rms_norm(&mut graph, &block.attention_norm, &hidden);
        let attended = attention(&mut graph, &block.attention, seq_len, &normed);
        hidden = graph.op("Add", &[&hidden, &attended], "residual", &[]);

        let normed = rms_norm(&mut graph, &block.feed_forward_norm, &hidden);
        let projected = feed_forward(&mut graph, &block.feed_forward, &normed)?;
        hidden = graph.op("Add", &[&hidden, &projected], "residual", &[]);
    }

    let hidden = rms_norm(&mut graph, &model.final_norm, &hidden);
    let logits = match &model.lm_head {
        Some(head) => graph.project("logits", head, &hidden),
        // Tied embeddings: the table is already an initializer, so the output
        // projection reads it back rather than storing it twice.
        None => {
            let transposed = graph.op(
                "Transpose",
                &[&table],
                "unembedding",
                &[("perm", Attr::Ints(&[1, 0]))],
            );
            graph.op("MatMul", &[&hidden, &transposed], "logits", &[])
        }
    };

    let mut body = Vec::new();
    for node in &graph.nodes {
        bytes_field(1, node, &mut body);
    }
    string_field(2, "rusting_brain", &mut body);
    for initializer in &graph.initializers {
        bytes_field(5, initializer, &mut body);
    }
    bytes_field(11, &value_info_fixed("ids", 7, &[seq_len]), &mut body);
    bytes_field(
        12,
        &value_info_fixed(&logits, 1, &[seq_len, model.config.vocab_size]),
        &mut body,
    );

    let mut opset = Vec::new();
    varint_field(2, 13, &mut opset);

    let mut proto = Vec::new();
    varint_field(1, 8, &mut proto); // ir_version 8, which is ONNX 1.12
    string_field(2, "rusting_brain", &mut proto);
    bytes_field(7, &body, &mut proto);
    bytes_field(8, &opset, &mut proto);

    std::fs::write(path, proto)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::losses::Loss;
    use crate::optimizers::Optimizer;
    #[cfg(feature = "onnx")]
    use rand::SeedableRng;

    fn trained_network() -> Network {
        let dataset = crate::dataset::Dataset::new(
            vec![
                vec![0.0, 0.0],
                vec![0.0, 1.0],
                vec![1.0, 0.0],
                vec![1.0, 1.0],
            ],
            vec![
                vec![1.0, 0.0],
                vec![0.0, 1.0],
                vec![0.0, 1.0],
                vec![1.0, 0.0],
            ],
        );
        let mut network = Network::builder()
            .input_size(2)
            .dense(6, Activation::Tanh)
            .dense(4, Activation::Relu)
            .dense(2, Activation::Softmax)
            .loss(Loss::CrossEntropy)
            .optimizer(Optimizer::adam(0.05))
            .seed(11)
            .build();
        network
            .fit(
                &dataset,
                crate::network::TrainConfig {
                    epochs: 200,
                    batch_size: 4,
                    shuffle: true,
                    seed: Some(3),
                },
            )
            .unwrap();
        network
    }

    #[test]
    fn the_exported_graph_has_a_node_and_two_initializers_per_layer() {
        let network = trained_network();
        let path = std::env::temp_dir().join(format!("rb_export_{}.onnx", std::process::id()));
        network.save_onnx(&path).unwrap();

        let bytes = std::fs::read(&path).unwrap();
        // Names are length-delimited strings in the wire format, so they are
        // readable in the bytes whatever the framing around them is.
        let text = String::from_utf8_lossy(&bytes);
        for name in [
            "input", "W0", "B0", "W2", "B2", "Gemm", "Tanh", "Relu", "Softmax",
        ] {
            assert!(text.contains(name), "{name} is missing from the graph");
        }
        // Every weight and bias value is in there as raw little-endian f32.
        let weights: usize = network.layers().iter().map(|l| l.weights.data.len()).sum();
        let biases: usize = network.layers().iter().map(|l| l.biases.data.len()).sum();
        assert!(bytes.len() > (weights + biases) * 4);

        std::fs::remove_file(path).unwrap();
    }

    fn tiny_language_model(tied: bool) -> TransformerLm {
        TransformerLm::builder()
            .vocab_size(16)
            .d_model(8)
            .n_layers(2)
            .heads(4, 2, 4)
            .d_ff(16)
            .moe_layers([])
            .max_seq_len(12)
            .tie_embeddings(tied)
            .seed(7)
            .build()
            .unwrap()
    }

    #[test]
    fn a_mixture_of_experts_model_is_refused_rather_than_exported_dense() {
        let model = TransformerLm::builder()
            .vocab_size(16)
            .d_model(8)
            .n_layers(2)
            .heads(4, 2, 4)
            .d_ff(16)
            .moe_d_ff(8)
            .experts(4, 2)
            .moe_layers([1])
            .max_seq_len(12)
            .seed(7)
            .build()
            .unwrap();
        let path = std::env::temp_dir().join(format!("rb_moe_{}.onnx", std::process::id()));

        let error = model.save_onnx(&path, 4).unwrap_err().to_string();
        assert!(error.contains("mixture-of-experts"), "{error}");

        // A sequence longer than the model was built for is refused too, since
        // the rotary tables the graph carries stop there.
        assert!(tiny_language_model(true).save_onnx(&path, 13).is_err());
    }

    /// The check that matters: another runtime reads the file and agrees with
    /// [`Network::predict`] to a float's worth of precision.
    #[cfg(feature = "onnx")]
    #[test]
    fn tract_runs_the_exported_graph_and_agrees_with_predict() {
        let network = trained_network();
        let path = std::env::temp_dir().join(format!("rb_roundtrip_{}.onnx", std::process::id()));
        network.save_onnx(&path).unwrap();

        let loaded = crate::onnx::OnnxModel::load_with_input_shape(&path, &[1, 2]).unwrap();
        for input in [
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
            vec![0.25, 0.75],
        ] {
            let ours = network.predict(&input).unwrap();
            let theirs = loaded.predict(&input).unwrap();
            assert_eq!(ours.len(), theirs.len());
            for (ours, theirs) in ours.iter().zip(&theirs) {
                assert!((ours - theirs).abs() < 1e-5, "{ours} vs {theirs}");
            }
        }

        std::fs::remove_file(path).unwrap();
    }

    /// The same check for the transformer graph, which is where the work is:
    /// rotary positions, grouped-query heads, a causal mask and SwiGLU all
    /// have to come out of the file doing what they do here.
    #[cfg(feature = "onnx")]
    #[test]
    fn tract_runs_the_exported_transformer_and_agrees_with_the_forward_pass() {
        use tract_onnx::prelude::*;

        let ids: Vec<u32> = vec![1, 5, 3, 2, 7, 0];
        for (index, model) in [
            tiny_language_model(true),
            tiny_language_model(false),
            {
                // A GELU block, which the builder does not produce but the
                // public fields allow.
                let mut model = tiny_language_model(true);
                let mut rng = rand::rngs::StdRng::seed_from_u64(3);
                model.blocks[1].feed_forward = FeedForward::gelu(8, 16, &mut rng);
                model
            },
            {
                // Bidirectional: the graph drops the mask node, so the file and
                // the forward pass have to agree without it.
                let mut model = tiny_language_model(true);
                for block in &mut model.blocks {
                    block.attention.set_causal(false);
                }
                model
            },
        ]
        .into_iter()
        .enumerate()
        {
            let path = std::env::temp_dir().join(format!(
                "rb_transformer_{}_{index}.onnx",
                std::process::id()
            ));
            model.save_onnx(&path, ids.len()).unwrap();

            let runnable = tract_onnx::onnx()
                .model_for_path(&path)
                .unwrap()
                .into_optimized()
                .unwrap()
                .into_runnable()
                .unwrap();
            let input: Tensor = tract_ndarray::Array1::from(
                ids.iter().map(|&id| i64::from(id)).collect::<Vec<_>>(),
            )
            .into();
            let outputs = runnable.run(tvec!(input.into())).unwrap();
            let theirs = outputs[0].to_array_view::<f32>().unwrap();

            let (ours, _) = model
                .forward_batch(&crate::batch::TokenBatch::new(std::slice::from_ref(&ids)).unwrap())
                .unwrap();
            assert_eq!(theirs.len(), ours.data.len());
            for (theirs, ours) in theirs.iter().zip(&ours.data) {
                assert!((theirs - ours).abs() < 1e-4, "{theirs} vs {ours}");
            }

            std::fs::remove_file(path).unwrap();
        }
    }
}
