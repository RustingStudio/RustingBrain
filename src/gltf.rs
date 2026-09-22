//! Reading glTF binary files, which is what a 3D asset dataset is made of.
//!
//! GLB is a twelve-byte header followed by chunks: one of JSON describing the
//! scene, one of the bytes the JSON points into. The JSON names *accessors*,
//! each of which is a typed view — so many `VEC3`s of `f32`, at this offset,
//! with this stride — onto a slice of that binary chunk. Reading a mesh is
//! walking the scene graph for primitives, resolving their accessors, and
//! applying the transform each node's ancestors put on it.
//!
//! [`Glb::read`] keeps the two chunks and [`Glb::mesh`] flattens the whole
//! scene into one [`Mesh`]. That is the right shape for this pipeline, which
//! wants one surface to sample distances from and has no use for the material
//! or animation the file also carries.
//!
//! NOTE: no Draco, no sparse accessors, no external buffer files, no
//! materials or textures. Draco is a whole decompressor and is refused by name
//! rather than silently producing an empty mesh; the others are rare enough in
//! published asset sets that the error message is a better use of the lines
//! than the implementation.

use crate::mesh::Mesh;
use crate::network::NetworkError;
use serde_json::Value;
use std::path::Path;

/// A parsed GLB file: its JSON description and its binary chunk.
pub struct Glb {
    json: Value,
    binary: Vec<u8>,
}

const MAGIC: u32 = 0x4654_6C67;
const JSON_CHUNK: u32 = 0x4E4F_534A;
const BINARY_CHUNK: u32 = 0x004E_4942;
/// What this writer emitted before: the four bytes `BIN\0` read as two. Files
/// written then are still read, because nothing else about them was wrong.
const LEGACY_BINARY_CHUNK: u32 = 0x0000_4E42;

impl Glb {
    /// Reads a `.glb` file.
    pub fn read<P: AsRef<Path>>(path: P) -> Result<Self, NetworkError> {
        Self::parse(std::fs::read(path)?)
    }

    /// The same, from bytes that are already in memory.
    pub fn parse(bytes: Vec<u8>) -> Result<Self, NetworkError> {
        let word = |at: usize| -> Result<u32, NetworkError> {
            bytes
                .get(at..at + 4)
                .map(|slice| u32::from_le_bytes(slice.try_into().unwrap()))
                .ok_or_else(|| NetworkError::InvalidDataset("the file ends mid-header".into()))
        };
        if word(0)? != MAGIC {
            return Err(NetworkError::InvalidDataset(
                "this is not a GLB file: the magic word is wrong".into(),
            ));
        }
        if word(4)? != 2 {
            return Err(NetworkError::InvalidDataset(format!(
                "GLB version {}, and only 2 is defined",
                word(4)?
            )));
        }

        let mut json = None;
        let mut binary = Vec::new();
        let mut at = 12;
        while at + 8 <= bytes.len() {
            let length = word(at)? as usize;
            let kind = word(at + 4)?;
            let start = at + 8;
            let chunk = bytes
                .get(start..start + length)
                .ok_or_else(|| NetworkError::InvalidDataset("a chunk runs past the file".into()))?;
            match kind {
                JSON_CHUNK => json = Some(serde_json::from_slice::<Value>(chunk)?),
                BINARY_CHUNK | LEGACY_BINARY_CHUNK => binary = chunk.to_vec(),
                // Unknown chunk types are to be skipped, says the format.
                _ => {}
            }
            // Chunks are padded to a four-byte boundary.
            at = start + length.next_multiple_of(4);
        }

        Ok(Self {
            json: json
                .ok_or_else(|| NetworkError::InvalidDataset("the file has no JSON chunk".into()))?,
            binary,
        })
    }

    /// The whole scene as one mesh, with every node's transform applied.
    ///
    /// Primitives are concatenated and their indices renumbered, so what comes
    /// back is a single surface whatever the file's node structure was.
    pub fn mesh(&self) -> Result<Mesh, NetworkError> {
        let scene = self.json["scene"].as_u64().unwrap_or(0) as usize;
        let roots = self.json["scenes"][scene]["nodes"]
            .as_array()
            .ok_or_else(|| NetworkError::InvalidDataset("the file has no scene".into()))?
            .clone();

        let mut mesh = Mesh::default();
        for root in roots {
            let index = root.as_u64().ok_or_else(|| {
                NetworkError::InvalidDataset("a scene names a node that is not a number".into())
            })? as usize;
            self.walk(index, IDENTITY, &mut mesh)?;
        }
        if mesh.indices.is_empty() {
            return Err(NetworkError::InvalidDataset(
                "the scene holds no triangles".into(),
            ));
        }
        // Half a vertex's attributes is worse than none: one primitive with
        // normals and one without would leave the arrays out of step.
        if mesh.normals.len() != mesh.positions.len() {
            mesh.normals.clear();
        }
        if mesh.uvs.len() != mesh.positions.len() {
            mesh.uvs.clear();
        }
        if mesh.colors.len() != mesh.positions.len() {
            mesh.colors.clear();
        }
        Ok(mesh)
    }

    fn walk(&self, index: usize, parent: [f32; 16], mesh: &mut Mesh) -> Result<(), NetworkError> {
        let node = &self.json["nodes"][index];
        let local = node_transform(node);
        let transform = multiply(parent, local);

        if let Some(which) = node["mesh"].as_u64() {
            let primitives = self.json["meshes"][which as usize]["primitives"]
                .as_array()
                .ok_or_else(|| {
                    NetworkError::InvalidDataset(format!("mesh {which} has no primitives"))
                })?;
            for primitive in primitives {
                self.append(primitive, transform, mesh)?;
            }
        }

        if let Some(children) = node["children"].as_array() {
            for child in children {
                let child = child.as_u64().ok_or_else(|| {
                    NetworkError::InvalidDataset("a child index is not a number".into())
                })? as usize;
                self.walk(child, transform, mesh)?;
            }
        }
        Ok(())
    }

    fn append(
        &self,
        primitive: &Value,
        transform: [f32; 16],
        mesh: &mut Mesh,
    ) -> Result<(), NetworkError> {
        if primitive["extensions"]
            .get("KHR_draco_mesh_compression")
            .is_some()
        {
            return Err(NetworkError::InvalidDataset(
                "this file is Draco compressed, which is not read here".into(),
            ));
        }
        // 4 is TRIANGLES, and is the default when the mode is left out.
        match primitive["mode"].as_u64().unwrap_or(4) {
            4 => {}
            mode => {
                return Err(NetworkError::InvalidDataset(format!(
                    "primitive mode {mode} is not triangles"
                )));
            }
        }

        let Some(position) = primitive["attributes"]["POSITION"].as_u64() else {
            return Ok(());
        };
        let positions = self.floats(position as usize, 3)?;
        let base = mesh.positions.len() as u32;
        for corner in positions.chunks_exact(3) {
            mesh.positions.push(transform_point(
                transform,
                [corner[0], corner[1], corner[2]],
            ));
        }

        if let Some(normal) = primitive["attributes"]["NORMAL"].as_u64() {
            for corner in self.floats(normal as usize, 3)?.chunks_exact(3) {
                mesh.normals.push(transform_normal(
                    transform,
                    [corner[0], corner[1], corner[2]],
                ));
            }
        }
        if let Some(uv) = primitive["attributes"]["TEXCOORD_0"].as_u64() {
            for corner in self.floats(uv as usize, 2)?.chunks_exact(2) {
                mesh.uvs.push([corner[0], corner[1]]);
            }
        }

        let added = mesh.positions.len() - base as usize;
        for colour in self.colours(primitive, added)? {
            mesh.colors.push(colour);
        }

        let count = (mesh.positions.len() as u32 - base) as usize;
        match primitive["indices"].as_u64() {
            Some(indices) => {
                let indices = self.integers(indices as usize)?;
                for face in indices.chunks_exact(3) {
                    mesh.indices
                        .push([base + face[0], base + face[1], base + face[2]]);
                }
            }
            // No index buffer means the vertices are already in triangle order.
            None => {
                for face in 0..count / 3 {
                    let first = base + face as u32 * 3;
                    mesh.indices.push([first, first + 1, first + 2]);
                }
            }
        }
        Ok(())
    }

    /// One colour per vertex of a primitive, or nothing if the file says none.
    ///
    /// `COLOR_0` first, which a vertex-painted asset carries directly. Failing
    /// that, the material's flat base colour, which is what most exported
    /// assets have and is still a better answer than grey. A base colour
    /// *texture* is not read: sampling it needs an image decoder and the UVs,
    /// and the stand-in this feeds — per-vertex colour regressed off the shape
    /// latent — cannot represent the detail a texture holds anyway.
    ///
    /// NOTE: no `KHR_materials_*` extension, no texture sampling. Both
    /// belong with the rasterizer the texture stage would bring.
    fn colours(&self, primitive: &Value, vertices: usize) -> Result<Vec<[f32; 3]>, NetworkError> {
        if let Some(attribute) = primitive["attributes"]["COLOR_0"].as_u64() {
            // The spec allows VEC3 and VEC4 here, and the alpha is dropped
            // either way: this pipeline has no transparency.
            let components = self.accessor(attribute as usize)?.components;
            let values = self.floats(attribute as usize, components)?;
            return Ok(values
                .chunks_exact(components)
                .map(|colour| [colour[0], colour[1], colour[2]])
                .collect());
        }

        let Some(material) = primitive["material"].as_u64() else {
            return Ok(Vec::new());
        };
        let factor =
            &self.json["materials"][material as usize]["pbrMetallicRoughness"]["baseColorFactor"];
        // A material with no factor is the spec's default, which is white.
        let colour = match factor.as_array() {
            Some(values) if values.len() >= 3 => {
                std::array::from_fn(|channel| values[channel].as_f64().unwrap_or(1.0) as f32)
            }
            _ => [1.0, 1.0, 1.0],
        };
        Ok(vec![colour; vertices])
    }

    /// An accessor of floating-point vectors, widened to `f32` and flattened.
    fn floats(&self, index: usize, components: usize) -> Result<Vec<f32>, NetworkError> {
        let accessor = self.accessor(index)?;
        if accessor.components != components {
            return Err(NetworkError::InvalidDataset(format!(
                "accessor {index} holds {}-vectors and a {components}-vector was wanted",
                accessor.components
            )));
        }
        self.elements(&accessor, |bytes, kind| match kind {
            5126 => f32::from_le_bytes(bytes[..4].try_into().unwrap()),
            // Normalized integer attributes, which is how compact files store
            // texture coordinates.
            5121 => f32::from(bytes[0]) / 255.0,
            5123 => f32::from(u16::from_le_bytes(bytes[..2].try_into().unwrap())) / 65535.0,
            5120 => (f32::from(bytes[0] as i8) / 127.0).max(-1.0),
            5122 => {
                (f32::from(i16::from_le_bytes(bytes[..2].try_into().unwrap())) / 32767.0).max(-1.0)
            }
            _ => 0.0,
        })
    }

    /// An accessor of scalar indices, widened to `u32`.
    fn integers(&self, index: usize) -> Result<Vec<u32>, NetworkError> {
        let accessor = self.accessor(index)?;
        if accessor.components != 1 {
            return Err(NetworkError::InvalidDataset(format!(
                "accessor {index} is not scalar, so it is not an index buffer"
            )));
        }
        Ok(self
            .elements(&accessor, |bytes, kind| match kind {
                5121 => f32::from(bytes[0]),
                5123 => f32::from(u16::from_le_bytes(bytes[..2].try_into().unwrap())),
                5125 => u32::from_le_bytes(bytes[..4].try_into().unwrap()) as f32,
                _ => 0.0,
            })?
            .into_iter()
            .map(|value| value as u32)
            .collect())
    }

    fn accessor(&self, index: usize) -> Result<Accessor, NetworkError> {
        let accessor = &self.json["accessors"][index];
        if accessor.is_null() {
            return Err(NetworkError::InvalidDataset(format!(
                "accessor {index} is not in the file"
            )));
        }
        if accessor.get("sparse").is_some() {
            return Err(NetworkError::InvalidDataset(
                "sparse accessors are not read here".into(),
            ));
        }
        let kind = accessor["componentType"].as_u64().unwrap_or(0) as u32;
        let width = match kind {
            5120 | 5121 => 1,
            5122 | 5123 => 2,
            5125 | 5126 => 4,
            other => {
                return Err(NetworkError::InvalidDataset(format!(
                    "component type {other} is not one glTF defines"
                )));
            }
        };
        let components = match accessor["type"].as_str().unwrap_or("") {
            "SCALAR" => 1,
            "VEC2" => 2,
            "VEC3" => 3,
            "VEC4" => 4,
            "MAT4" => 16,
            other => {
                return Err(NetworkError::InvalidDataset(format!(
                    "accessor type {other} is not read here"
                )));
            }
        };

        let view = &self.json["bufferViews"][accessor["bufferView"].as_u64().unwrap_or(0) as usize];
        let stride = view["byteStride"].as_u64().unwrap_or(0) as usize;
        Ok(Accessor {
            kind,
            components,
            count: accessor["count"].as_u64().unwrap_or(0) as usize,
            offset: view["byteOffset"].as_u64().unwrap_or(0) as usize
                + accessor["byteOffset"].as_u64().unwrap_or(0) as usize,
            // A stride of zero means tightly packed, which is the common case.
            stride: match stride {
                0 => width * components,
                stride => stride,
            },
            width,
        })
    }

    fn elements(
        &self,
        accessor: &Accessor,
        widen: impl Fn(&[u8], u32) -> f32,
    ) -> Result<Vec<f32>, NetworkError> {
        let mut values = Vec::with_capacity(accessor.count * accessor.components);
        for element in 0..accessor.count {
            let start = accessor.offset + element * accessor.stride;
            for component in 0..accessor.components {
                let at = start + component * accessor.width;
                let bytes = self.binary.get(at..at + accessor.width).ok_or_else(|| {
                    NetworkError::InvalidDataset(
                        "an accessor reads past the end of the buffer".into(),
                    )
                })?;
                values.push(widen(bytes, accessor.kind));
            }
        }
        Ok(values)
    }
}

/// One typed view onto the binary chunk, with the JSON already resolved.
struct Accessor {
    kind: u32,
    components: usize,
    count: usize,
    offset: usize,
    stride: usize,
    width: usize,
}

impl Mesh {
    /// Reads the whole scene of a `.glb` file as one mesh.
    pub fn read_glb<P: AsRef<Path>>(path: P) -> Result<Self, NetworkError> {
        Glb::read(path)?.mesh()
    }
}

impl Glb {
    /// A GLB from a scene description and the bytes it points into.
    pub fn new(json: Value, binary: Vec<u8>) -> Self {
        Self { json, binary }
    }

    /// The container: header, JSON chunk, binary chunk.
    ///
    /// Both chunks are padded to four bytes — the JSON with spaces so it still
    /// parses, the binary with zeros — because the header's lengths are read
    /// as words and a reader is entitled to assume the alignment.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut json = serde_json::to_vec(&self.json).unwrap_or_default();
        json.resize(json.len().next_multiple_of(4), b' ');
        let mut binary = self.binary.clone();
        binary.resize(binary.len().next_multiple_of(4), 0);

        let mut bytes = Vec::with_capacity(28 + json.len() + binary.len());
        bytes.extend(MAGIC.to_le_bytes());
        bytes.extend(2u32.to_le_bytes());
        bytes.extend(((28 + json.len() + binary.len()) as u32).to_le_bytes());
        bytes.extend((json.len() as u32).to_le_bytes());
        bytes.extend(JSON_CHUNK.to_le_bytes());
        bytes.extend(&json);
        bytes.extend((binary.len() as u32).to_le_bytes());
        bytes.extend(BINARY_CHUNK.to_le_bytes());
        bytes.extend(&binary);
        bytes
    }

    pub fn write<P: AsRef<Path>>(&self, path: P) -> Result<(), NetworkError> {
        Ok(std::fs::write(path, self.to_bytes())?)
    }
}

/// One accessor being assembled: where its bytes went and what they are.
struct Written {
    view: usize,
    kind: u32,
    components: &'static str,
    count: usize,
}

impl Mesh {
    /// Writes the mesh as a `.glb`.
    ///
    /// One node, one mesh, one triangle primitive, one buffer view per
    /// attribute, and one material — an untextured PBR white, because nothing
    /// in this pipeline produces a texture or the UVs that would place one.
    /// Normals, UVs and vertex colours are written when the mesh carries them;
    /// a viewer multiplies `COLOR_0` by the material's base colour, which is
    /// why that base colour is white.
    pub fn write_glb<P: AsRef<Path>>(&self, path: P) -> Result<(), NetworkError> {
        self.to_glb()?.write(path)
    }

    /// The same, kept in memory.
    pub fn to_glb(&self) -> Result<Glb, NetworkError> {
        if self.indices.is_empty() || self.positions.is_empty() {
            return Err(NetworkError::InvalidDataset(
                "a glTF primitive needs at least one triangle".into(),
            ));
        }
        let vertices = self.positions.len();
        for face in &self.indices {
            for corner in face {
                if *corner as usize >= vertices {
                    return Err(NetworkError::InvalidDataset(format!(
                        "a face names vertex {corner} of {vertices}"
                    )));
                }
            }
        }
        if !self.normals.is_empty() && self.normals.len() != vertices {
            return Err(NetworkError::InvalidDataset(format!(
                "{} normals for {vertices} vertices",
                self.normals.len()
            )));
        }
        if !self.uvs.is_empty() && self.uvs.len() != vertices {
            return Err(NetworkError::InvalidDataset(format!(
                "{} texture coordinates for {vertices} vertices",
                self.uvs.len()
            )));
        }
        if !self.colors.is_empty() && self.colors.len() != vertices {
            return Err(NetworkError::InvalidDataset(format!(
                "{} vertex colours for {vertices} vertices",
                self.colors.len()
            )));
        }

        let mut binary: Vec<u8> = Vec::new();
        let mut views = Vec::new();
        // Every accessor here is four bytes wide, so padding each view to four
        // keeps all of them aligned to their component size.
        let mut push = |binary: &mut Vec<u8>, bytes: Vec<u8>, target: u32| {
            binary.resize(binary.len().next_multiple_of(4), 0);
            let view = serde_json::json!({
                "buffer": 0,
                "byteOffset": binary.len(),
                "byteLength": bytes.len(),
                "target": target,
            });
            binary.extend(bytes);
            views.push(view);
            views.len() - 1
        };

        const ARRAY: u32 = 34962;
        const ELEMENT_ARRAY: u32 = 34963;
        const FLOAT: u32 = 5126;
        const UNSIGNED_INT: u32 = 5125;

        let index_view = push(
            &mut binary,
            self.indices
                .iter()
                .flatten()
                .flat_map(|corner| corner.to_le_bytes())
                .collect(),
            ELEMENT_ARRAY,
        );
        let mut written = vec![Written {
            view: index_view,
            kind: UNSIGNED_INT,
            components: "SCALAR",
            count: self.indices.len() * 3,
        }];

        let mut attributes = serde_json::Map::new();
        let mut attribute = |binary: &mut Vec<u8>,
                             written: &mut Vec<Written>,
                             name: &str,
                             data: Vec<u8>,
                             components: &'static str,
                             count: usize| {
            let view = push(binary, data, ARRAY);
            written.push(Written {
                view,
                kind: FLOAT,
                components,
                count,
            });
            attributes.insert(name.into(), (written.len() - 1).into());
        };

        attribute(
            &mut binary,
            &mut written,
            "POSITION",
            self.positions
                .iter()
                .flatten()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
            "VEC3",
            vertices,
        );
        if !self.normals.is_empty() {
            attribute(
                &mut binary,
                &mut written,
                "NORMAL",
                self.normals
                    .iter()
                    .flatten()
                    .flat_map(|value| value.to_le_bytes())
                    .collect(),
                "VEC3",
                vertices,
            );
        }
        if !self.uvs.is_empty() {
            attribute(
                &mut binary,
                &mut written,
                "TEXCOORD_0",
                self.uvs
                    .iter()
                    .flatten()
                    .flat_map(|value| value.to_le_bytes())
                    .collect(),
                "VEC2",
                vertices,
            );
        }
        if !self.colors.is_empty() {
            attribute(
                &mut binary,
                &mut written,
                "COLOR_0",
                self.colors
                    .iter()
                    .flatten()
                    // A viewer clamps this anyway, and a regressed colour can
                    // land slightly outside the range it was trained on.
                    .flat_map(|value| value.clamp(0.0, 1.0).to_le_bytes())
                    .collect(),
                "VEC3",
                vertices,
            );
        }

        // The position accessor has to carry the bounding box: a viewer sizes
        // its camera from it without touching the binary chunk, and a
        // validator rejects the file without it.
        let (low, high) = self.bounds();
        let accessors: Vec<Value> = written
            .iter()
            .enumerate()
            .map(|(index, written)| {
                let mut accessor = serde_json::json!({
                    "bufferView": written.view,
                    "componentType": written.kind,
                    "count": written.count,
                    "type": written.components,
                });
                if index == 1 {
                    accessor["min"] = low.to_vec().into();
                    accessor["max"] = high.to_vec().into();
                }
                accessor
            })
            .collect();

        let json = serde_json::json!({
            "asset": { "version": "2.0", "generator": "rusting_brain" },
            "scene": 0,
            "scenes": [{ "nodes": [0] }],
            "nodes": [{ "mesh": 0 }],
            "meshes": [{ "primitives": [{
                "attributes": attributes,
                "indices": 0,
                "material": 0,
                "mode": 4,
            }] }],
            "materials": [{
                "pbrMetallicRoughness": {
                    "baseColorFactor": [0.8, 0.8, 0.8, 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.8,
                },
            }],
            "buffers": [{ "byteLength": binary.len().next_multiple_of(4) }],
            "bufferViews": views,
            "accessors": accessors,
        });

        Ok(Glb::new(json, binary))
    }
}

/// glTF stores matrices column-major, so this is the column-major identity.
const IDENTITY: [f32; 16] = [
    1.0, 0.0, 0.0, 0.0, //
    0.0, 1.0, 0.0, 0.0, //
    0.0, 0.0, 1.0, 0.0, //
    0.0, 0.0, 0.0, 1.0,
];

/// A node's local transform: either the matrix it states, or the translation,
/// rotation and scale it states instead.
fn node_transform(node: &Value) -> [f32; 16] {
    if let Some(matrix) = node["matrix"].as_array() {
        let mut out = IDENTITY;
        for (slot, value) in out.iter_mut().zip(matrix) {
            *slot = value.as_f64().unwrap_or(0.0) as f32;
        }
        return out;
    }

    let read = |name: &str, fallback: [f32; 4]| -> [f32; 4] {
        match node[name].as_array() {
            Some(values) => {
                let mut out = fallback;
                for (slot, value) in out.iter_mut().zip(values) {
                    *slot = value.as_f64().unwrap_or(0.0) as f32;
                }
                out
            }
            None => fallback,
        }
    };
    let translation = read("translation", [0.0; 4]);
    let rotation = read("rotation", [0.0, 0.0, 0.0, 1.0]);
    let scale = read("scale", [1.0, 1.0, 1.0, 1.0]);

    // The quaternion as a rotation matrix, then scaled by column and given the
    // translation, which is the order glTF defines: T * R * S.
    let [x, y, z, w] = rotation;
    let rotation = [
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y + z * w),
        2.0 * (x * z - y * w),
        0.0,
        2.0 * (x * y - z * w),
        1.0 - 2.0 * (x * x + z * z),
        2.0 * (y * z + x * w),
        0.0,
        2.0 * (x * z + y * w),
        2.0 * (y * z - x * w),
        1.0 - 2.0 * (x * x + y * y),
        0.0,
        translation[0],
        translation[1],
        translation[2],
        1.0,
    ];
    let mut out = rotation;
    for column in 0..3 {
        for row in 0..3 {
            out[column * 4 + row] *= scale[column];
        }
    }
    out
}

/// Column-major matrix product, `a` applied after `b`.
fn multiply(a: [f32; 16], b: [f32; 16]) -> [f32; 16] {
    let mut out = [0.0; 16];
    for column in 0..4 {
        for row in 0..4 {
            out[column * 4 + row] = (0..4).map(|k| a[k * 4 + row] * b[column * 4 + k]).sum();
        }
    }
    out
}

fn transform_point(matrix: [f32; 16], point: [f32; 3]) -> [f32; 3] {
    std::array::from_fn(|row| {
        (0..3).map(|k| matrix[k * 4 + row] * point[k]).sum::<f32>() + matrix[12 + row]
    })
}

/// NOTE: the upper three by three, not its inverse transpose, so a node
/// with a non-uniform scale tilts its normals. Nothing here reads normals —
/// the distance field comes from the triangles — and [`Mesh::recompute_normals`]
/// is the fix for a caller that does.
fn transform_normal(matrix: [f32; 16], normal: [f32; 3]) -> [f32; 3] {
    let rotated: [f32; 3] =
        std::array::from_fn(|row| (0..3).map(|k| matrix[k * 4 + row] * normal[k]).sum());
    let length = rotated
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    match length == 0.0 {
        true => [0.0; 3],
        false => rotated.map(|value| value / length),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a GLB out of a JSON description and a binary chunk, the way a
    /// real exporter would, so the tests exercise the actual byte layout.
    fn glb(json: &str, binary: &[u8]) -> Vec<u8> {
        let mut json = json.as_bytes().to_vec();
        json.resize(json.len().next_multiple_of(4), b' ');
        let mut binary = binary.to_vec();
        binary.resize(binary.len().next_multiple_of(4), 0);

        let mut bytes = Vec::new();
        bytes.extend(MAGIC.to_le_bytes());
        bytes.extend(2u32.to_le_bytes());
        bytes.extend(((12 + 8 + json.len() + 8 + binary.len()) as u32).to_le_bytes());
        bytes.extend((json.len() as u32).to_le_bytes());
        bytes.extend(JSON_CHUNK.to_le_bytes());
        bytes.extend(&json);
        bytes.extend((binary.len() as u32).to_le_bytes());
        bytes.extend(BINARY_CHUNK.to_le_bytes());
        bytes.extend(&binary);
        bytes
    }

    #[test]
    fn a_mesh_survives_a_round_trip_through_a_written_glb() {
        let mut original = Mesh {
            positions: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            indices: vec![[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]],
            uvs: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            ..Mesh::default()
        };
        original.recompute_normals();

        let bytes = original.to_glb().unwrap().to_bytes();
        let read = Glb::parse(bytes).unwrap().mesh().unwrap();

        assert_eq!(read.positions, original.positions);
        assert_eq!(read.indices, original.indices);
        assert_eq!(read.uvs, original.uvs);
        for (read, original) in read.normals.iter().zip(&original.normals) {
            for axis in 0..3 {
                assert!((read[axis] - original[axis]).abs() < 1e-6);
            }
        }
    }

    /// A mesh with no normals and no UVs is the common case coming out of
    /// marching tetrahedra, and it has to write as well as a complete one.
    #[test]
    fn a_bare_mesh_writes_only_the_attributes_it_has() {
        let mesh = Mesh {
            positions: vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0]],
            indices: vec![[0, 1, 2]],
            ..Mesh::default()
        };

        let glb = mesh.to_glb().unwrap();
        let attributes = &glb.json["meshes"][0]["primitives"][0]["attributes"];
        assert!(attributes.get("POSITION").is_some());
        assert!(attributes.get("NORMAL").is_none());
        assert!(attributes.get("TEXCOORD_0").is_none());
        assert!(attributes.get("COLOR_0").is_none());

        // The bounding box the spec requires on the position accessor.
        assert_eq!(
            glb.json["accessors"][1]["min"],
            serde_json::json!([0.0, 0.0, 0.0])
        );
        assert_eq!(
            glb.json["accessors"][1]["max"],
            serde_json::json!([2.0, 3.0, 0.0])
        );

        let read = Glb::parse(glb.to_bytes()).unwrap().mesh().unwrap();
        assert_eq!(read.positions, mesh.positions);
        assert_eq!(read.indices, mesh.indices);
        assert!(read.normals.is_empty());
    }

    /// Vertex colours are the pipeline's stand-in for a texture, so they have
    /// to survive the trip out and back in.
    #[test]
    fn vertex_colours_survive_a_round_trip_through_a_written_glb() {
        let mesh = Mesh {
            positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            indices: vec![[0, 1, 2]],
            colors: vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.25, 0.5, 0.75]],
            ..Mesh::default()
        };

        let glb = mesh.to_glb().unwrap();
        assert!(glb.json["meshes"][0]["primitives"][0]["attributes"]["COLOR_0"].is_number());

        let read = Glb::parse(glb.to_bytes()).unwrap().mesh().unwrap();
        assert_eq!(read.colors, mesh.colors);
    }

    /// A regressed colour can land a little outside the range it was trained
    /// on, and the file should still hold a legal colour.
    #[test]
    fn a_colour_outside_the_unit_range_is_clamped_on_the_way_out() {
        let mesh = Mesh {
            positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            indices: vec![[0, 1, 2]],
            colors: vec![[-0.2, 1.4, 0.5]; 3],
            ..Mesh::default()
        };

        let read = Glb::parse(mesh.to_glb().unwrap().to_bytes())
            .unwrap()
            .mesh()
            .unwrap();
        assert_eq!(read.colors, vec![[0.0, 1.0, 0.5]; 3]);
    }

    /// Most meshes in the wild carry no COLOR_0 at all, only a flat base
    /// colour on the material, and that is the colour the mesh should get.
    #[test]
    fn a_material_base_colour_stands_in_for_missing_vertex_colours() {
        let mesh = Mesh {
            positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            indices: vec![[0, 1, 2]],
            ..Mesh::default()
        };
        let mut glb = mesh.to_glb().unwrap();
        glb.json["materials"][0]["pbrMetallicRoughness"]["baseColorFactor"] =
            serde_json::json!([0.2, 0.4, 0.6, 1.0]);

        let read = Glb::parse(glb.to_bytes()).unwrap().mesh().unwrap();
        assert_eq!(read.colors, vec![[0.2, 0.4, 0.6]; 3]);
    }

    /// The header and chunk layout, checked as bytes rather than through the
    /// reader, since a reader that shares the writer's mistake agrees with it.
    #[test]
    fn a_written_glb_has_the_header_and_padding_the_spec_asks_for() {
        let mesh = Mesh {
            positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            indices: vec![[0, 1, 2]],
            ..Mesh::default()
        };
        let bytes = mesh.to_glb().unwrap().to_bytes();

        let word = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
        assert_eq!(word(0), MAGIC);
        assert_eq!(word(4), 2);
        assert_eq!(word(8) as usize, bytes.len());
        assert_eq!(bytes.len() % 4, 0);

        let json_length = word(12) as usize;
        assert_eq!(word(16), JSON_CHUNK);
        assert_eq!(json_length % 4, 0);
        // The JSON is padded with spaces, so it still parses as it stands.
        serde_json::from_slice::<Value>(&bytes[20..20 + json_length]).unwrap();

        let binary_at = 20 + json_length;
        assert_eq!(word(binary_at + 4), BINARY_CHUNK);
        assert_eq!(word(binary_at) as usize % 4, 0);
        assert_eq!(binary_at + 8 + word(binary_at) as usize, bytes.len());
    }

    #[test]
    fn a_mesh_that_is_not_a_mesh_is_refused() {
        assert!(Mesh::default().to_glb().is_err());

        let broken = Mesh {
            positions: vec![[0.0, 0.0, 0.0]],
            indices: vec![[0, 1, 2]],
            ..Mesh::default()
        };
        let error = match broken.to_glb() {
            Ok(_) => panic!("a face naming a vertex that is not there was written"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("vertex 1 of 1"), "{error}");

        let mismatched = Mesh {
            positions: vec![[0.0; 3], [1.0; 3], [2.0; 3]],
            indices: vec![[0, 1, 2]],
            normals: vec![[0.0, 0.0, 1.0]],
            ..Mesh::default()
        };
        assert!(mismatched.to_glb().is_err());
    }

    /// The whole geometry path end to end: a distance field becomes a mesh,
    /// the mesh is simplified, and what comes back out of the file is still
    /// the same surface.
    #[test]
    fn a_simplified_sphere_writes_and_reads_back_as_the_same_surface() {
        let mut mesh = crate::mesh::marching_tetrahedra(
            |points| {
                Ok((0..points.rows)
                    .map(|row| {
                        let point = points.row(row);
                        (point[0] * point[0] + point[1] * point[1] + point[2] * point[2]).sqrt()
                            - 0.7
                    })
                    .collect())
            },
            20,
            ([-1.0; 3], [1.0; 3]),
            0.0,
        )
        .unwrap();
        mesh.simplify(mesh.indices.len() / 3);
        mesh.recompute_normals();

        let read = Glb::parse(mesh.to_glb().unwrap().to_bytes())
            .unwrap()
            .mesh()
            .unwrap();
        assert_eq!(read.indices.len(), mesh.indices.len());
        for point in &read.positions {
            let radius = (point[0] * point[0] + point[1] * point[1] + point[2] * point[2]).sqrt();
            assert!((radius - 0.7).abs() < 0.1, "{point:?}");
        }
    }

    /// One triangle: three `f32` positions then three `u16` indices.
    fn triangle_binary() -> Vec<u8> {
        let positions: [f32; 9] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let mut bytes: Vec<u8> = positions.iter().flat_map(|v| v.to_le_bytes()).collect();
        bytes.extend([0u16, 1, 2].iter().flat_map(|v| v.to_le_bytes()));
        bytes
    }

    fn triangle_json(node: &str) -> String {
        format!(
            r#"{{
              "scene": 0,
              "scenes": [{{"nodes": [0]}}],
              "nodes": [{{"mesh": 0 {node}}}],
              "meshes": [{{"primitives": [{{"attributes": {{"POSITION": 0}}, "indices": 1}}]}}],
              "accessors": [
                {{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"}},
                {{"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"}}
              ],
              "bufferViews": [
                {{"buffer": 0, "byteOffset": 0, "byteLength": 36}},
                {{"buffer": 0, "byteOffset": 36, "byteLength": 6}}
              ],
              "buffers": [{{"byteLength": 42}}]
            }}"#
        )
    }

    #[test]
    fn a_triangle_comes_back_with_its_indices() {
        let bytes = glb(&triangle_json(""), &triangle_binary());
        let mesh = Glb::parse(bytes).unwrap().mesh().unwrap();

        assert_eq!(mesh.positions.len(), 3);
        assert_eq!(mesh.indices, vec![[0, 1, 2]]);
        assert_eq!(mesh.triangle(0)[1], [1.0, 0.0, 0.0]);
        assert!(mesh.normals.is_empty());
    }

    #[test]
    fn a_nodes_translation_and_scale_move_its_vertices() {
        let node = r#", "translation": [5.0, 0.0, 0.0], "scale": [2.0, 3.0, 1.0]"#;
        let bytes = glb(&triangle_json(node), &triangle_binary());
        let mesh = Glb::parse(bytes).unwrap().mesh().unwrap();

        // Scale first, then translate: (1, 0, 0) scales to (2, 0, 0).
        assert_eq!(mesh.positions[0], [5.0, 0.0, 0.0]);
        assert_eq!(mesh.positions[1], [7.0, 0.0, 0.0]);
        assert_eq!(mesh.positions[2], [5.0, 3.0, 0.0]);
    }

    #[test]
    fn a_quarter_turn_about_z_sends_x_to_y() {
        // The quaternion for ninety degrees about z.
        let half = std::f32::consts::FRAC_1_SQRT_2;
        let node = format!(r#", "rotation": [0.0, 0.0, {half}, {half}]"#);
        let bytes = glb(&triangle_json(&node), &triangle_binary());
        let mesh = Glb::parse(bytes).unwrap().mesh().unwrap();

        let moved = mesh.positions[1];
        assert!((moved[0] - 0.0).abs() < 1e-6, "{moved:?}");
        assert!((moved[1] - 1.0).abs() < 1e-6, "{moved:?}");
    }

    #[test]
    fn a_child_node_inherits_its_parents_transform() {
        let json = r#"{
          "scene": 0,
          "scenes": [{"nodes": [0]}],
          "nodes": [
            {"translation": [0.0, 10.0, 0.0], "children": [1]},
            {"mesh": 0, "translation": [0.0, 1.0, 0.0]}
          ],
          "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
          "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
            {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"}
          ],
          "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": 36},
            {"buffer": 0, "byteOffset": 36, "byteLength": 6}
          ],
          "buffers": [{"byteLength": 42}]
        }"#;
        let bytes = glb(json, &triangle_binary());
        let mesh = Glb::parse(bytes).unwrap().mesh().unwrap();

        assert_eq!(mesh.positions[0], [0.0, 11.0, 0.0]);
    }

    #[test]
    fn an_interleaved_buffer_is_read_through_its_stride() {
        // Position and normal packed together, twenty-four bytes per vertex.
        let vertices: [[f32; 6]; 3] = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        ];
        let mut binary: Vec<u8> = vertices
            .iter()
            .flat_map(|vertex| vertex.iter().flat_map(|v| v.to_le_bytes()))
            .collect();
        binary.extend([0u16, 1, 2].iter().flat_map(|v| v.to_le_bytes()));

        let json = r#"{
          "scene": 0,
          "scenes": [{"nodes": [0]}],
          "nodes": [{"mesh": 0}],
          "meshes": [{"primitives": [
            {"attributes": {"POSITION": 0, "NORMAL": 1}, "indices": 2}
          ]}],
          "accessors": [
            {"bufferView": 0, "byteOffset": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
            {"bufferView": 0, "byteOffset": 12, "componentType": 5126, "count": 3, "type": "VEC3"},
            {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"}
          ],
          "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": 72, "byteStride": 24},
            {"buffer": 0, "byteOffset": 72, "byteLength": 6}
          ],
          "buffers": [{"byteLength": 78}]
        }"#;
        let mesh = Glb::parse(glb(json, &binary)).unwrap().mesh().unwrap();

        assert_eq!(mesh.positions[2], [0.0, 1.0, 0.0]);
        assert_eq!(mesh.normals[2], [0.0, 0.0, 1.0]);
    }

    #[test]
    fn a_file_that_is_not_a_glb_is_refused() {
        let error = match Glb::parse(b"not a gltf file at all".to_vec()) {
            Ok(_) => panic!("a file with no magic word was read as a GLB"),
            Err(error) => error,
        };
        assert!(format!("{error}").contains("magic"), "{error}");
    }

    #[test]
    fn a_draco_compressed_primitive_is_refused_by_name() {
        let json = r#"{
          "scene": 0,
          "scenes": [{"nodes": [0]}],
          "nodes": [{"mesh": 0}],
          "meshes": [{"primitives": [{
            "attributes": {"POSITION": 0},
            "extensions": {"KHR_draco_mesh_compression": {"bufferView": 0}}
          }]}],
          "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"}],
          "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 36}],
          "buffers": [{"byteLength": 36}]
        }"#;
        let error = Glb::parse(glb(json, &triangle_binary()))
            .unwrap()
            .mesh()
            .unwrap_err();
        assert!(format!("{error}").contains("Draco"), "{error}");
    }

    #[test]
    fn an_accessor_that_reads_past_the_buffer_is_refused() {
        let json = r#"{
          "scene": 0,
          "scenes": [{"nodes": [0]}],
          "nodes": [{"mesh": 0}],
          "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
          "accessors": [{"bufferView": 0, "componentType": 5126, "count": 300, "type": "VEC3"}],
          "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 36}],
          "buffers": [{"byteLength": 36}]
        }"#;
        let error = Glb::parse(glb(json, &triangle_binary()))
            .unwrap()
            .mesh()
            .unwrap_err();
        assert!(format!("{error}").contains("past the end"), "{error}");
    }

    #[test]
    fn a_glb_survives_a_trip_through_the_distance_field() {
        // The geometry layer end to end from a real container: read a cube out
        // of a GLB, march its distance field back, and measure the surfaces.
        let mut binary: Vec<u8> = Vec::new();
        let corners: [[f32; 3]; 8] = [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ];
        for corner in corners {
            binary.extend(corner.iter().flat_map(|v| v.to_le_bytes()));
        }
        let faces: [[u16; 3]; 12] = [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [2, 3, 7],
            [2, 7, 6],
            [1, 2, 6],
            [1, 6, 5],
            [0, 4, 7],
            [0, 7, 3],
        ];
        for face in faces {
            binary.extend(face.iter().flat_map(|v| v.to_le_bytes()));
        }
        let json = r#"{
          "scene": 0,
          "scenes": [{"nodes": [0]}],
          "nodes": [{"mesh": 0}],
          "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
          "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 8, "type": "VEC3"},
            {"bufferView": 1, "componentType": 5123, "count": 36, "type": "SCALAR"}
          ],
          "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": 96},
            {"buffer": 0, "byteOffset": 96, "byteLength": 72}
          ],
          "buffers": [{"byteLength": 168}]
        }"#;
        let mut mesh = Glb::parse(glb(json, &binary)).unwrap().mesh().unwrap();
        assert_eq!(mesh.indices.len(), 12);

        mesh.normalize();
        let bvh = crate::mesh::Bvh::build(&mesh);
        let marched = crate::mesh::marching_tetrahedra(
            |queries| {
                let points: Vec<[f32; 3]> = (0..queries.rows)
                    .map(|row| {
                        let at = row * queries.cols;
                        [queries.data[at], queries.data[at + 1], queries.data[at + 2]]
                    })
                    .collect();
                Ok(bvh.signed_distance(&points))
            },
            48,
            ([-1.2; 3], [1.2; 3]),
            0.0,
        )
        .unwrap();

        // Every marched vertex should sit on the original cube's surface.
        let worst = marched
            .positions
            .iter()
            .map(|point| bvh.closest_point(*point).unwrap().1)
            .fold(0.0f32, f32::max);
        assert!(worst < 0.05, "worst vertex sits {worst} from the surface");
    }
}
