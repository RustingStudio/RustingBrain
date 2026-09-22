//! Triangle meshes, and turning them into the signed distances a shape model
//! trains on.
//!
//! Nothing here learns anything. It is the offline half of an image-to-3D
//! pipeline: read a mesh, put it in a unit cube, ask where its surface is, and
//! turn a field of distances back into a mesh at the end.
//!
//! The three pieces that carry the rest:
//!
//! - [`Bvh`] over the triangles, built once and asked two questions:
//!   [`Bvh::closest_point`] for how far a point is from the surface, and
//!   [`Bvh::winding_number`] for which side of it the point is on.
//! - The generalized winding number rather than a ray cast, because a real
//!   dataset is full of meshes with holes, duplicated faces and flipped
//!   normals. A ray cast needs a closed surface to count crossings against; the
//!   winding number degrades smoothly instead, reading a little over one inside
//!   a mesh with a hole in it rather than flipping to the wrong answer.
//! - [`marching_tetrahedra`] to go back, which is watertight by construction.
//!
//! ```no_run
//! # use rusting_brain::mesh::{Mesh, Bvh, QuerySampling};
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut mesh = Mesh::read_obj("chair.obj")?;
//! let transform = mesh.normalize();          // into [-1, 1], invertible
//! let bvh = Bvh::build(&mesh);
//!
//! let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(0);
//! let queries = bvh.sample_queries(&QuerySampling::default(), &mut rng);
//! let distances = bvh.signed_distance(&queries);
//! # let _ = (transform, distances);
//! # Ok(())
//! # }
//! ```

use crate::matrix::Matrix;
use crate::network::NetworkError;
use rand::Rng;
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::Path;

/// A triangle mesh, indexed.
///
/// `normals`, `uvs` and `colors` are either empty or as long as `positions`.
/// The geometry here needs none of them; they are carried so a reader and a
/// writer round-trip, and `colors` is what the vertex-colour stand-in for
/// texture is trained on and written from.
#[derive(Clone, Debug, Default)]
pub struct Mesh {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    /// Linear RGB per vertex, 0 to 1.
    pub colors: Vec<[f32; 3]>,
    pub indices: Vec<[u32; 3]>,
}

/// The similarity transform [`Mesh::normalize`] applied, kept so a mesh that
/// comes back out of the model can be put where the original was.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transform {
    pub center: [f32; 3],
    pub scale: f32,
}

impl Transform {
    /// Undoes the normalization: the point in the original mesh's coordinates.
    pub fn invert(&self, point: [f32; 3]) -> [f32; 3] {
        std::array::from_fn(|axis| point[axis] / self.scale + self.center[axis])
    }
}

impl Mesh {
    /// The three corners of a face.
    pub fn triangle(&self, face: usize) -> [[f32; 3]; 3] {
        let face = self.indices[face];
        std::array::from_fn(|corner| self.positions[face[corner] as usize])
    }

    /// The smallest axis-aligned box holding every vertex.
    pub fn bounds(&self) -> ([f32; 3], [f32; 3]) {
        let mut low = [f32::INFINITY; 3];
        let mut high = [f32::NEG_INFINITY; 3];
        for position in &self.positions {
            for axis in 0..3 {
                low[axis] = low[axis].min(position[axis]);
                high[axis] = high[axis].max(position[axis]);
            }
        }
        (low, high)
    }

    /// Centres the mesh on its bounding box and scales it to fill `[-1, 1]`,
    /// returning the transform that was applied.
    ///
    /// The longest axis reaches the faces of the cube and the others fall short
    /// of them, because scaling the axes separately would change the shape.
    pub fn normalize(&mut self) -> Transform {
        let (low, high) = self.bounds();
        let center: [f32; 3] = std::array::from_fn(|axis| (low[axis] + high[axis]) / 2.0);
        let extent = (0..3)
            .map(|axis| high[axis] - low[axis])
            .fold(0.0f32, f32::max);
        // A mesh with no extent at all is a point; scaling it is meaningless,
        // so it is left where it is rather than divided by zero.
        let scale = match extent > 0.0 {
            true => 2.0 / extent,
            false => 1.0,
        };
        for position in &mut self.positions {
            for axis in 0..3 {
                position[axis] = (position[axis] - center[axis]) * scale;
            }
        }
        Transform { center, scale }
    }

    /// Replaces the per-vertex normals with area-weighted face normals.
    ///
    /// Weighting by area rather than averaging the unit face normals is what
    /// keeps a vertex where one huge triangle meets several slivers from being
    /// dominated by the slivers.
    pub fn recompute_normals(&mut self) {
        let mut normals = vec![[0.0f32; 3]; self.positions.len()];
        for face in 0..self.indices.len() {
            let [a, b, c] = self.triangle(face);
            // The cross product's length is twice the triangle's area, so
            // accumulating it unnormalized is the weighting.
            let normal = cross(sub(b, a), sub(c, a));
            for corner in self.indices[face] {
                for axis in 0..3 {
                    normals[corner as usize][axis] += normal[axis];
                }
            }
        }
        for normal in &mut normals {
            *normal = normalize(*normal);
        }
        self.normals = normals;
    }

    /// Total surface area.
    pub fn area(&self) -> f32 {
        (0..self.indices.len())
            .map(|face| {
                let [a, b, c] = self.triangle(face);
                length(cross(sub(b, a), sub(c, a))) / 2.0
            })
            .sum()
    }

    /// Reads a Wavefront OBJ file.
    ///
    /// Positions, normals, texture coordinates and faces. Faces with more than
    /// three corners are fanned into triangles, which is right for the convex
    /// quads an exporter emits. Materials, groups and smoothing are skipped:
    /// the geometry is what a distance field is built from.
    pub fn read_obj<P: AsRef<Path>>(path: P) -> Result<Self, NetworkError> {
        let text = std::fs::read_to_string(path)?;
        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut uvs = Vec::new();
        let mut colours: Vec<[f32; 3]> = Vec::new();
        // OBJ indexes position, normal and texture coordinate separately, and a
        // mesh indexes them together, so each distinct triple becomes a vertex.
        let mut vertices: HashMap<(i64, i64, i64), u32> = HashMap::new();
        let mut mesh = Mesh::default();

        for (number, line) in text.lines().enumerate() {
            let mut fields = line.split_whitespace();
            let complain =
                |what: &str| NetworkError::InvalidDataset(format!("line {}: {what}", number + 1));
            match fields.next() {
                Some("v") => {
                    positions.push(
                        triple(&mut fields)
                            .ok_or_else(|| complain("a vertex needs three coordinates"))?,
                    );
                    // The extended form writes the vertex colour in the three
                    // fields after the position. Exporters that do not write it
                    // leave the line at three fields, and the mismatch check
                    // below drops a half-coloured mesh.
                    if let Some(colour) = triple(&mut fields) {
                        colours.push(colour);
                    }
                }
                Some("vn") => normals.push(
                    triple(&mut fields)
                        .ok_or_else(|| complain("a normal needs three components"))?,
                ),
                Some("vt") => {
                    let u = fields.next().and_then(|f| f.parse().ok());
                    let v: f32 = fields.next().and_then(|f| f.parse().ok()).unwrap_or(0.0);
                    // An .obj counts v up from the bottom of the image, glTF
                    // counts it down from the top. Whatever reads `uvs` back
                    // cannot tell which file they came from, so both readers
                    // hand it the same convention: glTF's.
                    uvs.push([
                        u.ok_or_else(|| complain("a texture coordinate needs a u"))?,
                        1.0 - v,
                    ]);
                }
                Some("f") => {
                    let corners = fields
                        .map(|field| {
                            let mut parts = field.split('/');
                            let position = parts.next().and_then(|p| p.parse::<i64>().ok());
                            let uv = parts.next().and_then(|p| p.parse::<i64>().ok());
                            let normal = parts.next().and_then(|p| p.parse::<i64>().ok());
                            position
                                .map(|position| (position, uv.unwrap_or(0), normal.unwrap_or(0)))
                        })
                        .collect::<Option<Vec<_>>>()
                        .ok_or_else(|| complain("a face corner needs a vertex index"))?;
                    if corners.len() < 3 {
                        return Err(complain("a face needs three corners"));
                    }

                    let mut resolved = Vec::with_capacity(corners.len());
                    for key in corners {
                        let index = match vertices.get(&key) {
                            Some(index) => *index,
                            None => {
                                let index = mesh.positions.len() as u32;
                                mesh.positions.push(*pick(&positions, key.0).ok_or_else(|| {
                                    complain("a face names a vertex that is not there")
                                })?);
                                if let Some(colour) = pick(&colours, key.0) {
                                    mesh.colors.push(*colour);
                                }
                                if let Some(uv) = pick(&uvs, key.1) {
                                    mesh.uvs.push(*uv);
                                }
                                if let Some(normal) = pick(&normals, key.2) {
                                    mesh.normals.push(*normal);
                                }
                                vertices.insert(key, index);
                                index
                            }
                        };
                        resolved.push(index);
                    }
                    // A fan, which is correct for the convex polygons an
                    // exporter writes and is what every other reader does.
                    for corner in 1..resolved.len() - 1 {
                        mesh.indices
                            .push([resolved[0], resolved[corner], resolved[corner + 1]]);
                    }
                }
                _ => {}
            }
        }

        if mesh.indices.is_empty() {
            return Err(NetworkError::InvalidDataset("the file has no faces".into()));
        }
        // Half a vertex's attributes is worse than none: a consumer cannot tell
        // which vertices have them.
        if mesh.uvs.len() != mesh.positions.len() {
            mesh.uvs.clear();
        }
        if mesh.normals.len() != mesh.positions.len() {
            mesh.normals.clear();
        }
        if mesh.colors.len() != mesh.positions.len() {
            mesh.colors.clear();
        }
        Ok(mesh)
    }

    /// Writes a Wavefront OBJ file: positions, normals if there are any, and
    /// faces.
    pub fn write_obj<P: AsRef<Path>>(&self, path: P) -> Result<(), NetworkError> {
        use std::io::Write;
        let mut file = std::io::BufWriter::new(std::fs::File::create(path)?);
        for [x, y, z] in &self.positions {
            writeln!(file, "v {x} {y} {z}")?;
        }
        for [x, y, z] in &self.normals {
            writeln!(file, "vn {x} {y} {z}")?;
        }
        let normals = !self.normals.is_empty();
        for face in &self.indices {
            let [a, b, c] = face.map(|index| index + 1);
            match normals {
                true => writeln!(file, "f {a}//{a} {b}//{b} {c}//{c}")?,
                false => writeln!(file, "f {a} {b} {c}")?,
            }
        }
        Ok(())
    }
}

/// OBJ indices count from one, and a negative index counts back from the end.
fn pick<T>(values: &[T], index: i64) -> Option<&T> {
    match index {
        0 => None,
        index if index > 0 => values.get(index as usize - 1),
        index => values
            .len()
            .checked_sub(index.unsigned_abs() as usize)
            .map(|at| &values[at]),
    }
}

fn triple<'a>(fields: &mut impl Iterator<Item = &'a str>) -> Option<[f32; 3]> {
    let mut values = [0.0; 3];
    for value in &mut values {
        *value = fields.next()?.parse().ok()?;
    }
    Some(values)
}

// -- Vector helpers. Three components, no dependency, no trait. ---------------

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    std::array::from_fn(|axis| a[axis] - b[axis])
}

fn add3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    std::array::from_fn(|axis| a[axis] + b[axis])
}

fn scale(a: [f32; 3], factor: f32) -> [f32; 3] {
    std::array::from_fn(|axis| a[axis] * factor)
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn length(a: [f32; 3]) -> f32 {
    dot(a, a).sqrt()
}

fn normalize(a: [f32; 3]) -> [f32; 3] {
    match length(a) {
        0.0 => [0.0; 3],
        length => scale(a, 1.0 / length),
    }
}

// -- The tree ----------------------------------------------------------------

/// One node of a [`Bvh`]. Interior nodes keep the left child next to them and
/// the right child's index; leaves keep a run of faces.
#[derive(Clone, Debug)]
struct Node {
    low: [f32; 3],
    high: [f32; 3],
    /// First face in `order`, for a leaf.
    start: u32,
    /// Faces in the leaf, or zero for an interior node.
    count: u32,
    /// Right child, for an interior node.
    right: u32,
    /// Area-weighted normal summed over the node's faces, and the area-weighted
    /// mean of their centroids. Together they are the dipole the winding number
    /// approximates a distant node with.
    moment: [f32; 3],
    centroid: [f32; 3],
    /// Distance from `centroid` to the furthest point of the node's faces,
    /// which is what decides whether the approximation is allowed.
    radius: f32,
}

/// A bounding volume hierarchy over a mesh's triangles.
///
/// Built once per mesh and asked two questions many times over, so it borrows
/// the mesh rather than copying it.
pub struct Bvh<'a> {
    mesh: &'a Mesh,
    nodes: Vec<Node>,
    /// Face indices, permuted so a leaf is a contiguous run.
    order: Vec<u32>,
}

/// Faces in a leaf. Four is the usual answer: fewer makes the tree deeper than
/// the traversal saves, more makes the leaf scan dominate.
const LEAF_SIZE: usize = 4;

/// How far a node has to be, in multiples of its own radius, before its faces
/// are replaced by one dipole. Jacobson's paper uses 2; larger is more accurate
/// and slower.
const WINDING_BETA: f32 = 2.0;

impl<'a> Bvh<'a> {
    /// Builds the tree. `O(n log n)`, single-threaded, and fast enough that it
    /// has never been the thing worth parallelizing.
    pub fn build(mesh: &'a Mesh) -> Self {
        let centroids: Vec<[f32; 3]> = (0..mesh.indices.len())
            .map(|face| {
                let [a, b, c] = mesh.triangle(face);
                scale(add3(add3(a, b), c), 1.0 / 3.0)
            })
            .collect();

        let mut tree = Self {
            mesh,
            nodes: Vec::new(),
            order: (0..mesh.indices.len() as u32).collect(),
        };
        if !tree.order.is_empty() {
            tree.split(0, tree.mesh.indices.len(), &centroids);
        }
        tree
    }

    /// Builds the node covering `order[start..end]` and returns its index.
    fn split(&mut self, start: usize, end: usize, centroids: &[[f32; 3]]) -> u32 {
        let index = self.nodes.len() as u32;
        let (low, high) = self.face_bounds(start, end);
        self.nodes.push(Node {
            low,
            high,
            start: start as u32,
            count: (end - start) as u32,
            right: 0,
            moment: [0.0; 3],
            centroid: [0.0; 3],
            radius: 0.0,
        });
        self.summarize(index as usize, start, end);

        if end - start > LEAF_SIZE {
            // Split at the median along the widest axis of the centroids. A
            // surface-area heuristic builds a better tree; the median one is
            // twenty lines and has never been what a query waits on.
            let axis = (0..3)
                .max_by(|a, b| {
                    (high[*a] - low[*a])
                        .partial_cmp(&(high[*b] - low[*b]))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(0);
            let middle = start + (end - start) / 2;
            self.order[start..end].select_nth_unstable_by(middle - start, |a, b| {
                centroids[*a as usize][axis]
                    .partial_cmp(&centroids[*b as usize][axis])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            self.nodes[index as usize].count = 0;
            self.split(start, middle, centroids);
            let right = self.split(middle, end, centroids);
            self.nodes[index as usize].right = right;
        }
        index
    }

    fn face_bounds(&self, start: usize, end: usize) -> ([f32; 3], [f32; 3]) {
        let mut low = [f32::INFINITY; 3];
        let mut high = [f32::NEG_INFINITY; 3];
        for face in &self.order[start..end] {
            for corner in self.mesh.triangle(*face as usize) {
                for axis in 0..3 {
                    low[axis] = low[axis].min(corner[axis]);
                    high[axis] = high[axis].max(corner[axis]);
                }
            }
        }
        (low, high)
    }

    /// Fills in the dipole the winding number uses for a distant node.
    fn summarize(&mut self, index: usize, start: usize, end: usize) {
        let mut moment = [0.0f32; 3];
        let mut centroid = [0.0f32; 3];
        let mut area = 0.0f32;
        for face in &self.order[start..end] {
            let [a, b, c] = self.mesh.triangle(*face as usize);
            // Twice the area times the unit normal, which is the weight the
            // solid angle of a distant triangle is proportional to.
            let doubled = cross(sub(b, a), sub(c, a));
            let face_area = length(doubled) / 2.0;
            moment = add3(moment, scale(doubled, 0.5));
            centroid = add3(centroid, scale(add3(add3(a, b), c), face_area / 3.0));
            area += face_area;
        }
        if area > 0.0 {
            centroid = scale(centroid, 1.0 / area);
        }
        let mut radius = 0.0f32;
        for face in &self.order[start..end] {
            for corner in self.mesh.triangle(*face as usize) {
                radius = radius.max(length(sub(corner, centroid)));
            }
        }
        let node = &mut self.nodes[index];
        node.moment = moment;
        node.centroid = centroid;
        node.radius = radius;
    }

    /// The point on the mesh closest to `query`, and its distance.
    ///
    /// Returns `None` for a mesh with no faces.
    pub fn closest_point(&self, query: [f32; 3]) -> Option<([f32; 3], f32)> {
        if self.nodes.is_empty() {
            return None;
        }
        let mut best = ([0.0; 3], f32::INFINITY);
        self.descend(0, query, &mut best);
        Some(best)
    }

    fn descend(&self, index: u32, query: [f32; 3], best: &mut ([f32; 3], f32)) {
        let node = &self.nodes[index as usize];
        if box_distance(node.low, node.high, query) >= best.1 {
            return;
        }
        if node.count > 0 {
            let range = node.start as usize..(node.start + node.count) as usize;
            for face in &self.order[range] {
                let point = closest_on_triangle(query, self.mesh.triangle(*face as usize));
                let distance = length(sub(point, query));
                if distance < best.1 {
                    *best = (point, distance);
                }
            }
            return;
        }
        // Nearer child first, so the further one is more likely to be culled.
        let (left, right) = (index + 1, node.right);
        let order = match box_distance(
            self.nodes[left as usize].low,
            self.nodes[left as usize].high,
            query,
        ) <= box_distance(
            self.nodes[right as usize].low,
            self.nodes[right as usize].high,
            query,
        ) {
            true => [left, right],
            false => [right, left],
        };
        for child in order {
            self.descend(child, query, best);
        }
    }

    /// The generalized winding number at `query`: about 1 inside a closed mesh,
    /// about 0 outside, and something in between near a hole.
    ///
    /// Exact for nearby triangles and a first-order dipole for distant nodes,
    /// which is the Barnes-Hut arrangement from Jacobson, Kavan and Sorkine-
    /// Hornung's *Robust Inside-Outside Segmentation using Generalized Winding
    /// Numbers*. Without the approximation every query would touch every
    /// triangle, because a solid angle has no cutoff.
    ///
    /// NOTE: first-order only. Adding the second-order term would let
    /// [`WINDING_BETA`] drop and the traversal stop sooner; the first order is
    /// what the paper shows is enough to classify points, which is all this is
    /// used for.
    pub fn winding_number(&self, query: [f32; 3]) -> f32 {
        match self.nodes.is_empty() {
            true => 0.0,
            false => self.wind(0, query) / (4.0 * std::f32::consts::PI),
        }
    }

    fn wind(&self, index: u32, query: [f32; 3]) -> f32 {
        let node = &self.nodes[index as usize];
        let offset = sub(node.centroid, query);
        let distance = length(offset);
        if distance > WINDING_BETA * node.radius && distance > 0.0 {
            return dot(node.moment, offset) / (distance * distance * distance);
        }
        if node.count > 0 {
            let range = node.start as usize..(node.start + node.count) as usize;
            return self.order[range]
                .iter()
                .map(|face| solid_angle(query, self.mesh.triangle(*face as usize)))
                .sum();
        }
        self.wind(index + 1, query) + self.wind(node.right, query)
    }

    /// Distance to the surface, negative inside it.
    ///
    /// One thread per query point: they are independent, and a field of two
    /// million of them is the usual ask.
    pub fn signed_distance(&self, queries: &[[f32; 3]]) -> Vec<f32> {
        queries
            .par_iter()
            .map(|query| {
                let distance = self.closest_point(*query).map_or(0.0, |(_, d)| d);
                // A half turn is the boundary: a closed mesh reads 1 inside and
                // 0 outside, and a mesh with a hole reads somewhere between.
                match self.winding_number(*query) > 0.5 {
                    true => -distance,
                    false => distance,
                }
            })
            .collect()
    }
}

/// Distance from a point to a box, zero inside it.
fn box_distance(low: [f32; 3], high: [f32; 3], query: [f32; 3]) -> f32 {
    let outside: [f32; 3] = std::array::from_fn(|axis| {
        (low[axis] - query[axis])
            .max(query[axis] - high[axis])
            .max(0.0)
    });
    length(outside)
}

/// The point of a triangle closest to `query`, by the region test from
/// Ericson's *Real-Time Collision Detection*: check the three vertices, then
/// the three edges, then the interior.
fn closest_on_triangle(query: [f32; 3], [a, b, c]: [[f32; 3]; 3]) -> [f32; 3] {
    let (ab, ac, aq) = (sub(b, a), sub(c, a), sub(query, a));
    let (d1, d2) = (dot(ab, aq), dot(ac, aq));
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }
    let bq = sub(query, b);
    let (d3, d4) = (dot(ab, bq), dot(ac, bq));
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return add3(a, scale(ab, d1 / (d1 - d3)));
    }
    let cq = sub(query, c);
    let (d5, d6) = (dot(ab, cq), dot(ac, cq));
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return add3(a, scale(ac, d2 / (d2 - d6)));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        return add3(b, scale(sub(c, b), (d4 - d3) / ((d4 - d3) + (d5 - d6))));
    }
    let denominator = va + vb + vc;
    if denominator == 0.0 {
        // A degenerate triangle, which a real dataset does contain.
        return a;
    }
    let (v, w) = (vb / denominator, vc / denominator);
    add3(add3(a, scale(ab, v)), scale(ac, w))
}

/// The signed solid angle a triangle subtends at `query`, by Van Oosterom and
/// Strackee's formula. Its sum over a closed mesh is `4π` inside and 0 outside.
fn solid_angle(query: [f32; 3], [a, b, c]: [[f32; 3]; 3]) -> f32 {
    let (a, b, c) = (sub(a, query), sub(b, query), sub(c, query));
    let (la, lb, lc) = (length(a), length(b), length(c));
    let denominator = la * lb * lc + dot(a, b) * lc + dot(a, c) * lb + dot(b, c) * la;
    match denominator == 0.0 && dot(a, cross(b, c)) == 0.0 {
        // The query is on the triangle's plane inside it, where the angle is
        // undefined and `atan2(0, 0)` would answer zero.
        true => 0.0,
        false => 2.0 * dot(a, cross(b, c)).atan2(denominator),
    }
}

// -- Sampling ----------------------------------------------------------------

/// How the query points a shape model trains on are spread around.
///
/// The ratio between the two kinds is the single biggest lever on how a learned
/// distance field turns out. All-uniform and the model never sees the surface
/// closely enough to place it; all-near-surface and it has no idea which side
/// of the shape the empty space is on, so it hallucinates a second surface out
/// in the open. Start at the default and move it, rather than treating it as
/// settled.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuerySampling {
    /// Points to draw in total.
    pub count: usize,
    /// The fraction drawn near the surface rather than uniformly in the box.
    pub near_surface: f32,
    /// Standard deviation of the offset applied to a near-surface point, in the
    /// units the mesh is normalized to.
    pub jitter: f32,
    /// Half-width of the uniform box, `1.0` for a mesh through
    /// [`Mesh::normalize`]. A little over 1 leaves the model some empty space
    /// outside the shape to learn from.
    pub extent: f32,
}

impl Default for QuerySampling {
    fn default() -> Self {
        Self {
            count: 200_000,
            near_surface: 0.7,
            jitter: 0.02,
            extent: 1.1,
        }
    }
}

/// Points, normals or colours: one `[f32; 3]` per surface sample.
pub type Cloud = Vec<[f32; 3]>;

impl Bvh<'_> {
    /// Points on the surface, area-weighted, with the face normal at each.
    ///
    /// Area weighting is what makes this uniform over the surface instead of
    /// uniform over the faces: without it a mesh with one huge floor triangle
    /// and a thousand small ones on a lamp would put almost every sample on the
    /// lamp.
    pub fn sample_surface<R: Rng>(
        &self,
        count: usize,
        rng: &mut R,
    ) -> (Vec<[f32; 3]>, Vec<[f32; 3]>) {
        let (points, normals, _) = self.sample_surface_colored(count, rng);
        (points, normals)
    }

    /// The same sampling, with the vertex colour interpolated at each point.
    ///
    /// The colours come back empty when the mesh carries none, so a caller that
    /// wants them still has to check.
    pub fn sample_surface_colored<R: Rng>(
        &self,
        count: usize,
        rng: &mut R,
    ) -> (Cloud, Cloud, Cloud) {
        let faces = self.mesh.indices.len();
        if faces == 0 || count == 0 {
            return (Vec::new(), Vec::new(), Vec::new());
        }
        let colored = self.mesh.colors.len() == self.mesh.positions.len();
        // Cumulative area, so a uniform draw lands in a face with probability
        // proportional to its area.
        let mut cumulative = Vec::with_capacity(faces);
        let mut total = 0.0f32;
        for face in 0..faces {
            let [a, b, c] = self.mesh.triangle(face);
            total += length(cross(sub(b, a), sub(c, a))) / 2.0;
            cumulative.push(total);
        }

        let mut points = Vec::with_capacity(count);
        let mut normals = Vec::with_capacity(count);
        let mut colors = Vec::with_capacity(if colored { count } else { 0 });
        for _ in 0..count {
            let target = rng.gen_range(0.0..total.max(f32::MIN_POSITIVE));
            let face = cumulative
                .partition_point(|value| *value < target)
                .min(faces - 1);
            let [a, b, c] = self.mesh.triangle(face);
            // The square root keeps the pair uniform over the triangle rather
            // than bunched towards `a`.
            let (u, v) = (rng.r#gen::<f32>().sqrt(), rng.r#gen::<f32>());
            let point = add3(
                add3(a, scale(sub(b, a), u * (1.0 - v))),
                scale(sub(c, a), u * v),
            );
            points.push(point);
            normals.push(normalize(cross(sub(b, a), sub(c, a))));
            if colored {
                // The same barycentric weights the point was built from.
                let corners = self.mesh.indices[face];
                let weights = [1.0 - u, u * (1.0 - v), u * v];
                colors.push(std::array::from_fn(|channel| {
                    (0..3)
                        .map(|corner| {
                            weights[corner] * self.mesh.colors[corners[corner] as usize][channel]
                        })
                        .sum()
                }));
            }
        }
        (points, normals, colors)
    }

    /// The query points a distance field is fitted on: some uniform in the box,
    /// the rest hugging the surface.
    pub fn sample_queries<R: Rng>(&self, config: &QuerySampling, rng: &mut R) -> Vec<[f32; 3]> {
        let near = (config.count as f32 * config.near_surface.clamp(0.0, 1.0)) as usize;
        let (surface, _) = self.sample_surface(near, rng);
        let mut queries: Vec<[f32; 3]> = surface
            .into_iter()
            .map(|point| std::array::from_fn(|axis| point[axis] + gaussian(rng) * config.jitter))
            .collect();
        for _ in near..config.count {
            queries.push(std::array::from_fn(|_| {
                rng.gen_range(-config.extent..config.extent)
            }));
        }
        queries
    }
}

/// One standard normal, by Box-Muller. `rand_distr` would bring a dependency
/// for three lines.
fn gaussian<R: Rng>(rng: &mut R) -> f32 {
    let uniform: f32 = rng.gen_range(f32::MIN_POSITIVE..1.0);
    let angle: f32 = rng.gen_range(0.0..std::f32::consts::TAU);
    (-2.0 * uniform.ln()).sqrt() * angle.cos()
}

// -- Back to a mesh ----------------------------------------------------------

/// Turns a distance field into a mesh: the surface where it crosses `iso`.
///
/// `field` is handed a `[n, 3]` matrix of query points and returns one value
/// per row, which is the shape a chunked decoder already speaks. It is called
/// once per slab of the grid rather than once for the whole thing, so a
/// `128³` grid never has all 2.1 million points resident at once.
///
/// NOTE: tetrahedra, not cubes. Each cell is cut into six tetrahedra, which
/// have sixteen cases between them instead of a cube's 256, so the lookup table
/// is four lines instead of three hundred and there is no ambiguous case to get
/// wrong and leave a hole in. The cost is about twice the triangles for the
/// same surface. Swap in the full marching-cubes tables if the triangle count
/// ever matters more than the code does.
pub fn marching_tetrahedra(
    field: impl Fn(&Matrix) -> Result<Vec<f32>, NetworkError> + Sync,
    resolution: usize,
    bounds: ([f32; 3], [f32; 3]),
    iso: f32,
) -> Result<Mesh, NetworkError> {
    let grid = Grid::new(resolution, bounds)?;
    let mut mesh = Mesh::default();
    let mut welded = HashMap::new();
    let mut previous: Option<Vec<f32>> = None;

    for z in 0..resolution {
        let mut points = Matrix::new(grid.plane, 3);
        for y in 0..resolution {
            for x in 0..resolution {
                points
                    .row_mut(y * resolution + x)
                    .copy_from_slice(&grid.at(x, y, z));
            }
        }
        let values = grid.evaluate(&field, &points)?;
        if let Some(lower) = previous {
            grid.emit_cells(&mut mesh, &mut welded, iso, z, &lower, &values);
        }
        previous = Some(values);
    }

    mesh.recompute_normals();
    Ok(mesh)
}

/// [`marching_tetrahedra`] with the empty space skipped, for a field that is
/// expensive to evaluate.
///
/// The marcher itself costs almost nothing — two planes of values live at a
/// time, so a `256³` grid holds 27 MB whatever the field is. What a high
/// resolution costs is *calls*: `256³` is 16.7 million query points, and a
/// decoder that answers a point in a few microseconds spends half an hour on
/// them. Nearly all of those points are deep inside or far outside the shape,
/// where there is nothing to find.
///
/// So this runs the grid twice. A coarse pass every `coarse` samples finds the
/// cells the surface crosses; the fine pass evaluates only the points inside
/// those cells and their immediate neighbours, and marches the fine cells whose
/// eight corners it has. A sphere at `256³` with a `coarse` of 4 costs 1.6
/// million query points instead of 16.7 million, and gives back the same mesh
/// vertex for vertex.
///
/// The caveat is the one every hierarchical extractor has: a feature that fits
/// entirely inside a coarse cell without crossing any of its corners is not
/// seen, and the fine pass never looks there. `coarse` is the thickness of the
/// thinnest wall this can find, in samples: 4 on a `256³` grid means a wall
/// thinner than about 1/64 of the box may be missed. It is also what decides
/// the saving, and the two pull the same way — a larger stride skips more
/// coarse cells but each one drags in its 26 neighbours, so past about 4 the
/// dilated cells start to overlap and the saving falls off. Use
/// [`marching_tetrahedra`] when a thin wall matters more than the time.
pub fn marching_tetrahedra_sparse(
    field: impl Fn(&Matrix) -> Result<Vec<f32>, NetworkError> + Sync,
    resolution: usize,
    bounds: ([f32; 3], [f32; 3]),
    iso: f32,
    coarse: usize,
) -> Result<Mesh, NetworkError> {
    if coarse < 2 {
        return Err(NetworkError::InvalidConfig(
            "a coarse stride of one samples the whole grid twice; call \
             marching_tetrahedra instead"
                .into(),
        ));
    }
    let grid = Grid::new(resolution, bounds)?;
    let occupied = grid.coarse_pass(&field, iso, coarse)?;

    let mut mesh = Mesh::default();
    let mut welded = HashMap::new();
    let mut previous: Option<Vec<f32>> = None;

    for z in 0..resolution {
        // A plane of values, `NaN` where the coarse pass said there is nothing
        // to look at. A cell with a `NaN` corner is skipped rather than
        // guessed at.
        let mut values = vec![f32::NAN; grid.plane];
        let mut wanted: Vec<usize> = Vec::new();
        for y in 0..resolution {
            for x in 0..resolution {
                if occupied.holds(x, y, z) {
                    wanted.push(y * resolution + x);
                }
            }
        }
        if !wanted.is_empty() {
            let mut points = Matrix::new(wanted.len(), 3);
            for (row, &index) in wanted.iter().enumerate() {
                let (x, y) = (index % resolution, index / resolution);
                points.row_mut(row).copy_from_slice(&grid.at(x, y, z));
            }
            let evaluated = grid.evaluate(&field, &points)?;
            for (&index, value) in wanted.iter().zip(evaluated) {
                values[index] = value;
            }
        }
        if let Some(lower) = previous {
            grid.emit_cells(&mut mesh, &mut welded, iso, z, &lower, &values);
        }
        previous = Some(values);
    }

    mesh.recompute_normals();
    Ok(mesh)
}

/// The sample grid both marchers walk: where a sample sits, and how a cell
/// between two planes becomes triangles.
struct Grid {
    resolution: usize,
    plane: usize,
    low: [f32; 3],
    step: [f32; 3],
}

impl Grid {
    fn new(resolution: usize, bounds: ([f32; 3], [f32; 3])) -> Result<Self, NetworkError> {
        if resolution < 2 {
            return Err(NetworkError::InvalidConfig(
                "a grid needs at least two samples along each axis".into(),
            ));
        }
        let (low, high) = bounds;
        Ok(Self {
            resolution,
            plane: resolution * resolution,
            low,
            step: std::array::from_fn(|axis| (high[axis] - low[axis]) / (resolution - 1) as f32),
        })
    }

    fn at(&self, x: usize, y: usize, z: usize) -> [f32; 3] {
        [
            self.low[0] + x as f32 * self.step[0],
            self.low[1] + y as f32 * self.step[1],
            self.low[2] + z as f32 * self.step[2],
        ]
    }

    /// One call into the field, with the row count checked: a field that
    /// answers a different number of points than it was asked would otherwise
    /// silently shift every value onto the wrong grid corner.
    fn evaluate(
        &self,
        field: &(impl Fn(&Matrix) -> Result<Vec<f32>, NetworkError> + Sync),
        points: &Matrix,
    ) -> Result<Vec<f32>, NetworkError> {
        let values = field(points)?;
        if values.len() != points.rows {
            return Err(NetworkError::InvalidTarget {
                expected: points.rows,
                actual: values.len(),
            });
        }
        Ok(values)
    }

    /// Marches every cell between plane `z - 1` and plane `z`, skipping the
    /// ones a sparse pass left unevaluated.
    fn emit_cells(
        &self,
        mesh: &mut Mesh,
        welded: &mut HashMap<(u32, u32), u32>,
        iso: f32,
        z: usize,
        lower: &[f32],
        upper: &[f32],
    ) {
        let resolution = self.resolution;
        for y in 0..resolution - 1 {
            for x in 0..resolution - 1 {
                // The eight corners, in the order CUBE_TETRAHEDRA indexes.
                let corners = [
                    (x, y, z - 1),
                    (x + 1, y, z - 1),
                    (x + 1, y + 1, z - 1),
                    (x, y + 1, z - 1),
                    (x, y, z),
                    (x + 1, y, z),
                    (x + 1, y + 1, z),
                    (x, y + 1, z),
                ];
                let value = |corner: usize| {
                    let (cx, cy, _) = corners[corner];
                    let index = cy * resolution + cx;
                    match corner < 4 {
                        true => lower[index],
                        false => upper[index],
                    }
                };
                if (0..8).any(|corner| value(corner).is_nan()) {
                    continue;
                }

                for tetrahedron in CUBE_TETRAHEDRA {
                    emit_tetrahedron(
                        mesh,
                        welded,
                        iso,
                        tetrahedron.map(|corner| {
                            let (cx, cy, cz) = corners[corner];
                            let identity = (cz * self.plane + cy * resolution + cx) as u32;
                            (identity, self.at(cx, cy, cz), value(corner))
                        }),
                    );
                }
            }
        }
    }

    /// Walks the grid every `coarse` samples and reports which coarse cells the
    /// surface passes through, each one grown by a cell in every direction so
    /// that a fine cell inside a marked one always has all eight of its corners
    /// evaluated.
    fn coarse_pass(
        &self,
        field: &(impl Fn(&Matrix) -> Result<Vec<f32>, NetworkError> + Sync),
        iso: f32,
        coarse: usize,
    ) -> Result<Occupancy, NetworkError> {
        let cells = (self.resolution - 1).div_ceil(coarse);
        let samples = cells + 1;
        let sample_at = |index: usize| (index * coarse).min(self.resolution - 1);

        let mut straddles = vec![false; cells * cells * cells];
        let mut previous: Option<Vec<f32>> = None;
        for k in 0..samples {
            let mut points = Matrix::new(samples * samples, 3);
            for j in 0..samples {
                for i in 0..samples {
                    points.row_mut(j * samples + i).copy_from_slice(&self.at(
                        sample_at(i),
                        sample_at(j),
                        sample_at(k),
                    ));
                }
            }
            let values = self.evaluate(field, &points)?;
            if let Some(lower) = previous {
                for j in 0..cells {
                    for i in 0..cells {
                        let corners = [
                            lower[j * samples + i],
                            lower[j * samples + i + 1],
                            lower[(j + 1) * samples + i],
                            lower[(j + 1) * samples + i + 1],
                            values[j * samples + i],
                            values[j * samples + i + 1],
                            values[(j + 1) * samples + i],
                            values[(j + 1) * samples + i + 1],
                        ];
                        let inside = corners.iter().any(|&value| value < iso);
                        let outside = corners.iter().any(|&value| value >= iso);
                        if inside && outside {
                            straddles[(k - 1) * cells * cells + j * cells + i] = true;
                        }
                    }
                }
            }
            previous = Some(values);
        }

        Ok(Occupancy::dilated(straddles, cells, coarse))
    }
}

/// Which coarse cells the fine pass has to look inside.
struct Occupancy {
    marked: Vec<bool>,
    cells: usize,
    coarse: usize,
}

impl Occupancy {
    /// Grows every marked cell into its 26 neighbours. A fine cell sitting at
    /// the edge of a marked coarse cell reaches one sample into the next one,
    /// so without this its far corners would be `NaN` and the cell would be
    /// dropped — a seam along every coarse boundary.
    fn dilated(straddles: Vec<bool>, cells: usize, coarse: usize) -> Self {
        let mut marked = vec![false; straddles.len()];
        for k in 0..cells {
            for j in 0..cells {
                for i in 0..cells {
                    marked[k * cells * cells + j * cells + i] = (-1i64..=1).any(|dk| {
                        (-1i64..=1).any(|dj| {
                            (-1i64..=1).any(|di| {
                                let (ni, nj, nk) = (i as i64 + di, j as i64 + dj, k as i64 + dk);
                                let inside = |value: i64| (0..cells as i64).contains(&value);
                                inside(ni)
                                    && inside(nj)
                                    && inside(nk)
                                    && straddles[nk as usize * cells * cells
                                        + nj as usize * cells
                                        + ni as usize]
                            })
                        })
                    });
                }
            }
        }
        Self {
            marked,
            cells,
            coarse,
        }
    }

    /// Whether the fine sample at `(x, y, z)` is worth evaluating.
    fn holds(&self, x: usize, y: usize, z: usize) -> bool {
        let cell = |index: usize| (index / self.coarse).min(self.cells - 1);
        self.marked[cell(z) * self.cells * self.cells + cell(y) * self.cells + cell(x)]
    }
}

#[cfg(test)]
mod sparse_tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A sphere of radius 0.6, and a count of how many points it was asked for.
    fn sphere(
        radius: f32,
        asked: &AtomicUsize,
    ) -> impl Fn(&Matrix) -> Result<Vec<f32>, NetworkError> + Sync {
        move |points: &Matrix| {
            asked.fetch_add(points.rows, Ordering::Relaxed);
            Ok((0..points.rows)
                .map(|row| {
                    let point = points.row(row);
                    (point[0] * point[0] + point[1] * point[1] + point[2] * point[2]).sqrt()
                        - radius
                })
                .collect())
        }
    }

    #[test]
    fn skipping_the_empty_space_gives_the_same_mesh_as_marching_all_of_it() {
        let bounds = ([-1.0; 3], [1.0; 3]);
        let dense_calls = AtomicUsize::new(0);
        let dense = marching_tetrahedra(sphere(0.6, &dense_calls), 64, bounds, 0.0).unwrap();

        let sparse_calls = AtomicUsize::new(0);
        let sparse =
            marching_tetrahedra_sparse(sphere(0.6, &sparse_calls), 64, bounds, 0.0, 4).unwrap();

        assert!(!dense.indices.is_empty(), "the sphere crosses the grid");
        assert_eq!(sparse.positions, dense.positions);
        assert_eq!(sparse.indices, dense.indices);
        assert_eq!(sparse.normals, dense.normals);
        assert!(
            sparse_calls.load(Ordering::Relaxed) * 2 < dense_calls.load(Ordering::Relaxed),
            "the sparse pass asked for {} points against {}",
            sparse_calls.load(Ordering::Relaxed),
            dense_calls.load(Ordering::Relaxed)
        );
    }

    #[test]
    fn a_shell_thinner_than_a_coarse_cell_needs_the_dense_marcher() {
        // The documented caveat, pinned: a shell 1/64 of the box thick sits
        // inside a coarse cell of 8 samples and the coarse pass steps over it.
        let bounds = ([-1.0; 3], [1.0; 3]);
        let shell = |points: &Matrix| -> Result<Vec<f32>, NetworkError> {
            Ok((0..points.rows)
                .map(|row| {
                    let point = points.row(row);
                    let radius =
                        (point[0] * point[0] + point[1] * point[1] + point[2] * point[2]).sqrt();
                    (radius - 0.6).abs() - 0.015
                })
                .collect())
        };
        let dense = marching_tetrahedra(shell, 64, bounds, 0.0).unwrap();
        let sparse = marching_tetrahedra_sparse(shell, 64, bounds, 0.0, 8).unwrap();
        assert!(!dense.indices.is_empty());
        assert!(sparse.indices.len() < dense.indices.len());
    }

    #[test]
    fn a_coarse_stride_of_one_is_refused() {
        let asked = AtomicUsize::new(0);
        let error =
            marching_tetrahedra_sparse(sphere(0.6, &asked), 16, ([-1.0; 3], [1.0; 3]), 0.0, 1);
        assert!(matches!(error, Err(NetworkError::InvalidConfig(_))));
    }

    #[test]
    fn a_grid_of_one_sample_is_refused_by_both_marchers() {
        let asked = AtomicUsize::new(0);
        let bounds = ([-1.0; 3], [1.0; 3]);
        assert!(marching_tetrahedra(sphere(0.6, &asked), 1, bounds, 0.0).is_err());
        assert!(marching_tetrahedra_sparse(sphere(0.6, &asked), 1, bounds, 0.0, 4).is_err());
    }
}

/// The six tetrahedra a cube is cut into, as corner indices. Every one of them
/// shares the cube's `0`-`6` diagonal, which is what makes the cut consistent
/// between neighbouring cells and the result watertight.
const CUBE_TETRAHEDRA: [[usize; 4]; 6] = [
    [0, 5, 1, 6],
    [0, 1, 2, 6],
    [0, 2, 3, 6],
    [0, 3, 7, 6],
    [0, 7, 4, 6],
    [0, 4, 5, 6],
];

/// The six edges of a tetrahedron, as pairs of its corners.
const TETRAHEDRON_EDGES: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];

/// Which edges the surface crosses, and how they join up, for each of the
/// sixteen ways four corners can be inside or outside.
///
/// One triangle when a single corner is on its own, two when the corners split
/// two and two. `usize::MAX` ends the list.
const TETRAHEDRON_TABLE: [[usize; 6]; 16] = [
    [usize::MAX; 6],
    [0, 1, 2, usize::MAX, usize::MAX, usize::MAX],
    [0, 3, 4, usize::MAX, usize::MAX, usize::MAX],
    [1, 2, 4, 1, 4, 3],
    [1, 3, 5, usize::MAX, usize::MAX, usize::MAX],
    [0, 2, 5, 0, 5, 3],
    [0, 1, 5, 0, 5, 4],
    [2, 5, 4, usize::MAX, usize::MAX, usize::MAX],
    [2, 4, 5, usize::MAX, usize::MAX, usize::MAX],
    [0, 4, 5, 0, 5, 1],
    [0, 5, 2, 0, 3, 5],
    [1, 5, 3, usize::MAX, usize::MAX, usize::MAX],
    [1, 4, 2, 1, 3, 4],
    [0, 4, 3, usize::MAX, usize::MAX, usize::MAX],
    [0, 2, 1, usize::MAX, usize::MAX, usize::MAX],
    [usize::MAX; 6],
];

/// Adds whatever of the surface passes through one tetrahedron.
fn emit_tetrahedron(
    mesh: &mut Mesh,
    welded: &mut HashMap<(u32, u32), u32>,
    iso: f32,
    corners: [(u32, [f32; 3], f32); 4],
) {
    let mut code = 0usize;
    for (bit, (_, _, value)) in corners.iter().enumerate() {
        if *value < iso {
            code |= 1 << bit;
        }
    }
    let table = TETRAHEDRON_TABLE[code];
    if table[0] == usize::MAX {
        return;
    }

    let mut vertex = |edge: usize| -> u32 {
        let (from, to) = TETRAHEDRON_EDGES[edge];
        let (first, second) = (corners[from], corners[to]);
        // The key is the pair of grid corners, smaller first, so the two cells
        // either side of the edge agree on it.
        let key = match first.0 < second.0 {
            true => (first.0, second.0),
            false => (second.0, first.0),
        };
        if let Some(index) = welded.get(&key) {
            return *index;
        }
        let span = second.2 - first.2;
        // Where the field crosses the isovalue, linearly. A flat edge is cut in
        // half, which is arbitrary but keeps the vertex on the edge.
        let t = match span == 0.0 {
            true => 0.5,
            false => ((iso - first.2) / span).clamp(0.0, 1.0),
        };
        let point = add3(first.1, scale(sub(second.1, first.1), t));
        let index = mesh.positions.len() as u32;
        mesh.positions.push(point);
        welded.insert(key, index);
        index
    };

    for triangle in table.chunks_exact(3) {
        if triangle[0] == usize::MAX {
            break;
        }
        let face = [
            vertex(triangle[0]),
            vertex(triangle[1]),
            vertex(triangle[2]),
        ];
        // A tetrahedron with two corners at exactly the isovalue can produce a
        // triangle with a repeated vertex, which is not a face.
        if face[0] != face[1] && face[1] != face[2] && face[0] != face[2] {
            mesh.indices.push(face);
        }
    }
}

/// Mean of the two one-sided nearest-neighbour distances between two point
/// sets, which is how a reconstruction is scored against its original.
///
/// Symmetric on purpose: the one-sided distance alone is happy with a
/// reconstruction that covers part of the shape very well and leaves the rest
/// off entirely.
pub fn chamfer_distance(a: &[[f32; 3]], b: &[[f32; 3]]) -> f32 {
    fn one_sided(from: &[[f32; 3]], to: &[[f32; 3]]) -> f32 {
        from.par_iter()
            .map(|point| {
                to.iter()
                    .map(|other| length(sub(*point, *other)))
                    .fold(f32::INFINITY, f32::min)
            })
            .sum::<f32>()
            / from.len().max(1) as f32
    }
    (one_sided(a, b) + one_sided(b, a)) / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    /// A sphere as a latitude-longitude grid, which gives the tests a mesh with
    /// a distance field they can check against arithmetic.
    fn sphere(radius: f32, rings: usize, segments: usize) -> Mesh {
        let mut mesh = Mesh::default();
        for ring in 0..=rings {
            let phi = std::f32::consts::PI * ring as f32 / rings as f32;
            for segment in 0..=segments {
                let theta = std::f32::consts::TAU * segment as f32 / segments as f32;
                mesh.positions.push([
                    radius * phi.sin() * theta.cos(),
                    radius * phi.cos(),
                    radius * phi.sin() * theta.sin(),
                ]);
            }
        }
        let stride = segments + 1;
        for ring in 0..rings {
            for segment in 0..segments {
                let a = (ring * stride + segment) as u32;
                let (b, c, d) = (a + 1, a + stride as u32, a + stride as u32 + 1);
                // Wound so the normals point out of the sphere.
                mesh.indices.push([a, b, c]);
                mesh.indices.push([b, d, c]);
            }
        }
        mesh
    }

    /// An axis-aligned box, which has flat faces and sharp edges where a sphere
    /// has neither.
    fn cube(half: f32) -> Mesh {
        let mut mesh = Mesh::default();
        for corner in 0..8 {
            mesh.positions.push([
                if corner & 1 == 0 { -half } else { half },
                if corner & 2 == 0 { -half } else { half },
                if corner & 4 == 0 { -half } else { half },
            ]);
        }
        // Two triangles per face, wound so the normals point outwards.
        for face in [
            [0, 2, 1, 3],
            [4, 5, 6, 7],
            [0, 1, 4, 5],
            [2, 6, 3, 7],
            [0, 4, 2, 6],
            [1, 3, 5, 7],
        ] {
            mesh.indices.push([face[0], face[1], face[2]]);
            mesh.indices.push([face[1], face[3], face[2]]);
        }
        mesh
    }

    #[test]
    fn normalizing_fills_the_cube_without_changing_the_shape() {
        let mut mesh = sphere(3.0, 8, 12);
        for position in &mut mesh.positions {
            position[0] += 10.0;
        }
        let transform = mesh.normalize();
        let (low, high) = mesh.bounds();
        for axis in 0..3 {
            assert!((low[axis] + 1.0).abs() < 1e-5, "{low:?}");
            assert!((high[axis] - 1.0).abs() < 1e-5, "{high:?}");
        }
        // The transform puts a point back where it came from.
        let restored = transform.invert(mesh.positions[0]);
        assert!((restored[0] - 10.0).abs() < 1e-3, "{restored:?}");
    }

    #[test]
    fn the_winding_number_separates_inside_from_outside() {
        let mesh = cube(1.0);
        let bvh = Bvh::build(&mesh);
        for point in [[0.0, 0.0, 0.0], [0.5, -0.5, 0.25], [0.9, 0.9, 0.9]] {
            let winding = bvh.winding_number(point);
            assert!((winding - 1.0).abs() < 0.05, "{point:?} read {winding}");
        }
        for point in [[2.0, 0.0, 0.0], [0.0, -3.0, 0.0], [1.5, 1.5, 1.5]] {
            let winding = bvh.winding_number(point);
            assert!(winding.abs() < 0.05, "{point:?} read {winding}");
        }
    }

    #[test]
    fn a_holed_mesh_still_classifies_its_interior() {
        // A ray cast needs a closed surface to count crossings against, which
        // is the case this is here to cover: drop a face and the answer should
        // still be close to one deep inside.
        let mut mesh = cube(1.0);
        mesh.indices.truncate(mesh.indices.len() - 2);
        let bvh = Bvh::build(&mesh);
        assert!(bvh.winding_number([0.0, 0.0, 0.0]) > 0.7);
        assert!(bvh.winding_number([4.0, 0.0, 0.0]).abs() < 0.05);
    }

    #[test]
    fn the_signed_distance_to_a_sphere_is_the_radius_minus_the_offset() {
        let mesh = sphere(1.0, 64, 128);
        let bvh = Bvh::build(&mesh);
        let queries = [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ];
        let distances = bvh.signed_distance(&queries);
        for (query, distance) in queries.iter().zip(&distances) {
            let expected = length(*query) - 1.0;
            // The mesh is a polyhedron inscribed in the sphere, so it sits a
            // little inside it.
            assert!(
                (distance - expected).abs() < 0.01,
                "{query:?}: {distance} against {expected}"
            );
        }
    }

    /// Colours interpolate linearly across a face, so a mesh whose colour is a
    /// linear function of position must sample back as that same function.
    #[test]
    fn a_sampled_colour_is_the_colour_at_the_sampled_point() {
        let mesh = Mesh {
            positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            indices: vec![[0, 1, 2]],
            colors: vec![[0.0, 0.0, 0.5], [1.0, 0.0, 0.5], [0.0, 1.0, 0.5]],
            ..Mesh::default()
        };
        let bvh = Bvh::build(&mesh);
        let mut rng = StdRng::seed_from_u64(7);

        let (points, _, colors) = bvh.sample_surface_colored(128, &mut rng);
        assert_eq!(colors.len(), points.len());
        for (point, color) in points.iter().zip(&colors) {
            assert!((color[0] - point[0]).abs() < 1e-5);
            assert!((color[1] - point[1]).abs() < 1e-5);
            assert!((color[2] - 0.5).abs() < 1e-5);
        }
    }

    #[test]
    fn surface_samples_land_on_the_surface_and_spread_over_it() {
        let mesh = sphere(1.0, 32, 64);
        let bvh = Bvh::build(&mesh);
        let mut rng = StdRng::seed_from_u64(7);
        let (points, normals) = bvh.sample_surface(2_000, &mut rng);
        assert_eq!(points.len(), 2_000);
        for (point, normal) in points.iter().zip(&normals) {
            assert!((length(*point) - 1.0).abs() < 0.01, "{point:?}");
            // Outward winding means the normal points away from the centre.
            assert!(dot(*normal, *point) > 0.0, "{normal:?} at {point:?}");
        }
        // Area weighting means every octant gets samples; weighting by face
        // would crowd them around the poles, where the rings are finest.
        let mut octants = [0usize; 8];
        for point in &points {
            let index = (point[0] > 0.0) as usize
                | ((point[1] > 0.0) as usize) << 1
                | ((point[2] > 0.0) as usize) << 2;
            octants[index] += 1;
        }
        assert!(octants.iter().all(|count| *count > 150), "{octants:?}");
    }

    #[test]
    fn query_points_split_between_the_surface_and_the_box() {
        let mesh = sphere(1.0, 16, 32);
        let bvh = Bvh::build(&mesh);
        let mut rng = StdRng::seed_from_u64(11);
        let config = QuerySampling {
            count: 4_000,
            near_surface: 0.75,
            jitter: 0.01,
            extent: 1.1,
        };
        let queries = bvh.sample_queries(&config, &mut rng);
        assert_eq!(queries.len(), 4_000);
        let near = queries
            .iter()
            .filter(|point| (length(**point) - 1.0).abs() < 0.05)
            .count();
        // Every near-surface point is close, and a few uniform ones land close
        // by chance, so the count is at least the near-surface share.
        assert!(near >= 3_000, "{near} of 4000 near the surface");
        assert!(
            queries
                .iter()
                .all(|point| point.iter().all(|value| value.abs() <= 1.2)),
            "a query left the box"
        );
    }

    #[test]
    fn marching_tetrahedra_puts_vertices_on_the_isosurface() {
        let mesh = marching_tetrahedra(
            |points| {
                Ok((0..points.rows)
                    .map(|row| {
                        let point = points.row(row);
                        (point[0] * point[0] + point[1] * point[1] + point[2] * point[2]).sqrt()
                            - 0.6
                    })
                    .collect())
            },
            24,
            ([-1.0; 3], [1.0; 3]),
            0.0,
        )
        .unwrap();

        assert!(!mesh.indices.is_empty());
        for position in &mesh.positions {
            // A vertex is placed by linear interpolation along a grid edge, so
            // it is off by the curvature over one cell.
            assert!((length(*position) - 0.6).abs() < 0.02, "{position:?}");
        }
        assert_eq!(mesh.normals.len(), mesh.positions.len());
    }

    #[test]
    fn the_reconstructed_surface_is_watertight() {
        let mesh = marching_tetrahedra(
            |points| {
                Ok((0..points.rows)
                    .map(|row| {
                        let point = points.row(row);
                        point[0].abs().max(point[1].abs()).max(point[2].abs()) - 0.5
                    })
                    .collect())
            },
            16,
            ([-1.0; 3], [1.0; 3]),
            0.0,
        )
        .unwrap();

        // Every edge shared by exactly two faces is what closed means, and it
        // is the property the shared-vertex welding is there to give.
        let mut edges: HashMap<(u32, u32), usize> = HashMap::new();
        for face in &mesh.indices {
            for corner in 0..3 {
                let (a, b) = (face[corner], face[(corner + 1) % 3]);
                *edges.entry((a.min(b), a.max(b))).or_default() += 1;
            }
        }
        let open = edges.values().filter(|count| **count != 2).count();
        assert_eq!(open, 0, "{open} edges are not shared by two faces");
    }

    #[test]
    fn a_mesh_survives_a_trip_through_its_own_distance_field() {
        // The check the whole stage exists for: sample a mesh's signed
        // distance, rebuild a mesh from it, and compare the two surfaces.
        let mut original = sphere(0.8, 24, 48);
        original.normalize();
        let bvh = Bvh::build(&original);

        let rebuilt = marching_tetrahedra(
            |points| {
                let queries: Vec<[f32; 3]> = (0..points.rows)
                    .map(|row| {
                        let row = points.row(row);
                        [row[0], row[1], row[2]]
                    })
                    .collect();
                Ok(bvh.signed_distance(&queries))
            },
            48,
            ([-1.2; 3], [1.2; 3]),
            0.0,
        )
        .unwrap();

        // Point to surface rather than point to point: two independent clouds
        // of a few thousand samples are about 0.06 apart on a unit sphere
        // purely from where the samples landed, which would swamp the error
        // being measured.
        let mut rng = StdRng::seed_from_u64(3);
        let rebuilt_bvh = Bvh::build(&rebuilt);
        let (samples, _) = bvh.sample_surface(3_000, &mut rng);
        let onto_rebuilt = samples
            .iter()
            .map(|point| rebuilt_bvh.closest_point(*point).unwrap().1)
            .fold(0.0f32, f32::max);
        let onto_original = rebuilt
            .positions
            .iter()
            .map(|point| bvh.closest_point(*point).unwrap().1)
            .fold(0.0f32, f32::max);

        // The grid is 48 samples over 2.4 units, so a cell is 0.051 across and
        // a vertex placed by interpolating along one edge is off by the
        // curvature over that cell.
        assert!(onto_rebuilt < 0.02, "original to rebuilt {onto_rebuilt}");
        assert!(onto_original < 0.02, "rebuilt to original {onto_original}");
    }

    #[test]
    fn the_chamfer_distance_is_zero_for_a_cloud_against_itself() {
        let mesh = sphere(1.0, 16, 32);
        let bvh = Bvh::build(&mesh);
        let mut rng = StdRng::seed_from_u64(5);
        let (points, _) = bvh.sample_surface(500, &mut rng);
        assert!(chamfer_distance(&points, &points) < 1e-6);

        // A rigid shift moves every nearest neighbour by the same amount, so
        // the answer is the shift itself.
        let shifted: Vec<[f32; 3]> = points
            .iter()
            .map(|point| [point[0] + 0.1, point[1], point[2]])
            .collect();
        let distance = chamfer_distance(&points, &shifted);
        assert!(distance > 0.05 && distance <= 0.1, "{distance}");
    }

    #[test]
    fn an_obj_file_round_trips() {
        let path =
            std::env::temp_dir().join(format!("rusting-brain-mesh-{}.obj", std::process::id()));
        let mut original = cube(1.0);
        original.recompute_normals();
        original.write_obj(&path).unwrap();

        let read = Mesh::read_obj(&path).unwrap();
        assert_eq!(read.indices.len(), original.indices.len());
        assert_eq!(read.positions.len(), original.positions.len());
        // The reader renumbers vertices in the order the faces first name
        // them, so the faces are what has to match, not the vertex list.
        for face in 0..read.indices.len() {
            for (a, b) in read.triangle(face).iter().zip(&original.triangle(face)) {
                assert!(length(sub(*a, *b)) < 1e-6, "{a:?} against {b:?}");
            }
        }
        assert_eq!(read.normals.len(), read.positions.len());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn a_quad_face_is_triangulated_and_a_negative_index_counts_back() {
        let path =
            std::env::temp_dir().join(format!("rusting-brain-quad-{}.obj", std::process::id()));
        std::fs::write(&path, "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf -4 -3 -2 -1\n").unwrap();
        let mesh = Mesh::read_obj(&path).unwrap();
        assert_eq!(mesh.indices.len(), 2, "a quad is two triangles");
        assert_eq!(mesh.positions.len(), 4);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn an_obj_texture_coordinate_is_flipped_into_the_gltf_convention() {
        let path =
            std::env::temp_dir().join(format!("rusting-brain-uv-{}.obj", std::process::id()));
        std::fs::write(
            &path,
            "v 0 0 0\nv 1 0 0\nv 1 1 0\nvt 0 0\nvt 1 0.25\nvt 1 1\nf 1/1 2/2 3/3\n",
        )
        .unwrap();
        let mesh = Mesh::read_obj(&path).unwrap();
        assert_eq!(mesh.uvs, vec![[0.0, 1.0], [1.0, 0.75], [1.0, 0.0]]);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn a_file_with_no_faces_is_refused() {
        let path =
            std::env::temp_dir().join(format!("rusting-brain-empty-{}.obj", std::process::id()));
        std::fs::write(&path, "v 0 0 0\nv 1 0 0\n").unwrap();
        assert!(Mesh::read_obj(&path).is_err());
        std::fs::remove_file(&path).ok();
    }
}

// -- Simplification ----------------------------------------------------------

/// A symmetric 4x4 quadric, as its ten distinct coefficients.
///
/// The order is the upper triangle read row by row: `a2 ab ac ad b2 bc bd c2
/// cd d2`, where `(a, b, c, d)` is a plane with a unit normal. The error at a
/// point is `v^T Q v` with `v = (x, y, z, 1)`, which is the sum of squared
/// distances to every plane that went into it.
type Quadric = [f32; 10];

/// The quadric of one plane, scaled by how much surface it stands for.
fn plane_quadric(normal: [f32; 3], point: [f32; 3], weight: f32) -> Quadric {
    let [a, b, c] = normal;
    let d = -dot(normal, point);
    [
        a * a,
        a * b,
        a * c,
        a * d,
        b * b,
        b * c,
        b * d,
        c * c,
        c * d,
        d * d,
    ]
    .map(|value| value * weight)
}

fn add_quadric(left: Quadric, right: Quadric) -> Quadric {
    std::array::from_fn(|index| left[index] + right[index])
}

/// `v^T Q v`, the squared distance to the planes the quadric was built from.
fn quadric_error(quadric: Quadric, [x, y, z]: [f32; 3]) -> f32 {
    let [a2, ab, ac, ad, b2, bc, bd, c2, cd, d2] = quadric;
    a2 * x * x
        + 2.0 * ab * x * y
        + 2.0 * ac * x * z
        + 2.0 * ad * x
        + b2 * y * y
        + 2.0 * bc * y * z
        + 2.0 * bd * y
        + c2 * z * z
        + 2.0 * cd * z
        + d2
}

/// The point that minimizes the quadric, if there is a unique one.
///
/// The minimum is where the gradient vanishes, which is a 3x3 solve. A flat
/// region or a straight edge leaves the system singular — there is a whole
/// plane or line of equally good answers — and the caller falls back to
/// picking the best of the candidates it already has.
fn quadric_minimum(quadric: Quadric) -> Option<[f32; 3]> {
    let [a2, ab, ac, ad, b2, bc, bd, c2, cd, _] = quadric;
    let matrix = [[a2, ab, ac], [ab, b2, bc], [ac, bc, c2]];
    let rhs = [-ad, -bd, -cd];

    let determinant = matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]);
    // The scale of the quadric sets the scale of the determinant, so the
    // threshold is relative to it rather than an absolute epsilon.
    let scale = (a2 + b2 + c2).abs().max(1e-12);
    if determinant.abs() < 1e-9 * scale * scale * scale {
        return None;
    }

    // Cramer's rule: three determinants with one column replaced.
    Some(std::array::from_fn(|column| {
        let mut replaced = matrix;
        for row in 0..3 {
            replaced[row][column] = rhs[row];
        }
        (replaced[0][0] * (replaced[1][1] * replaced[2][2] - replaced[1][2] * replaced[2][1])
            - replaced[0][1] * (replaced[1][0] * replaced[2][2] - replaced[1][2] * replaced[2][0])
            + replaced[0][2] * (replaced[1][0] * replaced[2][1] - replaced[1][1] * replaced[2][0]))
            / determinant
    }))
}

/// One pending edge collapse, ordered by the error it would cost.
///
/// `stamp` is how many collapses each endpoint had taken when this was queued.
/// A vertex that has moved since invalidates the entry, which is what lets the
/// queue hold stale entries instead of being searched and repaired.
struct Collapse {
    key: u32,
    from: u32,
    into: u32,
    stamps: [u32; 2],
    position: [f32; 3],
}

impl Ord for Collapse {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reversed, because `BinaryHeap` is a max-heap and the cheapest
        // collapse is the one to take.
        other.key.cmp(&self.key)
    }
}
impl PartialOrd for Collapse {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl PartialEq for Collapse {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}
impl Eq for Collapse {}

impl Mesh {
    /// Collapses edges until the mesh is down to `target_faces`, cheapest
    /// first.
    ///
    /// This is Garland and Heckbert's quadric error metric: every vertex
    /// carries the sum of the squared-distance quadrics of the faces around
    /// it, an edge costs whatever that sum evaluates to at the point it would
    /// collapse onto, and the point itself is the one that minimizes it. A
    /// boundary edge gets an extra plane at right angles to its face, weighted
    /// heavily, so an open edge is held in place rather than eaten away.
    ///
    /// Returns the face count it finished on, which can be above the target if
    /// there was nothing left that could be collapsed safely.
    ///
    /// Normals are recomputed if the mesh had them. UVs are dropped: a
    /// collapse merges two vertices that generally sit in different places in
    /// a texture atlas, and inventing a coordinate for the merged one would be
    /// worse than admitting there isn't one.
    pub fn simplify(&mut self, target_faces: usize) -> usize {
        self.collapse_edges(target_faces, f32::INFINITY).0
    }

    /// Collapses edges for as long as the quadric error stays under
    /// `max_error`, and returns the face count it finished on together with the
    /// largest error it accepted.
    ///
    /// This is [`Mesh::simplify`] asking the other question. A face target says
    /// how big the mesh may be and lets the error land where it lands, which is
    /// the wrong way round for a pipeline that extracted the mesh from a field
    /// and wants it as small as it can be *without* visibly moving the surface.
    /// Given a bound, this collapses the cheap edges — a flat region, a
    /// redundant row of vertices along a plane — and stops at the first edge
    /// that would cost more than the bound.
    ///
    /// The units are the metric's, not the world's: an error is a sum of
    /// squared distances to the planes around the edge, each weighted by the
    /// area of the face it came from. So it scales with the mesh, and the way
    /// to pick a bound is to simplify one representative mesh to a face count
    /// that looks right and use the largest error that run reports. The second
    /// return value is there for exactly that.
    ///
    /// Everything else matches [`Mesh::simplify`]: the same metric, the same
    /// refusal to fold a face over, normals recomputed, UVs dropped.
    pub fn simplify_within(&mut self, max_error: f32) -> (usize, f32) {
        self.collapse_edges(0, max_error)
    }

    /// The collapse loop both entry points run: cheapest edge first, stopping
    /// at `target_faces` or at the first collapse over `max_error`, whichever
    /// comes first.
    fn collapse_edges(&mut self, target_faces: usize, max_error: f32) -> (usize, f32) {
        let mut worst = 0.0f32;
        if self.indices.len() <= target_faces {
            return (self.indices.len(), worst);
        }

        let mut positions = self.positions.clone();
        let mut faces = self.indices.clone();
        let mut face_alive = vec![true; faces.len()];
        let mut live_faces = faces.len();

        // A vertex's quadric is the sum over the faces touching it, each one
        // weighted by its area so a sliver does not count as much as a slab.
        let mut quadrics = vec![[0.0f32; 10]; positions.len()];
        let mut vertex_faces: Vec<Vec<u32>> = vec![Vec::new(); positions.len()];
        for (face, corners) in faces.iter().enumerate() {
            let [a, b, c] = corners.map(|corner| positions[corner as usize]);
            let cross = cross(sub(b, a), sub(c, a));
            let area = length(cross) / 2.0;
            let quadric = plane_quadric(normalize(cross), a, area);
            for corner in *corners {
                quadrics[corner as usize] = add_quadric(quadrics[corner as usize], quadric);
                vertex_faces[corner as usize].push(face as u32);
            }
        }

        // An edge with one face behind it is a boundary. The plane standing on
        // it, at right angles to that face, is what keeps the boundary where
        // it is; the weight is what makes it expensive to cross.
        let mut edge_faces: HashMap<(u32, u32), usize> = HashMap::new();
        for corners in &faces {
            for corner in 0..3 {
                let edge = ordered(corners[corner], corners[(corner + 1) % 3]);
                *edge_faces.entry(edge).or_insert(0) += 1;
            }
        }
        for corners in &faces {
            let [a, b, c] = corners.map(|corner| positions[corner as usize]);
            let normal = cross(sub(b, a), sub(c, a));
            let area = length(normal) / 2.0;
            let normal = normalize(normal);
            for corner in 0..3 {
                let (first, second) = (corners[corner], corners[(corner + 1) % 3]);
                if edge_faces.get(&ordered(first, second)) != Some(&1) {
                    continue;
                }
                let along = sub(positions[second as usize], positions[first as usize]);
                let wall = normalize(cross(along, normal));
                let quadric =
                    plane_quadric(wall, positions[first as usize], area * BOUNDARY_WEIGHT);
                for vertex in [first, second] {
                    quadrics[vertex as usize] = add_quadric(quadrics[vertex as usize], quadric);
                }
            }
        }

        let mut stamps = vec![0u32; positions.len()];
        let mut vertex_alive = vec![true; positions.len()];
        let mut queue = std::collections::BinaryHeap::new();
        for edge in edge_faces.keys() {
            if let Some(collapse) = weigh(*edge, &positions, &quadrics, &stamps) {
                queue.push(collapse);
            }
        }

        while live_faces > target_faces {
            let Some(collapse) = queue.pop() else { break };
            let (from, into) = (collapse.from as usize, collapse.into as usize);
            if !vertex_alive[from] || !vertex_alive[into] {
                continue;
            }
            if [stamps[from], stamps[into]] != collapse.stamps {
                continue;
            }
            // The queue is ordered by error, so the first live collapse that
            // costs more than the bound is also the cheapest one left.
            let error = f32::from_bits(collapse.key);
            if error > max_error {
                break;
            }
            // A collapse that turns a face inside out is a fold, and a folded
            // mesh is worse than a coarse one.
            // NOTE: a refused edge is dropped rather than re-priced, so an
            // edge that only became safe later is never reconsidered. Pushing
            // it back risks cycling on a collapse that is never taken; if a
            // target is routinely missed, give the entry a retry count.
            if folds(
                &positions,
                &faces,
                &face_alive,
                &vertex_faces,
                from,
                into,
                collapse.position,
            ) {
                continue;
            }

            worst = worst.max(error);
            positions[into] = collapse.position;
            quadrics[into] = add_quadric(quadrics[into], quadrics[from]);
            vertex_alive[from] = false;
            stamps[into] += 1;

            // The faces along the collapsed edge become degenerate and go; the
            // rest move over to the surviving vertex.
            let moved = std::mem::take(&mut vertex_faces[from]);
            for face in moved {
                if !face_alive[face as usize] {
                    continue;
                }
                let corners = &mut faces[face as usize];
                if corners.contains(&(into as u32)) {
                    face_alive[face as usize] = false;
                    live_faces -= 1;
                    continue;
                }
                for corner in corners.iter_mut() {
                    if *corner == from as u32 {
                        *corner = into as u32;
                    }
                }
                vertex_faces[into].push(face);
            }
            vertex_faces[into].retain(|face| face_alive[*face as usize]);

            // Every edge now touching the survivor is a new candidate, and
            // every old entry mentioning it is stale by its stamp.
            let neighbours: Vec<u32> = vertex_faces[into]
                .iter()
                .flat_map(|face| faces[*face as usize])
                .filter(|corner| *corner != into as u32)
                .collect();
            for neighbour in neighbours {
                if !vertex_alive[neighbour as usize] {
                    continue;
                }
                if let Some(collapse) = weigh(
                    ordered(into as u32, neighbour),
                    &positions,
                    &quadrics,
                    &stamps,
                ) {
                    queue.push(collapse);
                }
            }
        }

        self.compact(positions, faces, face_alive);
        (self.indices.len(), worst)
    }

    /// Rebuilds the mesh from the surviving faces, dropping orphaned vertices.
    fn compact(&mut self, positions: Vec<[f32; 3]>, faces: Vec<[u32; 3]>, alive: Vec<bool>) {
        let mut remap = vec![u32::MAX; positions.len()];
        let had_normals = !self.normals.is_empty();
        // A collapse moves a vertex but does not blend what it carries, so a
        // surviving vertex keeps its own colour. Texture coordinates get no
        // such treatment: a moved vertex's UV names the wrong place in an atlas,
        // and there is no atlas here to be wrong about.
        let colours = std::mem::take(&mut self.colors);
        self.positions.clear();
        self.indices.clear();
        self.normals.clear();
        self.uvs.clear();

        for (face, corners) in faces.iter().enumerate() {
            // A collapse can leave a face whose three corners are not three
            // distinct vertices any more; it has no area and no place in the
            // output.
            if !alive[face]
                || corners[0] == corners[1]
                || corners[1] == corners[2]
                || corners[0] == corners[2]
            {
                continue;
            }
            let mut rebased = [0u32; 3];
            for (slot, corner) in rebased.iter_mut().zip(corners) {
                let corner = *corner as usize;
                if remap[corner] == u32::MAX {
                    remap[corner] = self.positions.len() as u32;
                    self.positions.push(positions[corner]);
                    if let Some(colour) = colours.get(corner) {
                        self.colors.push(*colour);
                    }
                }
                *slot = remap[corner];
            }
            self.indices.push(rebased);
        }

        if had_normals {
            self.recompute_normals();
        }
    }
}

/// How much more a boundary plane counts than the faces around it. Large
/// enough that an open edge is the last thing to go, small enough that a
/// boundary vertex can still slide along its own edge.
const BOUNDARY_WEIGHT: f32 = 1000.0;

fn ordered(first: u32, second: u32) -> (u32, u32) {
    match first < second {
        true => (first, second),
        false => (second, first),
    }
}

/// Prices one edge and picks the point it would collapse onto.
fn weigh(
    edge: (u32, u32),
    positions: &[[f32; 3]],
    quadrics: &[Quadric],
    stamps: &[u32],
) -> Option<Collapse> {
    let (from, into) = edge;
    if from == into {
        return None;
    }
    let quadric = add_quadric(quadrics[from as usize], quadrics[into as usize]);
    let candidates = [
        quadric_minimum(quadric),
        Some(positions[from as usize]),
        Some(positions[into as usize]),
        Some(scale(
            add3(positions[from as usize], positions[into as usize]),
            0.5,
        )),
    ];
    let (position, error) = candidates
        .into_iter()
        .flatten()
        .map(|point| (point, quadric_error(quadric, point)))
        .min_by(|left, right| left.1.total_cmp(&right.1))?;

    Some(Collapse {
        // Rounding can leave the error a hair below zero, and a negative float
        // does not sort by its bits.
        key: error.max(0.0).to_bits(),
        // The survivor is the second of the pair, so the collapse is
        // deterministic whichever way the edge was found.
        from,
        into,
        stamps: [stamps[from as usize], stamps[into as usize]],
        position,
    })
}

/// Whether moving `into` to `position` and merging `from` onto it would turn
/// any surviving face inside out.
fn folds(
    positions: &[[f32; 3]],
    faces: &[[u32; 3]],
    face_alive: &[bool],
    vertex_faces: &[Vec<u32>],
    from: usize,
    into: usize,
    position: [f32; 3],
) -> bool {
    for vertex in [from, into] {
        for face in &vertex_faces[vertex] {
            let corners = faces[*face as usize];
            if !face_alive[*face as usize] {
                continue;
            }
            // The faces along the edge itself are the ones meant to vanish.
            if corners.contains(&(from as u32)) && corners.contains(&(into as u32)) {
                continue;
            }
            let before = corners.map(|corner| positions[corner as usize]);
            let after =
                corners.map(
                    |corner| match corner as usize == from || corner as usize == into {
                        true => position,
                        false => positions[corner as usize],
                    },
                );
            let old = cross(sub(before[1], before[0]), sub(before[2], before[0]));
            let new = cross(sub(after[1], after[0]), sub(after[2], after[0]));
            if dot(old, new) <= 0.0 {
                return true;
            }
        }
    }
    false
}

#[cfg(test)]
mod simplify_tests {
    use super::*;

    /// A flat grid of `n x n` quads, triangulated. Every interior vertex sits
    /// exactly on the plane its neighbours define, so collapsing it costs
    /// nothing.
    fn grid(n: usize) -> Mesh {
        let mut mesh = Mesh::default();
        for row in 0..=n {
            for column in 0..=n {
                mesh.positions
                    .push([column as f32 / n as f32, row as f32 / n as f32, 0.0]);
            }
        }
        let index = |row: usize, column: usize| (row * (n + 1) + column) as u32;
        for row in 0..n {
            for column in 0..n {
                mesh.indices.push([
                    index(row, column),
                    index(row, column + 1),
                    index(row + 1, column),
                ]);
                mesh.indices.push([
                    index(row, column + 1),
                    index(row + 1, column + 1),
                    index(row + 1, column),
                ]);
            }
        }
        mesh
    }

    /// A sphere from the distance field, which is what the pipeline actually
    /// hands the simplifier.
    fn sphere(resolution: usize) -> Mesh {
        marching_tetrahedra(
            |points| {
                Ok((0..points.rows)
                    .map(|row| {
                        let point = points.row(row);
                        (point[0] * point[0] + point[1] * point[1] + point[2] * point[2]).sqrt()
                            - 0.8
                    })
                    .collect())
            },
            resolution,
            ([-1.0; 3], [1.0; 3]),
            0.0,
        )
        .unwrap()
    }

    #[test]
    fn a_target_above_the_face_count_changes_nothing() {
        let mut mesh = grid(4);
        let before = mesh.indices.clone();
        assert_eq!(mesh.simplify(10_000), before.len());
        assert_eq!(mesh.indices, before);
    }

    #[test]
    fn a_flat_sheet_collapses_to_its_corners_for_nothing() {
        let mut mesh = grid(6);
        assert_eq!(mesh.indices.len(), 72);
        mesh.simplify(2);

        // The interior is free to remove and the boundary is not, so what
        // survives is the four corners and the two triangles across them.
        assert_eq!(mesh.indices.len(), 2);
        assert_eq!(mesh.positions.len(), 4);
        let (low, high) = mesh.bounds();
        assert_eq!(low, [0.0, 0.0, 0.0]);
        assert_eq!(high, [1.0, 1.0, 0.0]);
    }

    #[test]
    fn an_error_bound_removes_the_flat_interior_and_keeps_the_shape() {
        // A flat sheet: every interior collapse is free, so a bound of nothing
        // at all still takes the mesh down to its corners.
        let mut sheet = grid(6);
        let (faces, worst) = sheet.simplify_within(1e-6);
        assert_eq!(faces, 2, "a flat sheet costs nothing to simplify");
        assert!(worst < 1e-6, "the worst collapse taken was {worst}");

        // A sphere is curved everywhere, so the same bound buys much less.
        let mut curved = sphere(24);
        let before = curved.indices.len();
        let (faces, worst) = curved.simplify_within(1e-6);
        assert!(faces < before, "{before} to {faces}");
        assert!(worst <= 1e-6);
        for point in &curved.positions {
            let distance = (length(*point) - 0.8).abs();
            assert!(distance < 0.02, "{point:?} sits {distance} off the surface");
        }
    }

    #[test]
    fn a_looser_bound_reaches_a_smaller_mesh_and_says_what_it_cost() {
        let mut tight = sphere(24);
        let (tight_faces, tight_worst) = tight.simplify_within(1e-6);
        let mut loose = sphere(24);
        let (loose_faces, loose_worst) = loose.simplify_within(1e-3);

        assert!(
            loose_faces < tight_faces,
            "a looser bound should collapse more: {loose_faces} against {tight_faces}"
        );
        assert!(loose_worst > tight_worst);
        assert!(
            loose_worst <= 1e-3,
            "the bound was not exceeded: {loose_worst}"
        );
    }

    #[test]
    fn a_bound_of_zero_takes_only_the_collapses_that_cost_nothing() {
        // A marched sphere carries rows of vertices that sit exactly on the
        // plane of their neighbours, which the metric prices at zero. Those go;
        // nothing that moves the surface does.
        let mut mesh = sphere(20);
        let before = mesh.indices.len();
        let (faces, worst) = mesh.simplify_within(0.0);
        assert!(faces < before, "{before} to {faces}");
        assert_eq!(worst, 0.0);
        for point in &mesh.positions {
            let distance = (length(*point) - 0.8).abs();
            assert!(distance < 0.01, "{point:?} sits {distance} off the surface");
        }
    }

    #[test]
    fn a_sphere_keeps_its_shape_at_a_fifth_of_its_faces() {
        let mut mesh = sphere(24);
        let before = mesh.indices.len();
        let radius = |mesh: &Mesh| {
            mesh.positions
                .iter()
                .map(|point| length(*point))
                .fold(0.0f32, f32::max)
        };
        let before_radius = radius(&mesh);

        let after = mesh.simplify(before / 5);
        assert!(after <= before / 5 + 8, "{before} to {after}");
        assert!(after > 0);

        // The surface is still the same sphere: no vertex has wandered off it.
        for point in &mesh.positions {
            let distance = (length(*point) - 0.8).abs();
            assert!(distance < 0.08, "{point:?} sits {distance} off the surface");
        }
        assert!((radius(&mesh) - before_radius).abs() < 0.08);
    }

    #[test]
    fn simplification_leaves_no_degenerate_or_dangling_geometry() {
        let mut mesh = sphere(20);
        mesh.recompute_normals();
        mesh.uvs = vec![[0.0, 0.0]; mesh.positions.len()];
        mesh.simplify(mesh.indices.len() / 4);

        let mut used = vec![false; mesh.positions.len()];
        for face in &mesh.indices {
            assert!(face[0] != face[1] && face[1] != face[2] && face[0] != face[2]);
            for corner in face {
                assert!((*corner as usize) < mesh.positions.len());
                used[*corner as usize] = true;
            }
        }
        assert!(used.iter().all(|used| *used), "an orphaned vertex survived");
        assert!(mesh.area() > 0.0);

        // Normals come back because there were normals; UVs do not, because a
        // collapse has nowhere sensible to put them.
        assert_eq!(mesh.normals.len(), mesh.positions.len());
        assert!(mesh.uvs.is_empty());
        for normal in &mesh.normals {
            assert!((length(*normal) - 1.0).abs() < 1e-4);
        }
    }

    /// The quadric is the machinery everything else rests on, so it is checked
    /// on its own: the error is the squared distance to the plane.
    #[test]
    fn a_quadric_measures_squared_distance_to_its_plane() {
        let quadric = plane_quadric([0.0, 0.0, 1.0], [0.0, 0.0, 2.0], 1.0);
        assert!(quadric_error(quadric, [5.0, -3.0, 2.0]).abs() < 1e-6);
        assert!((quadric_error(quadric, [1.0, 1.0, 5.0]) - 9.0).abs() < 1e-5);

        // Three planes meeting at a corner have exactly one minimum, and it is
        // the corner.
        let corner = add_quadric(
            add_quadric(
                plane_quadric([1.0, 0.0, 0.0], [1.0, 0.0, 0.0], 1.0),
                plane_quadric([0.0, 1.0, 0.0], [0.0, 2.0, 0.0], 1.0),
            ),
            plane_quadric([0.0, 0.0, 1.0], [0.0, 0.0, 3.0], 1.0),
        );
        let minimum = quadric_minimum(corner).unwrap();
        for (found, expected) in minimum.iter().zip(&[1.0, 2.0, 3.0]) {
            assert!((found - expected).abs() < 1e-4, "{minimum:?}");
        }
        // One plane on its own leaves a whole plane of minima, and no answer.
        assert!(quadric_minimum(plane_quadric([0.0, 0.0, 1.0], [0.0, 0.0, 2.0], 1.0)).is_none());
    }

    /// A collapse that would turn a face inside out is refused, which is what
    /// keeps a simplified mesh from self-intersecting.
    #[test]
    fn a_collapse_that_would_fold_a_face_is_refused() {
        let positions = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let faces = [[0, 1, 2], [0, 2, 3]];
        let alive = [true, true];
        let vertex_faces = vec![vec![0, 1], vec![0], vec![0, 1], vec![1]];

        // Pulling vertex 1 onto vertex 0 keeps both faces the right way round.
        assert!(!folds(
            &positions,
            &faces,
            &alive,
            &vertex_faces,
            1,
            0,
            [0.0, 0.0, 0.0]
        ));
        // Dragging vertex 3 far past the diagonal flips the face it is on.
        assert!(folds(
            &positions,
            &faces,
            &alive,
            &vertex_faces,
            3,
            0,
            [5.0, -5.0, 0.0]
        ));
    }
}
