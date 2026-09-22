//! Precomputes surface points and signed-distance samples for a directory of meshes.
//!
//! This is the other half of the offline pipeline: `vit_tokens` turns renders
//! into image tokens, and this turns the meshes those renders came from into
//! the point clouds the shape autoencoder trains on. Both are frozen inputs, so
//! both are computed once and read back from bfloat16 `.safetensors` shards.
//!
//! ```bash
//! cargo run --release --example mesh_samples -- \
//!     --meshes assets/ --out samples/ --surface 8192 --queries 200000
//! ```
//!
//! Each mesh contributes three tensors, named after its file stem, and a fourth
//! when it carries vertex colours:
//!
//! | Name | Shape | Contents |
//! |---|---|---|
//! | `{stem}.surface` | `[surface, 6]` | a point on the surface and its normal |
//! | `{stem}.queries` | `[queries, 4]` | a query point and its signed distance |
//! | `{stem}.transform` | `[1, 4]` | the centre and scale that fit it in the cube |
//! | `{stem}.colors` | `[surface, 3]` | the linear RGB at each surface point |
//!
//! The transform is what puts a generated mesh back in the source asset's
//! frame, so it is written even though training never reads it. The colours are
//! the colour head's targets, interpolated across the same face the surface
//! point was drawn from, and a mesh without them trains shape only.
//!
//! A run over ten thousand meshes takes hours, so it resumes: a shard whose
//! file is already there and opens cleanly is read for its names and skipped,
//! and `index.json` is rewritten after every shard rather than at the end. A
//! shard left half-written by a killed run does not open — `SafeTensors::open`
//! checks the header against the file's length — so it is simply redone.
//! Resuming assumes the same `--meshes` directory and the same `--shard`: the
//! chunking is by position, so a directory that gained a file reshuffles which
//! mesh lands in which shard.

use rand::SeedableRng;
use rand::rngs::StdRng;
use rusting_brain::mesh::{Bvh, Mesh, QuerySampling};
use rusting_brain::safetensors::{self, Dtype, SafeTensors};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

struct Args {
    meshes: PathBuf,
    out: PathBuf,
    shard: usize,
    surface: usize,
    queries: usize,
    near_surface: f32,
    jitter: f32,
    extent: f32,
    seed: u64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse()?;
    std::fs::create_dir_all(&args.out)?;

    let mut paths: Vec<PathBuf> = std::fs::read_dir(&args.meshes)?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| {
            path.extension().is_some_and(|extension| {
                matches!(extension.to_ascii_lowercase().to_str(), Some("obj" | "glb"))
            })
        })
        .collect();
    paths.sort();
    if paths.is_empty() {
        return Err(format!("no .obj or .glb files in {}", args.meshes.display()).into());
    }
    println!(
        "{} meshes, {} per shard, {} surface and {} query points each",
        paths.len(),
        args.shard,
        args.surface,
        args.queries
    );

    let sampling = QuerySampling {
        count: args.queries,
        near_surface: args.near_surface,
        jitter: args.jitter,
        extent: args.extent,
    };

    let mut index: BTreeMap<String, String> = BTreeMap::new();
    let mut skipped = 0;
    let mut colored = 0;
    let started = Instant::now();
    let shards = paths.len().div_ceil(args.shard);
    for (number, chunk) in paths.chunks(args.shard).enumerate() {
        let shard = format!("samples-{number:05}.safetensors");

        // Already done by an earlier run: take its names and move on.
        let done = completed(&args.out.join(&shard));
        if !done.is_empty() {
            for stem in &done {
                index.insert(stem.clone(), shard.clone());
            }
            write_index(&args.out, &index)?;
            println!("{shard}: {} meshes, already written", done.len());
            continue;
        }
        // One shard at a time in memory. A 200K-query mesh is 3 MB here and
        // half that on disk, so the shard size is the knob that bounds this.
        let mut sampled: Vec<(String, [usize; 2], Vec<f32>)> = Vec::new();
        let mut meshes = 0;
        for path in chunk {
            let stem = path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .ok_or_else(|| format!("{} has no usable name", path.display()))?
                .to_string();

            // One bad asset in a scraped dataset should not cost the whole run.
            let mesh = match read(path) {
                Ok(mesh) => mesh,
                Err(error) => {
                    eprintln!("skipping {}: {error}", path.display());
                    skipped += 1;
                    continue;
                }
            };
            // The seed is per mesh, so a rerun of one shard reproduces exactly
            // the samples the first run wrote.
            let mut rng = StdRng::seed_from_u64(args.seed ^ hash(&stem));
            let mut mesh = mesh;
            let placement = mesh.normalize();
            let bvh = Bvh::build(&mesh);

            let (points, normals, colors) = bvh.sample_surface_colored(args.surface, &mut rng);
            let mut surface = Vec::with_capacity(args.surface * 6);
            for (point, normal) in points.iter().zip(&normals) {
                surface.extend(point);
                surface.extend(normal);
            }

            let queries = bvh.sample_queries(&sampling, &mut rng);
            let distances = bvh.signed_distance(&queries);
            let mut samples = Vec::with_capacity(queries.len() * 4);
            for (query, distance) in queries.iter().zip(&distances) {
                samples.extend(query);
                samples.push(*distance);
            }

            index.insert(stem.clone(), shard.clone());
            meshes += 1;
            sampled.push((format!("{stem}.surface"), [points.len(), 6], surface));
            // Only the meshes that carry colour get a colour tensor; the loader
            // treats a missing one as "this mesh trains shape only".
            if !colors.is_empty() {
                colored += 1;
                sampled.push((format!("{stem}.colors"), [colors.len(), 3], colors.concat()));
            }
            sampled.push((format!("{stem}.queries"), [queries.len(), 4], samples));
            sampled.push((
                format!("{stem}.transform"),
                [1, 4],
                vec![
                    placement.center[0],
                    placement.center[1],
                    placement.center[2],
                    placement.scale,
                ],
            ));
        }
        if sampled.is_empty() {
            continue;
        }

        let borrowed: Vec<(&str, &[usize], &[f32])> = sampled
            .iter()
            .map(|(name, shape, data)| (name.as_str(), &shape[..], &data[..]))
            .collect();
        safetensors::write_as(
            args.out.join(&shard),
            &borrowed,
            &BTreeMap::from([
                ("source".to_string(), "mesh_samples".to_string()),
                ("surface".to_string(), args.surface.to_string()),
                ("queries".to_string(), args.queries.to_string()),
            ]),
            Dtype::Bf16,
        )?;
        write_index(&args.out, &index)?;
        println!(
            "{shard}: {meshes} meshes, {}",
            progress(started, number + 1, shards)
        );
    }

    write_index(&args.out, &index)?;
    println!(
        "wrote {} entries to index.json, {colored} with colour, skipped {skipped}",
        index.len()
    );
    Ok(())
}

/// The mesh names an already-written shard holds, or nothing if it is missing,
/// unreadable, or was cut short by a killed run.
fn completed(path: &Path) -> Vec<String> {
    let Ok(shard) = SafeTensors::open(path) else {
        return Vec::new();
    };
    shard
        .names()
        .filter_map(|name| name.strip_suffix(".surface").map(str::to_string))
        .collect()
}

/// Rewritten after every shard, so a killed run leaves an index that describes
/// exactly the shards that are on disk.
fn write_index(
    out: &Path,
    index: &BTreeMap<String, String>,
) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::write(out.join("index.json"), serde_json::to_vec_pretty(index)?)?;
    Ok(())
}

/// Elapsed and remaining, estimated from the shards finished so far. Shards
/// take roughly equal time, since they hold the same number of meshes.
fn progress(started: Instant, done: usize, total: usize) -> String {
    let elapsed = started.elapsed().as_secs_f32();
    let remaining = elapsed / done.max(1) as f32 * total.saturating_sub(done) as f32;
    format!(
        "{done}/{total} shards, {} elapsed, {} left",
        clock(elapsed),
        clock(remaining)
    )
}

fn clock(seconds: f32) -> String {
    let seconds = seconds as u64;
    match seconds / 60 {
        0 => format!("{seconds}s"),
        minutes if minutes < 60 => format!("{minutes}m"),
        minutes => format!("{}h{:02}m", minutes / 60, minutes % 60),
    }
}

fn read(path: &std::path::Path) -> Result<Mesh, rusting_brain::NetworkError> {
    match path.extension().and_then(|e| e.to_str()) {
        Some("glb") => Mesh::read_glb(path),
        _ => Mesh::read_obj(path),
    }
}

/// FNV-1a, so a mesh's seed depends on its name and not on its position in the
/// directory listing.
fn hash(name: &str) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in name.as_bytes() {
        hash = (hash ^ u64::from(*byte)).wrapping_mul(0x1000_0000_01b3);
    }
    hash
}

fn parse() -> Result<Args, Box<dyn std::error::Error>> {
    let mut args = Args {
        meshes: PathBuf::new(),
        out: PathBuf::from("samples"),
        shard: 64,
        surface: 8192,
        queries: 200_000,
        near_surface: 0.7,
        jitter: 0.02,
        extent: 1.1,
        seed: 0,
    };
    let mut rest = std::env::args().skip(1);
    while let Some(flag) = rest.next() {
        let mut value = || rest.next().ok_or(format!("{flag} needs a value"));
        match flag.as_str() {
            "--meshes" => args.meshes = value()?.into(),
            "--out" => args.out = value()?.into(),
            "--shard" => args.shard = value()?.parse()?,
            "--surface" => args.surface = value()?.parse()?,
            "--queries" => args.queries = value()?.parse()?,
            "--near-surface" => args.near_surface = value()?.parse()?,
            "--jitter" => args.jitter = value()?.parse()?,
            "--extent" => args.extent = value()?.parse()?,
            "--seed" => args.seed = value()?.parse()?,
            other => return Err(format!("unknown flag {other}").into()),
        }
    }
    if args.meshes.as_os_str().is_empty() {
        return Err("usage: --meshes <dir> [--out <dir>] [--shard n] [--surface n] [--queries n] [--near-surface f] [--jitter f] [--extent f] [--seed n]".into());
    }
    if args.shard == 0 {
        return Err("--shard needs at least one mesh".into());
    }
    if !(0.0..=1.0).contains(&args.near_surface) {
        return Err("--near-surface is the fraction of queries near the surface, so 0 to 1".into());
    }
    Ok(args)
}
