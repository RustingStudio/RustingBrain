//! Reading the preprocessed corpus the training loop is fed from.
//!
//! Two binaries write the corpus once: `examples/vit_tokens.rs` produces
//! `tokens-*.safetensors`, one tensor of frozen image tokens per render, and
//! `examples/mesh_samples.rs` produces `samples-*.safetensors` with a surface
//! point cloud, a set of signed-distance queries, the normalizing transform
//! and, for a mesh that has vertex colours, one colour per surface point. Both leave an `index.json` mapping a name to the shard holding
//! it.
//!
//! This module joins the two sides and hands the training loop batches.
//! Nothing is loaded until a batch asks for it: the shard header is read once
//! and the tensor bytes are seeked to and widened on demand, which is what
//! keeps a corpus larger than memory usable. A worker thread stays exactly one
//! batch ahead, so the disk and the model overlap without the queue growing.
//!
//! ```no_run
//! # use rusting_brain::shards::{Corpus, BatchConfig};
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let corpus = Corpus::open("tokens", "samples")?;
//! println!("{} examples", corpus.len());
//!
//! for batch in corpus.stream(BatchConfig { batch: 4, ..BatchConfig::default() })? {
//!     let batch = batch?;
//!     // batch.tokens is [batch * 197, 768], batch.queries is [batch * n, 3].
//!     let _ = batch.tokens.rows;
//!     break;
//! }
//! # Ok(())
//! # }
//! ```
//!
//! NOTE: no mmap. `SafeTensors` already reads a header once and seeks for
//! the bytes, which is what mmap would buy here minus a dependency and minus
//! the page-fault behaviour on a network filesystem. If a profile ever shows
//! the seek dominating, the change is inside `safetensors.rs`, not here.

use crate::matrix::Matrix;
use crate::network::NetworkError;
use crate::safetensors::SafeTensors;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// One directory of shards and the index that says which holds what.
///
/// The open shard is cached, because an index walked in order asks for the
/// same file many times running and reopening it would read the header again
/// each time.
pub struct Shards {
    root: PathBuf,
    index: BTreeMap<String, String>,
    open: Option<(String, SafeTensors)>,
}

impl Shards {
    /// Reads `index.json` from a directory written by one of the
    /// preprocessing binaries.
    pub fn open<P: AsRef<Path>>(root: P) -> Result<Self, NetworkError> {
        let root = root.as_ref().to_path_buf();
        let path = root.join("index.json");
        let index: BTreeMap<String, String> =
            serde_json::from_slice(&std::fs::read(&path).map_err(|error| {
                NetworkError::InvalidDataset(format!("{}: {error}", path.display()))
            })?)?;
        if index.is_empty() {
            return Err(NetworkError::InvalidDataset(format!(
                "{} lists no shards",
                path.display()
            )));
        }
        Ok(Self {
            root,
            index,
            open: None,
        })
    }

    /// The names in the index, in order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.index.keys().map(String::as_str)
    }

    pub fn len(&self) -> usize {
        self.index.len()
    }

    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    pub fn contains(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    /// One tensor, by the name the index knows and the suffix the writer used.
    ///
    /// The tokens shards name a tensor after the image's stem alone; the
    /// sample shards append `.surface`, `.queries` or `.transform`, which is
    /// what `suffix` carries.
    pub fn tensor(
        &mut self,
        name: &str,
        suffix: &str,
    ) -> Result<(Vec<f32>, Vec<usize>), NetworkError> {
        let file = self.shard_of(name)?;
        file.tensor(&format!("{name}{suffix}"))
    }

    /// Whether `{name}{suffix}` is in the shard, for the tensors that are
    /// optional — a mesh written without vertex colours carries no `.colors`.
    pub fn has(&mut self, name: &str, suffix: &str) -> bool {
        let full = format!("{name}{suffix}");
        self.shard_of(name)
            .is_ok_and(|file| file.names().any(|tensor| tensor == full))
    }

    /// The open shard holding `name`, opening it if another one is open.
    fn shard_of(&mut self, name: &str) -> Result<&mut SafeTensors, NetworkError> {
        let shard = self.index.get(name).cloned().ok_or_else(|| {
            NetworkError::InvalidDataset(format!("{}: nothing named {name}", self.root.display()))
        })?;
        if self.open.as_ref().map(|(open, _)| open.as_str()) != Some(shard.as_str()) {
            let file = SafeTensors::open(self.root.join(&shard))?;
            self.open = Some((shard, file));
        }
        let (_, file) = self.open.as_mut().expect("just opened");
        Ok(file)
    }

    /// The same as a [`Matrix`], which is what everything downstream wants.
    pub fn matrix(&mut self, name: &str, suffix: &str) -> Result<Matrix, NetworkError> {
        let (values, shape) = self.tensor(name, suffix)?;
        match shape.as_slice() {
            [rows, cols] => Ok(Matrix::from_vec(*rows, *cols, values)),
            _ => Err(NetworkError::InvalidDataset(format!(
                "{name}{suffix} has shape {shape:?}, which is not a matrix"
            ))),
        }
    }
}

/// One training example: a render's tokens and the mesh they describe.
pub struct Example {
    /// The render's name, which is what the tokens index knows it by.
    pub name: String,
    /// The mesh's name, which is what the samples index knows it by.
    pub mesh: String,
    /// `[tokens, cond_dim]`, frozen.
    pub tokens: Matrix,
    /// `[points, 6]`: position then normal.
    pub surface: Matrix,
    /// `[points, 3]`, linear RGB, in the same order as `surface`. `None` when
    /// the mesh was written without vertex colours.
    pub colors: Option<Matrix>,
    /// `[queries, 3]`.
    pub queries: Matrix,
    /// One signed distance per query row.
    pub distances: Vec<f32>,
}

/// A batch of examples, stacked into the packed layout the models read.
pub struct Batch {
    pub names: Vec<String>,
    /// `[batch * tokens, cond_dim]`.
    pub tokens: Matrix,
    /// `[batch * points, 6]`.
    pub surface: Matrix,
    /// `[batch * points, 3]`, present only when every example in the batch
    /// carries colours.
    ///
    /// NOTE: one mixed example drops the colour targets of the whole
    /// batch. A corpus that is mostly coloured wants a per-example mask in the
    /// colour loss instead; a corpus that is all one way or the other does not.
    pub colors: Option<Matrix>,
    /// `[batch * queries, 3]`.
    pub queries: Matrix,
    /// One per query row, in the same order.
    pub distances: Vec<f32>,
}

impl Batch {
    /// How many examples were stacked.
    pub fn len(&self) -> usize {
        self.names.len()
    }

    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }
}

/// The two halves of the corpus, paired.
pub struct Corpus {
    tokens: Shards,
    samples: Shards,
    /// Render name to mesh name, resolved once at open.
    pairs: Vec<(String, String)>,
}

impl Corpus {
    /// Opens a token directory and a sample directory and pairs them up.
    ///
    /// A render is paired with the mesh of the same name, or — since a mesh is
    /// usually rendered from several angles and the views are named
    /// `{mesh}_000` — with the mesh named by everything before the last
    /// underscore. A render that matches neither is skipped, because a corpus
    /// half-built by an interrupted preprocessing run is the normal case and
    /// refusing the whole directory over it would be useless.
    pub fn open<T: AsRef<Path>, S: AsRef<Path>>(
        tokens: T,
        samples: S,
    ) -> Result<Self, NetworkError> {
        let tokens = Shards::open(tokens)?;
        let samples = Shards::open(samples)?;

        let pairs: Vec<(String, String)> = tokens
            .names()
            .filter_map(|render| {
                let mesh = match samples.contains(render) {
                    true => Some(render.to_string()),
                    false => render
                        .rsplit_once('_')
                        .map(|(mesh, _)| mesh.to_string())
                        .filter(|mesh| samples.contains(mesh)),
                }?;
                Some((render.to_string(), mesh))
            })
            .collect();

        if pairs.is_empty() {
            return Err(NetworkError::InvalidDataset(
                "no render in the token index has a mesh in the sample index".into(),
            ));
        }
        Ok(Self {
            tokens,
            samples,
            pairs,
        })
    }

    /// How many paired examples there are.
    pub fn len(&self) -> usize {
        self.pairs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }

    /// The render and mesh names, in index order.
    pub fn pairs(&self) -> &[(String, String)] {
        &self.pairs
    }

    /// Loads one example whole, subsampling the surface and the queries.
    ///
    /// A mesh carries far more of both than a step uses — 200,000 queries is
    /// two orders of magnitude past a batch — so the subsample is what the
    /// record was written large for. `None` takes everything.
    pub fn example<R: Rng>(
        &mut self,
        index: usize,
        surface_points: Option<usize>,
        queries: Option<usize>,
        rng: &mut R,
    ) -> Result<Example, NetworkError> {
        let (name, mesh) = self
            .pairs
            .get(index)
            .cloned()
            .ok_or_else(|| NetworkError::InvalidDataset(format!("no example {index}")))?;

        let tokens = self.tokens.matrix(&name, "")?;
        let written = self.samples.matrix(&mesh, ".surface")?;
        // The colours have to keep the surface's rows, so both take the same
        // draw rather than two independent ones.
        let chosen = choose_rows(written.rows, surface_points, rng);
        let surface = take_rows(&written, &chosen);
        let colors = match self.samples.has(&mesh, ".colors") {
            true => Some(take_rows(&self.samples.matrix(&mesh, ".colors")?, &chosen)),
            false => None,
        };
        let packed = subsample(&self.samples.matrix(&mesh, ".queries")?, queries, rng);
        if packed.cols != 4 {
            return Err(NetworkError::InvalidDataset(format!(
                "{mesh}.queries is [{}, {}] and should be [n, 4]",
                packed.rows, packed.cols
            )));
        }

        // The queries are written as position and distance in one row, which
        // is one tensor instead of two on disk and a split here.
        let mut queries = Matrix::new(packed.rows, 3);
        let mut distances = Vec::with_capacity(packed.rows);
        for row in 0..packed.rows {
            queries.row_mut(row).copy_from_slice(&packed.row(row)[..3]);
            distances.push(packed.row(row)[3]);
        }

        Ok(Example {
            name,
            mesh,
            tokens,
            surface,
            colors,
            queries,
            distances,
        })
    }

    /// The normalizing transform the preprocessing binary applied, as
    /// `[center_x, center_y, center_z, scale]`.
    pub fn transform(&mut self, mesh: &str) -> Result<[f32; 4], NetworkError> {
        let (values, _) = self.samples.tensor(mesh, ".transform")?;
        values
            .get(..4)
            .map(|values| [values[0], values[1], values[2], values[3]])
            .ok_or_else(|| {
                NetworkError::InvalidDataset(format!(
                    "{mesh}.transform holds {} values, not 4",
                    values.len()
                ))
            })
    }

    /// Batches, on a worker thread that stays one batch ahead.
    ///
    /// The stream ends after `config.epochs` passes, or never if that is
    /// `None`, which is what a training loop that counts steps rather than
    /// epochs wants.
    pub fn stream(self, config: BatchConfig) -> Result<BatchStream, NetworkError> {
        BatchStream::new(self, config)
    }
}

/// `count` row numbers out of `rows`, without replacement, or `None` for all
/// of them.
fn choose_rows<R: Rng>(rows: usize, count: Option<usize>, rng: &mut R) -> Option<Vec<usize>> {
    let count = count.filter(|count| *count < rows)?;
    // A partial Fisher-Yates over the row numbers: the first `count` entries
    // are a uniform sample, and nothing the size of the source is allocated
    // twice.
    let mut order: Vec<usize> = (0..rows).collect();
    for slot in 0..count {
        order.swap(slot, rng.gen_range(slot..rows));
    }
    order.truncate(count);
    Some(order)
}

/// The rows [`choose_rows`] picked, or the whole matrix.
fn take_rows(source: &Matrix, chosen: &Option<Vec<usize>>) -> Matrix {
    let Some(chosen) = chosen else {
        return source.clone();
    };
    let mut out = Matrix::new(chosen.len(), source.cols);
    for (row, taken) in chosen.iter().enumerate() {
        out.row_mut(row).copy_from_slice(source.row(*taken));
    }
    out
}

/// Keeps `count` rows of a matrix, chosen at random without replacement.
fn subsample<R: Rng>(source: &Matrix, count: Option<usize>, rng: &mut R) -> Matrix {
    take_rows(source, &choose_rows(source.rows, count, rng))
}

/// How a [`BatchStream`] is built.
#[derive(Clone, Copy, Debug)]
pub struct BatchConfig {
    pub batch: usize,
    /// Surface points per example. `None` takes every one written.
    pub surface_points: Option<usize>,
    /// Distance queries per example. `None` takes every one written.
    pub queries: Option<usize>,
    /// Passes over the corpus. `None` never stops.
    pub epochs: Option<usize>,
    /// Whether each pass visits the examples in a fresh random order.
    pub shuffle: bool,
    pub seed: u64,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            batch: 8,
            surface_points: Some(2048),
            queries: Some(4096),
            epochs: None,
            shuffle: true,
            seed: 0,
        }
    }
}

/// Batches arriving from a worker thread, one ahead of the one being used.
///
/// Dropping the stream closes the channel, which is what tells the worker to
/// stop; it notices on its next send rather than mid-read, so a drop costs at
/// most one wasted batch.
pub struct BatchStream {
    receiver: std::sync::mpsc::Receiver<Result<Batch, NetworkError>>,
    worker: Option<std::thread::JoinHandle<()>>,
}

impl BatchStream {
    fn new(mut corpus: Corpus, config: BatchConfig) -> Result<Self, NetworkError> {
        if config.batch == 0 {
            return Err(NetworkError::InvalidConfig(
                "a batch needs at least one example".into(),
            ));
        }
        // A queue of one is the whole "decode one batch ahead" requirement:
        // the worker blocks on the second send until the first is taken.
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);

        let worker = std::thread::spawn(move || {
            let mut rng = StdRng::seed_from_u64(config.seed);
            let mut epoch = 0;
            while config.epochs.is_none_or(|epochs| epoch < epochs) {
                let mut order: Vec<usize> = (0..corpus.len()).collect();
                if config.shuffle {
                    let examples = order.len();
                    for slot in 0..examples {
                        order.swap(slot, rng.gen_range(slot..examples));
                    }
                }

                for chunk in order.chunks(config.batch) {
                    let batch = collect(&mut corpus, chunk, &config, &mut rng);
                    let failed = batch.is_err();
                    // A closed channel means the reader is gone, and a failed
                    // read means the corpus is broken; either way this is the
                    // last thing sent.
                    if sender.send(batch).is_err() || failed {
                        return;
                    }
                }
                epoch += 1;
            }
        });

        Ok(Self {
            receiver,
            worker: Some(worker),
        })
    }
}

impl Iterator for BatchStream {
    type Item = Result<Batch, NetworkError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.receiver.recv().ok()
    }
}

impl Drop for BatchStream {
    fn drop(&mut self) {
        // Take the channel down first, so a worker blocked on a full queue
        // wakes up instead of being waited on forever.
        let (_, receiver) = std::sync::mpsc::sync_channel(0);
        drop(std::mem::replace(&mut self.receiver, receiver));
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

/// Loads the examples of one batch and stacks them.
fn collect<R: Rng>(
    corpus: &mut Corpus,
    indices: &[usize],
    config: &BatchConfig,
    rng: &mut R,
) -> Result<Batch, NetworkError> {
    let mut names = Vec::with_capacity(indices.len());
    let mut loaded = Vec::with_capacity(indices.len());
    for index in indices {
        let example = corpus.example(*index, config.surface_points, config.queries, rng)?;
        names.push(example.name.clone());
        loaded.push(example);
    }

    // Every example in a batch has to be the same shape, or the packed layout
    // the models read cannot say where one sequence ends.
    let shape = |matrix: &Matrix| (matrix.rows, matrix.cols);
    for example in &loaded[1..] {
        for (name, (left, right)) in [
            ("tokens", (shape(&example.tokens), shape(&loaded[0].tokens))),
            (
                "surface",
                (shape(&example.surface), shape(&loaded[0].surface)),
            ),
            (
                "queries",
                (shape(&example.queries), shape(&loaded[0].queries)),
            ),
        ] {
            if left != right {
                return Err(NetworkError::InvalidDataset(format!(
                    "{}: {name} is {left:?} but {} has {right:?}",
                    example.name, loaded[0].name
                )));
            }
        }
    }

    // All or nothing: a packed batch cannot say which of its rows have a
    // colour target and which do not.
    let colors = loaded
        .iter()
        .map(|example| example.colors.as_ref())
        .collect::<Option<Vec<_>>>()
        .map(|parts| stack(parts.into_iter()));

    Ok(Batch {
        names,
        tokens: stack(loaded.iter().map(|example| &example.tokens)),
        surface: stack(loaded.iter().map(|example| &example.surface)),
        colors,
        queries: stack(loaded.iter().map(|example| &example.queries)),
        distances: loaded
            .iter()
            .flat_map(|example| example.distances.iter().copied())
            .collect(),
    })
}

/// Stacks matrices of equal width into one, rows in order.
fn stack<'a>(parts: impl Iterator<Item = &'a Matrix> + Clone) -> Matrix {
    let cols = parts.clone().next().map_or(0, |part| part.cols);
    let mut data = Vec::new();
    let mut rows = 0;
    for part in parts {
        data.extend_from_slice(&part.data);
        rows += part.rows;
    }
    Matrix::from_vec(rows, cols, data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::safetensors::{Dtype, write_as};

    /// Writes a corpus the way the two preprocessing binaries do: a token
    /// directory of one tensor per render, a sample directory of three per
    /// mesh, and an `index.json` on each side. Two shards on the sample side,
    /// so the open-shard cache is exercised rather than assumed.
    fn corpus(name: &str) -> (PathBuf, PathBuf) {
        let root = std::env::temp_dir().join(format!(
            "rusting_brain_shards_{name}_{}",
            std::process::id()
        ));
        let tokens = root.join("tokens");
        let samples = root.join("samples");
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&tokens).unwrap();
        std::fs::create_dir_all(&samples).unwrap();

        let meshes = ["cube", "sphere"];
        let renders = [
            ("cube_000", "cube"),
            ("cube_001", "cube"),
            ("sphere", "sphere"),
            ("orphan_000", ""),
        ];

        let mut token_index = BTreeMap::new();
        let held: Vec<(String, Vec<f32>)> = renders
            .iter()
            .enumerate()
            .map(|(number, (render, _))| {
                let data = (0..5 * 4)
                    .map(|value| (value + number * 100) as f32)
                    .collect();
                (render.to_string(), data)
            })
            .collect();
        let shard: Vec<(&str, &[usize], &[f32])> = held
            .iter()
            .map(|(name, data)| (name.as_str(), &[5usize, 4][..], &data[..]))
            .collect();
        write_as(
            tokens.join("tokens-00000.safetensors"),
            &shard,
            &BTreeMap::new(),
            Dtype::F32,
        )
        .unwrap();
        for (render, _) in renders {
            token_index.insert(render.to_string(), "tokens-00000.safetensors".to_string());
        }
        std::fs::write(
            tokens.join("index.json"),
            serde_json::to_vec(&token_index).unwrap(),
        )
        .unwrap();

        // One mesh per shard, so a batch holding both has to reopen.
        let mut sample_index = BTreeMap::new();
        for (number, mesh) in meshes.iter().enumerate() {
            let surface: Vec<f32> = (0..16 * 6).map(|value| value as f32).collect();
            // Row `q` is `(q, q, q, -q)`: the distance is the negated first
            // column, which makes the split checkable by eye.
            let queries: Vec<f32> = (0..32)
                .flat_map(|q| [q as f32, q as f32, q as f32, -(q as f32)])
                .collect();
            let transform = [0.5f32, -0.25, 0.75, 2.0];
            // Row `p` is `(p, 0, 1)`, and surface row `p` starts at `6p`, so a
            // colour still matched to its point is checkable by eye. Only the
            // first mesh gets colours, which is the mixed corpus the loader has
            // to cope with.
            let colors: Vec<f32> = (0..16).flat_map(|p| [p as f32, 0.0, 1.0]).collect();
            let file = format!("samples-{number:05}.safetensors");
            let (points, distances, placement, colored) = (
                format!("{mesh}.surface"),
                format!("{mesh}.queries"),
                format!("{mesh}.transform"),
                format!("{mesh}.colors"),
            );
            let mut tensors: Vec<(&str, &[usize], &[f32])> = vec![
                (&points, &[16, 6][..], &surface[..]),
                (&distances, &[32, 4][..], &queries[..]),
                (&placement, &[1, 4][..], &transform[..]),
            ];
            if number == 0 {
                tensors.push((&colored, &[16, 3][..], &colors[..]));
            }
            write_as(samples.join(&file), &tensors, &BTreeMap::new(), Dtype::F32).unwrap();
            sample_index.insert(mesh.to_string(), file);
        }
        std::fs::write(
            samples.join("index.json"),
            serde_json::to_vec(&sample_index).unwrap(),
        )
        .unwrap();

        (tokens, samples)
    }

    #[test]
    fn renders_pair_with_their_mesh_and_strays_are_skipped() {
        let (tokens, samples) = corpus("pairing");
        let corpus = Corpus::open(&tokens, &samples).unwrap();

        // `cube_000` and `cube_001` fall back to `cube`; `sphere` matches
        // outright; `orphan_000` has no mesh and is dropped.
        assert_eq!(corpus.len(), 3);
        let pairs: Vec<(&str, &str)> = corpus
            .pairs()
            .iter()
            .map(|(render, mesh)| (render.as_str(), mesh.as_str()))
            .collect();
        assert_eq!(
            pairs,
            vec![
                ("cube_000", "cube"),
                ("cube_001", "cube"),
                ("sphere", "sphere")
            ]
        );
    }

    #[test]
    fn a_missing_index_is_refused_by_path() {
        let error = match Shards::open(std::env::temp_dir().join("rusting_brain_no_such_corpus")) {
            Ok(_) => panic!("a directory with no index was opened"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("index.json"), "{error}");
    }

    /// A colour is only a target if it still belongs to its point, so the
    /// colours have to survive the surface's subsample row for row.
    #[test]
    fn colours_follow_the_surface_points_they_belong_to() {
        let (tokens, samples) = corpus("colors");
        let mut corpus = Corpus::open(&tokens, &samples).unwrap();
        let mut rng = StdRng::seed_from_u64(3);

        // `cube` was written with colours; `sphere` was not.
        let example = corpus.example(0, Some(5), Some(4), &mut rng).unwrap();
        let colors = example.colors.expect("cube carries colours");
        assert_eq!((colors.rows, colors.cols), (5, 3));
        for row in 0..colors.rows {
            // Surface row `p` starts at `6p` and colour row `p` starts at `p`.
            assert_eq!(colors.row(row)[0] * 6.0, example.surface.row(row)[0]);
            assert_eq!(colors.row(row)[2], 1.0);
        }

        let plain = corpus.example(2, Some(5), Some(4), &mut rng).unwrap();
        assert_eq!(plain.mesh, "sphere");
        assert!(plain.colors.is_none());

        // A batch that mixes the two carries no colours at all.
        let mixed = collect(
            &mut corpus,
            &[0, 2],
            &BatchConfig {
                batch: 2,
                surface_points: Some(5),
                queries: Some(4),
                ..BatchConfig::default()
            },
            &mut rng,
        )
        .unwrap();
        assert!(mixed.colors.is_none());

        let colored = collect(
            &mut corpus,
            &[0, 1],
            &BatchConfig {
                batch: 2,
                surface_points: Some(5),
                queries: Some(4),
                ..BatchConfig::default()
            },
            &mut rng,
        )
        .unwrap();
        let stacked = colored.colors.expect("both examples are the same mesh");
        assert_eq!((stacked.rows, stacked.cols), (10, 3));
    }

    #[test]
    fn an_example_subsamples_and_splits_its_queries() {
        let (tokens, samples) = corpus("example");
        let mut corpus = Corpus::open(&tokens, &samples).unwrap();
        let mut rng = StdRng::seed_from_u64(0);

        let example = corpus.example(0, Some(5), Some(7), &mut rng).unwrap();
        assert_eq!(example.name, "cube_000");
        assert_eq!(example.mesh, "cube");
        assert_eq!((example.tokens.rows, example.tokens.cols), (5, 4));
        assert_eq!((example.surface.rows, example.surface.cols), (5, 6));
        assert_eq!((example.queries.rows, example.queries.cols), (7, 3));
        assert_eq!(example.distances.len(), 7);

        // Every row kept its own distance through the subsample and the split.
        for row in 0..example.queries.rows {
            let point = example.queries.row(row);
            assert_eq!(point, [point[0]; 3]);
            assert_eq!(example.distances[row], -point[0]);
        }

        // Asking for more than there is takes everything, once.
        let whole = corpus.example(0, None, Some(1_000), &mut rng).unwrap();
        assert_eq!(whole.surface.rows, 16);
        assert_eq!(whole.queries.rows, 32);
        let mut seen: Vec<f32> = whole.distances.clone();
        seen.sort_by(f32::total_cmp);
        seen.dedup();
        assert_eq!(seen.len(), 32);
    }

    #[test]
    fn the_transform_comes_back_as_the_preprocessing_wrote_it() {
        let (tokens, samples) = corpus("transform");
        let mut corpus = Corpus::open(&tokens, &samples).unwrap();
        assert_eq!(corpus.transform("sphere").unwrap(), [0.5, -0.25, 0.75, 2.0]);
        assert!(corpus.transform("nothing").is_err());
    }

    #[test]
    fn a_batch_stacks_its_examples_in_order() {
        let (tokens, samples) = corpus("batch");
        let corpus = Corpus::open(&tokens, &samples).unwrap();
        let config = BatchConfig {
            batch: 3,
            surface_points: Some(4),
            queries: Some(6),
            epochs: Some(1),
            shuffle: false,
            seed: 1,
        };

        let batches: Vec<Batch> = corpus.stream(config).unwrap().map(Result::unwrap).collect();
        assert_eq!(batches.len(), 1);
        let batch = &batches[0];

        assert_eq!(batch.len(), 3);
        assert_eq!(batch.names, vec!["cube_000", "cube_001", "sphere"]);
        assert_eq!((batch.tokens.rows, batch.tokens.cols), (15, 4));
        assert_eq!((batch.surface.rows, batch.surface.cols), (12, 6));
        assert_eq!((batch.queries.rows, batch.queries.cols), (18, 3));
        assert_eq!(batch.distances.len(), 18);

        // The second example's tokens are the second render's, unshuffled and
        // in place: the packed layout is what the models index into.
        assert_eq!(batch.tokens.row(5)[0], 100.0);
        assert_eq!(batch.tokens.row(10)[0], 200.0);
    }

    #[test]
    fn a_short_last_batch_still_arrives() {
        let (tokens, samples) = corpus("partial");
        let corpus = Corpus::open(&tokens, &samples).unwrap();
        let sizes: Vec<usize> = corpus
            .stream(BatchConfig {
                batch: 2,
                epochs: Some(1),
                shuffle: false,
                surface_points: Some(2),
                queries: Some(2),
                seed: 0,
            })
            .unwrap()
            .map(|batch| batch.unwrap().len())
            .collect();
        assert_eq!(sizes, vec![2, 1]);
    }

    #[test]
    fn a_stream_with_no_epoch_limit_keeps_going_and_shuffles() {
        let (tokens, samples) = corpus("endless");
        let corpus = Corpus::open(&tokens, &samples).unwrap();
        let mut stream = corpus
            .stream(BatchConfig {
                batch: 1,
                epochs: None,
                shuffle: true,
                surface_points: Some(2),
                queries: Some(2),
                seed: 7,
            })
            .unwrap();

        // Well past one pass over three examples.
        let names: Vec<String> = (0..12)
            .map(|_| stream.next().unwrap().unwrap().names[0].clone())
            .collect();
        assert_eq!(names.len(), 12);
        // Shuffled, so the order is not the index order repeated.
        let ordered: Vec<String> = names[..3].to_vec();
        assert!(
            names.chunks(3).any(|chunk| chunk.to_vec() != ordered),
            "every pass came out in the same order: {names:?}"
        );

        // Dropping the stream mid-flight has to stop the worker rather than
        // hang on a full queue.
        drop(stream);
    }

    #[test]
    fn a_batch_of_nothing_is_refused() {
        let (tokens, samples) = corpus("empty");
        let corpus = Corpus::open(&tokens, &samples).unwrap();
        assert!(
            corpus
                .stream(BatchConfig {
                    batch: 0,
                    ..BatchConfig::default()
                })
                .is_err()
        );
    }
}
