//! In-memory tabular data: rows of inputs paired with rows of targets.
//!
//! This is the input to [`Network::fit`](crate::Network::fit). Token sequences
//! for a language model go through [`TokenBatch`](crate::TokenBatch) instead.

use crate::network::NetworkError;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use std::path::Path;

#[derive(Clone, Debug, PartialEq)]
pub struct Dataset {
    pub inputs: Vec<Vec<f32>>,
    pub targets: Vec<Vec<f32>>,
}

impl Dataset {
    pub fn new(inputs: Vec<Vec<f32>>, targets: Vec<Vec<f32>>) -> Self {
        assert_eq!(inputs.len(), targets.len());
        Self { inputs, targets }
    }

    /// Reads a comma-separated file in which the last `target_columns` columns
    /// are the target and every column before them is an input.
    ///
    /// A first row that does not parse as numbers is taken to be a header and
    /// skipped. Quoted fields are understood, including `""` for a literal
    /// quote.
    ///
    /// ```no_run
    /// # use rusting_brain::Dataset;
    /// // sepal_length,sepal_width,petal_length,is_setosa
    /// let dataset = Dataset::from_csv("iris.csv", 1)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn from_csv(path: impl AsRef<Path>, target_columns: usize) -> Result<Self, NetworkError> {
        Self::from_csv_str(&std::fs::read_to_string(path)?, target_columns)
    }

    fn from_csv_str(text: &str, target_columns: usize) -> Result<Self, NetworkError> {
        if target_columns == 0 {
            return Err(NetworkError::InvalidDataset(
                "a row needs at least one target column".into(),
            ));
        }

        let mut inputs = Vec::new();
        let mut targets = Vec::new();
        let mut width = None;

        for (index, line) in text.lines().enumerate() {
            let line = line.trim_end_matches('\r');
            if line.trim().is_empty() {
                continue;
            }

            let fields = split_csv_row(line);
            let row: Result<Vec<f32>, _> =
                fields.iter().map(|field| field.trim().parse()).collect();

            let Ok(row) = row else {
                // A header is only a header if it comes before any data.
                if inputs.is_empty() && width.is_none() {
                    width = Some(fields.len());
                    continue;
                }
                return Err(NetworkError::InvalidDataset(format!(
                    "line {} is not numeric: {line}",
                    index + 1
                )));
            };

            let expected = *width.get_or_insert(row.len());
            if row.len() != expected {
                return Err(NetworkError::InvalidDataset(format!(
                    "line {} has {} columns, expected {expected}",
                    index + 1,
                    row.len()
                )));
            }
            if row.len() <= target_columns {
                return Err(NetworkError::InvalidDataset(format!(
                    "line {} has {} columns, too few for {target_columns} target columns \
                     and at least one input",
                    index + 1,
                    row.len()
                )));
            }

            let (input, target) = row.split_at(row.len() - target_columns);
            inputs.push(input.to_vec());
            targets.push(target.to_vec());
        }

        if inputs.is_empty() {
            return Err(NetworkError::EmptyDataset);
        }

        Ok(Self::new(inputs, targets))
    }

    /// Reads a comma-separated file whose last column is a class name rather
    /// than a number.
    ///
    /// This is what a classification dataset off the internet actually looks
    /// like — `5.1,3.5,1.4,0.2,setosa` — and [`Dataset::from_csv`] rejects it,
    /// because the last column does not parse as a float. The distinct labels
    /// sorted become the classes, the target is a one-hot row over them, and
    /// the names come back so a prediction can be named.
    ///
    /// ```
    /// # use rusting_brain::Dataset;
    /// # let path = std::env::temp_dir().join("rusting_brain_doc_iris.csv");
    /// # std::fs::write(&path,
    /// #     "length,width,species\n5.1,3.5,setosa\n6.4,3.2,versicolor\n").unwrap();
    /// let (dataset, classes) = Dataset::from_csv_labeled(&path)?;
    /// assert_eq!(classes, ["setosa", "versicolor"]);
    /// assert_eq!(dataset.targets[0], vec![1.0, 0.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn from_csv_labeled(path: impl AsRef<Path>) -> Result<(Self, Vec<String>), NetworkError> {
        Self::from_csv_labeled_str(&std::fs::read_to_string(path)?)
    }

    fn from_csv_labeled_str(text: &str) -> Result<(Self, Vec<String>), NetworkError> {
        let mut inputs = Vec::new();
        let mut labels: Vec<String> = Vec::new();
        let mut width = None;

        for (index, line) in text.lines().enumerate() {
            let line = line.trim_end_matches('\r');
            if line.trim().is_empty() {
                continue;
            }

            let fields = split_csv_row(line);
            let Some((label, values)) = fields.split_last() else {
                continue;
            };
            if values.is_empty() {
                return Err(NetworkError::InvalidDataset(format!(
                    "line {} is a label with no input columns",
                    index + 1
                )));
            }

            let row: Result<Vec<f32>, _> =
                values.iter().map(|field| field.trim().parse()).collect();
            let Ok(row) = row else {
                // A header is only a header if it comes before any data. The
                // last column is a name on every line, so the numeric test runs
                // on the input columns alone.
                if inputs.is_empty() && width.is_none() {
                    width = Some(values.len());
                    continue;
                }
                return Err(NetworkError::InvalidDataset(format!(
                    "line {} is not numeric: {line}",
                    index + 1
                )));
            };

            let expected = *width.get_or_insert(row.len());
            if row.len() != expected {
                return Err(NetworkError::InvalidDataset(format!(
                    "line {} has {} input columns, expected {expected}",
                    index + 1,
                    row.len()
                )));
            }

            inputs.push(row);
            labels.push(label.trim().to_string());
        }

        if inputs.is_empty() {
            return Err(NetworkError::EmptyDataset);
        }

        let mut classes = labels.clone();
        classes.sort_unstable();
        classes.dedup();

        let targets = labels
            .iter()
            .map(|label| {
                let mut target = vec![0.0; classes.len()];
                target[classes.binary_search(label).expect("label is a class")] = 1.0;
                target
            })
            .collect();

        Ok((Self::new(inputs, targets), classes))
    }

    /// Reads a directory of directories: one subdirectory per class, its images
    /// inside it.
    ///
    /// Every image is resized to `width` by `height` and flattened into one
    /// input row of `width * height` values (grayscale) or three times that
    /// (RGB, red plane then green then blue), each scaled to `0.0..=1.0`. The
    /// target is a one-hot row over the classes, which are the subdirectory
    /// names sorted, and those names come back alongside the dataset so a
    /// prediction can be named.
    ///
    /// Files that are not `.png`, `.jpg` or `.jpeg` are skipped; a file that
    /// has one of those extensions and does not decode is an error rather than
    /// a silent omission.
    ///
    /// ```no_run
    /// # use rusting_brain::Dataset;
    /// // images/cat/*.png, images/dog/*.png
    /// let (dataset, classes) = Dataset::from_image_folder("images", 32, 32, true)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// NOTE: decoding happens once, into memory, which is what
    /// [`Network::fit`](crate::Network::fit) wants and what caps this at a
    /// dataset that fits in RAM. Augmentation and streaming belong in a
    /// loader that yields batches, not here.
    #[cfg(feature = "images")]
    pub fn from_image_folder(
        root: impl AsRef<Path>,
        width: u32,
        height: u32,
        grayscale: bool,
    ) -> Result<(Self, Vec<String>), NetworkError> {
        use rayon::prelude::*;

        let root = root.as_ref();
        let mut classes: Vec<String> = std::fs::read_dir(root)?
            .filter_map(Result::ok)
            .filter(|entry| entry.path().is_dir())
            .map(|entry| entry.file_name().to_string_lossy().into_owned())
            .collect();
        if classes.is_empty() {
            return Err(NetworkError::InvalidDataset(format!(
                "{} holds no class subdirectories",
                root.display()
            )));
        }
        // Sorted, so the class an index means does not depend on the order the
        // filesystem happened to hand the directories back.
        classes.sort();

        let mut paths = Vec::new();
        for (label, class) in classes.iter().enumerate() {
            for entry in std::fs::read_dir(root.join(class))?.filter_map(Result::ok) {
                let path = entry.path();
                let decodable = path
                    .extension()
                    .and_then(|extension| extension.to_str())
                    .is_some_and(|extension| {
                        matches!(
                            extension.to_ascii_lowercase().as_str(),
                            "png" | "jpg" | "jpeg"
                        )
                    });
                if decodable {
                    paths.push((label, path));
                }
            }
        }
        if paths.is_empty() {
            return Err(NetworkError::EmptyDataset);
        }
        paths.sort();

        let inputs = paths
            .par_iter()
            .map(|(_, path)| {
                let image = image::open(path).map_err(|error| {
                    NetworkError::InvalidDataset(format!("{}: {error}", path.display()))
                })?;
                let image =
                    image.resize_exact(width, height, image::imageops::FilterType::Triangle);
                Ok(if grayscale {
                    image
                        .to_luma8()
                        .iter()
                        .map(|&pixel| f32::from(pixel) / 255.0)
                        .collect()
                } else {
                    // Plane by plane rather than interleaved: a dense network
                    // does not care, and it is the layout a convolution would
                    // want if one ever reads this.
                    let rgb = image.to_rgb8();
                    (0..3)
                        .flat_map(|channel| {
                            rgb.pixels()
                                .map(move |pixel| f32::from(pixel[channel]) / 255.0)
                        })
                        .collect()
                })
            })
            .collect::<Result<Vec<Vec<f32>>, NetworkError>>()?;

        let targets = paths
            .iter()
            .map(|&(label, _)| {
                let mut one_hot = vec![0.0; classes.len()];
                one_hot[label] = 1.0;
                one_hot
            })
            .collect();

        Ok((Self::new(inputs, targets), classes))
    }

    /// Reads two NumPy `.npy` files, one of inputs and one of targets.
    ///
    /// This is the file `numpy.save` writes, which is how data leaves PyTorch,
    /// TensorFlow and scikit-learn without going through a CSV twenty times its
    /// size. A 2-D array is one row per sample; a 1-D array is one value per
    /// sample, which is what a column of labels looks like. `float32`,
    /// `float64`, `int32`, `int64` and `uint8` all arrive as `f32`.
    ///
    /// ```no_run
    /// # use rusting_brain::Dataset;
    /// // numpy.save("x.npy", x); numpy.save("y.npy", y)
    /// let mut dataset = Dataset::from_npy("x.npy", "y.npy")?;
    /// dataset.one_hot_targets(10)?;      // y held class indices, not one-hot rows
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    /// Reads a whole JSONL file into memory, one row per line.
    ///
    /// Each line is an object holding `input_field` and `target_field`, each
    /// one an array of numbers or a single number. [`JsonlStream`] reads the
    /// same format a batch at a time, for a file that does not fit in memory.
    ///
    /// ```no_run
    /// # use rusting_brain::Dataset;
    /// let dataset = Dataset::from_jsonl("corpus.jsonl", "pixels", "label")?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn from_jsonl(
        path: impl AsRef<Path>,
        input_field: &str,
        target_field: &str,
    ) -> Result<Self, NetworkError> {
        let mut stream = JsonlStream::open(path, input_field, target_field, 1024)?;
        let mut dataset = Self::new(Vec::new(), Vec::new());
        while let Some(batch) = stream.next_batch()? {
            dataset.inputs.extend_from_slice(batch.inputs);
            dataset.targets.extend_from_slice(batch.targets);
        }

        if dataset.is_empty() {
            return Err(NetworkError::EmptyDataset);
        }
        Ok(dataset)
    }

    pub fn from_npy(
        inputs: impl AsRef<Path>,
        targets: impl AsRef<Path>,
    ) -> Result<Self, NetworkError> {
        let inputs = read_npy(inputs.as_ref())?;
        let targets = read_npy(targets.as_ref())?;

        if inputs.len() != targets.len() {
            return Err(NetworkError::InvalidDataset(format!(
                "{} input rows and {} target rows",
                inputs.len(),
                targets.len()
            )));
        }
        if inputs.is_empty() {
            return Err(NetworkError::EmptyDataset);
        }
        Ok(Self::new(inputs, targets))
    }

    /// Reads a pair of IDX files, the format MNIST and its many descendants
    /// ship in.
    ///
    /// `images` is the `idx3-ubyte` file and `labels` the `idx1-ubyte` one. An
    /// image of any shape is flattened into one input row, and `u8` pixels are
    /// scaled to `0.0..=1.0`; labels are left as the class indices they are, so
    /// [`one_hot_targets`](Dataset::one_hot_targets) is the next call.
    ///
    /// The files the archives ship are gzipped and this reads the plain ones:
    /// `gunzip train-images-idx3-ubyte.gz` first.
    ///
    /// ```no_run
    /// # use rusting_brain::Dataset;
    /// let mut train = Dataset::from_idx("train-images-idx3-ubyte", "train-labels-idx1-ubyte")?;
    /// train.one_hot_targets(10)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn from_idx(
        images: impl AsRef<Path>,
        labels: impl AsRef<Path>,
    ) -> Result<Self, NetworkError> {
        let inputs = read_idx(images.as_ref(), true)?;
        let targets = read_idx(labels.as_ref(), false)?;

        if inputs.len() != targets.len() {
            return Err(NetworkError::InvalidDataset(format!(
                "{} images and {} labels",
                inputs.len(),
                targets.len()
            )));
        }
        if inputs.is_empty() {
            return Err(NetworkError::EmptyDataset);
        }
        Ok(Self::new(inputs, targets))
    }

    /// Turns a single target column of class indices into one-hot rows over
    /// `classes` classes.
    ///
    /// Labels arrive as indices far more often than as one-hot rows — from
    /// `.npy`, from a CSV of numbers — and a network with a softmax output
    /// needs the row.
    pub fn one_hot_targets(&mut self, classes: usize) -> Result<(), NetworkError> {
        if classes == 0 {
            return Err(NetworkError::InvalidDataset(
                "one-hot targets need at least one class".into(),
            ));
        }
        for (row, target) in self.targets.iter_mut().enumerate() {
            let [index] = target[..] else {
                return Err(NetworkError::InvalidDataset(format!(
                    "row {row} has {} target columns, and only a single column of class \
                     indices can be made one-hot",
                    target.len()
                )));
            };
            if index < 0.0 || index.fract() != 0.0 || index as usize >= classes {
                return Err(NetworkError::InvalidDataset(format!(
                    "row {row} holds {index}, which is not a class index below {classes}"
                )));
            }
            *target = vec![0.0; classes];
            target[index as usize] = 1.0;
        }
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.inputs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inputs.is_empty()
    }

    pub fn shuffle(&mut self, seed: Option<u64>) {
        let mut indexes: Vec<usize> = (0..self.len()).collect();

        if let Some(seed) = seed {
            let mut rng = StdRng::seed_from_u64(seed);
            indexes.shuffle(&mut rng);
        } else {
            let mut rng = rand::thread_rng();
            indexes.shuffle(&mut rng);
        }

        // Moves the rows; each index occurs once, so none is taken twice.
        let mut inputs = std::mem::take(&mut self.inputs);
        let mut targets = std::mem::take(&mut self.targets);
        self.inputs = indexes
            .iter()
            .map(|&i| std::mem::take(&mut inputs[i]))
            .collect();
        self.targets = indexes
            .iter()
            .map(|&i| std::mem::take(&mut targets[i]))
            .collect();
    }

    /// [`split`](Dataset::split), keeping each class's share of the rows on
    /// both sides.
    ///
    /// A plain split of an imbalanced set can leave a rare class out of the
    /// test half entirely, and the accuracy it reports then says nothing about
    /// that class. Classes are read off the targets: the largest entry of a
    /// one-hot row, or the value itself when the target is a single column.
    ///
    /// Row order inside a class is kept, so shuffle first — the split is
    /// otherwise the first rows of each class, which is whatever order the
    /// loader produced.
    ///
    /// ```
    /// # use rusting_brain::Dataset;
    /// let mut dataset = Dataset::new(vec![vec![0.0]; 10], vec![vec![1.0, 0.0]; 10]);
    /// dataset.targets[9] = vec![0.0, 1.0];             // one row of a rare class
    /// let (train, test) = dataset.split_stratified(0.8);
    /// assert_eq!((train.len(), test.len()), (8, 2));
    /// ```
    pub fn split_stratified(&self, train_ratio: f32) -> (Self, Self) {
        assert!((0.0..=1.0).contains(&train_ratio));

        let class_of = |target: &Vec<f32>| -> u32 {
            match target[..] {
                [value] => value.to_bits(),
                _ => target
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map_or(0, |(index, _)| index) as u32,
            }
        };

        // NOTE: one pass per class over the rows. Classes are tens, not
        // thousands, so this is cheaper than sorting and keeps the order.
        let mut classes: Vec<u32> = self.targets.iter().map(class_of).collect();
        let rows: Vec<u32> = classes.clone();
        classes.sort_unstable();
        classes.dedup();

        let (mut train, mut test) = (Vec::new(), Vec::new());
        for class in classes {
            let members: Vec<usize> = (0..self.len()).filter(|&row| rows[row] == class).collect();
            let take = ((members.len() as f32) * train_ratio).round() as usize;
            train.extend(&members[..take]);
            test.extend(&members[take..]);
        }
        // Back into the dataset's own order, so a stratified split of an
        // already-shuffled dataset is still shuffled.
        train.sort_unstable();
        test.sort_unstable();

        let gather = |rows: &[usize]| {
            Self::new(
                rows.iter().map(|&row| self.inputs[row].clone()).collect(),
                rows.iter().map(|&row| self.targets[row].clone()).collect(),
            )
        };
        (gather(&train), gather(&test))
    }

    pub fn split(&self, train_ratio: f32) -> (Self, Self) {
        assert!((0.0..=1.0).contains(&train_ratio));
        let train_len = ((self.len() as f32) * train_ratio).round() as usize;

        (
            Self::new(
                self.inputs[..train_len].to_vec(),
                self.targets[..train_len].to_vec(),
            ),
            Self::new(
                self.inputs[train_len..].to_vec(),
                self.targets[train_len..].to_vec(),
            ),
        )
    }

    /// Centers every input column on zero and scales it to unit deviation,
    /// returning the statistics it used.
    ///
    /// A network fed raw columns — an age beside an income — spends its first
    /// epochs undoing the scale difference, and with a saturating activation it
    /// may never recover. Fit on the training split and apply those statistics
    /// to the test split: fitting again on the test split leaks it.
    ///
    /// ```
    /// # use rusting_brain::Dataset;
    /// let mut train = Dataset::new(vec![vec![1.0, 900.0], vec![3.0, 1100.0]], vec![vec![0.0]; 2]);
    /// let mut test = Dataset::new(vec![vec![2.0, 1000.0]], vec![vec![1.0]]);
    ///
    /// let statistics = train.standardize();
    /// statistics.apply(&mut test);
    /// assert_eq!(test.inputs[0], vec![0.0, 0.0]);
    /// ```
    pub fn standardize(&mut self) -> Standardizer {
        let columns = self.inputs.first().map_or(0, Vec::len);
        let rows = self.len().max(1) as f32;

        let mut mean = vec![0.0; columns];
        for row in &self.inputs {
            for (sum, value) in mean.iter_mut().zip(row) {
                *sum += value / rows;
            }
        }

        let mut deviation = vec![0.0f32; columns];
        for row in &self.inputs {
            for ((sum, value), mean) in deviation.iter_mut().zip(row).zip(&mean) {
                *sum += (value - mean).powi(2) / rows;
            }
        }
        for value in &mut deviation {
            // A constant column has no deviation to divide by. Leaving the
            // scale at one turns it into a column of zeros, which is what it
            // carries anyway, rather than a column of NaN.
            *value = value.sqrt();
            if *value < 1e-8 {
                *value = 1.0;
            }
        }

        let statistics = Standardizer { mean, deviation };
        statistics.apply(self);
        statistics
    }

    /// Doubles the dataset with a left-right mirror of every input row.
    ///
    /// A flipped photograph of a cat is still a photograph of a cat, so the
    /// mirrored copy keeps its target and the model sees twice the data for
    /// the price of one decode. Rows are the flat layout
    /// [`from_image_folder`](Dataset::from_image_folder) produces — plane after
    /// plane, each plane row-major of `width` pixels — so an RGB image flips
    /// correctly with the same `width` as a grayscale one.
    ///
    /// Apply it to the training split only. Augmenting before
    /// [`split`](Dataset::split) puts an image and its mirror on opposite
    /// sides of the split and the test accuracy stops meaning anything.
    ///
    /// ```
    /// # use rusting_brain::Dataset;
    /// let mut dataset = Dataset::new(vec![vec![1.0, 2.0, 3.0, 4.0]], vec![vec![1.0]]);
    /// dataset.flip_horizontal(2)?;                      // two 2x1 rows
    /// assert_eq!(dataset.inputs[1], vec![2.0, 1.0, 4.0, 3.0]);
    /// assert_eq!(dataset.targets[1], vec![1.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// This doubles the dataset once, up front. Random crops and rotations,
    /// which want a fresh draw every epoch, are [`Augment`].
    pub fn flip_horizontal(&mut self, width: usize) -> Result<(), NetworkError> {
        if width == 0 {
            return Err(NetworkError::InvalidDataset(
                "an image width of zero has nothing to flip".into(),
            ));
        }
        let flipped: Vec<Vec<f32>> = self
            .inputs
            .iter()
            .map(|row| {
                if row.len() % width != 0 {
                    return Err(NetworkError::InvalidDataset(format!(
                        "a row of {} values is not a whole number of {width}-pixel image rows",
                        row.len()
                    )));
                }
                let mut flipped = row.clone();
                for line in flipped.chunks_mut(width) {
                    line.reverse();
                }
                Ok(flipped)
            })
            .collect::<Result<_, NetworkError>>()?;

        self.inputs.extend(flipped);
        self.targets.extend_from_within(..);
        Ok(())
    }

    /// Reads this dataset as a [`BatchSource`], which is what
    /// [`Network::fit`](crate::Network::fit) trains from.
    ///
    /// The stream takes the batch size, the shuffling and the seed from
    /// `config`, and shuffles the dataset in place at the start of every epoch,
    /// which is why it borrows mutably.
    ///
    /// ```
    /// # use rusting_brain::{BatchSource, Dataset, TrainConfig};
    /// let mut dataset = Dataset::new(vec![vec![0.0]; 5], vec![vec![1.0]; 5]);
    /// let config = TrainConfig { batch_size: 2, shuffle: false, ..Default::default() };
    /// let mut stream = dataset.stream(&config);
    ///
    /// stream.start_epoch(0)?;
    /// let mut sizes = Vec::new();
    /// while let Some(batch) = stream.next_batch()? {
    ///     sizes.push(batch.inputs.len());
    /// }
    /// assert_eq!(sizes, [2, 2, 1]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn stream(&mut self, config: &crate::TrainConfig) -> DatasetStream<'_> {
        DatasetStream {
            order: (0..self.len()).collect(),
            dataset: self,
            batch_size: config.batch_size.max(1),
            shuffle: config.shuffle,
            seed: config.seed,
            cursor: 0,
        }
    }

    pub fn batches(&self, batch_size: usize) -> Vec<DatasetBatch<'_>> {
        assert!(batch_size > 0);

        self.inputs
            .chunks(batch_size)
            .zip(self.targets.chunks(batch_size))
            .map(|(inputs, targets)| DatasetBatch { inputs, targets })
            .collect()
    }
}

/// Reads a `.npy` file into rows of `f32`.
///
/// NOTE: the header is a Python dict literal and this pulls the three keys
/// it needs out of it by string search rather than parsing it. That is enough
/// for what `numpy.save` writes, which is the only thing that writes this
/// format; a hand-built header with different spacing would need a parser.
fn read_npy(path: &Path) -> Result<Vec<Vec<f32>>, NetworkError> {
    let bytes = std::fs::read(path)?;
    let name = path.display();
    let invalid = |message: String| NetworkError::InvalidDataset(format!("{name}: {message}"));

    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err(invalid("not a .npy file".into()));
    }
    // Version 1 stores the header length as u16, versions 2 and 3 as u32.
    let (header_length, start) = match (bytes[6], bytes.get(8..12)) {
        (1, _) => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        (_, Some(length)) => (u32::from_le_bytes(length.try_into().unwrap()) as usize, 12),
        (_, None) => return Err(invalid("truncated header".into())),
    };
    let Some(header) = bytes
        .get(start..start + header_length)
        .and_then(|header| std::str::from_utf8(header).ok())
    else {
        return Err(invalid("truncated header".into()));
    };

    if header.contains("'fortran_order': True") {
        return Err(invalid(
            "column-major order, which numpy.save does not write and this does not read".into(),
        ));
    }

    let quoted = |key: &str| -> Option<String> {
        let rest = header.split_once(key)?.1;
        let rest = rest.trim_start().trim_start_matches(':').trim_start();
        let inner = rest.strip_prefix('\'')?;
        Some(inner.split('\'').next()?.to_string())
    };
    let descr = quoted("'descr'").ok_or_else(|| invalid("no dtype in the header".into()))?;

    let shape = header
        .split_once("'shape'")
        .and_then(|(_, rest)| rest.split_once('('))
        .and_then(|(_, rest)| rest.split_once(')'))
        .ok_or_else(|| invalid("no shape in the header".into()))?
        .0;
    let shape: Vec<usize> = shape
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| part.parse().map_err(|_| invalid(format!("shape {shape}"))))
        .collect::<Result<_, _>>()?;

    let (rows, columns) = match shape[..] {
        [rows] => (rows, 1),
        [rows, columns] => (rows, columns),
        _ => {
            return Err(invalid(format!(
                "{} dimensions, and a dataset row is one or two",
                shape.len()
            )));
        }
    };

    if rows == 0 || columns == 0 {
        return Err(invalid(format!("an empty {rows}x{columns} array")));
    }

    let data = &bytes[start + header_length..];
    let values: Vec<f32> = match descr.as_str() {
        "<f4" | "=f4" | "|f4" => data
            .chunks_exact(4)
            .map(|v| f32::from_le_bytes([v[0], v[1], v[2], v[3]]))
            .collect(),
        "<f8" | "=f8" => data
            .chunks_exact(8)
            .map(|v| f64::from_le_bytes([v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]]) as f32)
            .collect(),
        "<i4" | "=i4" => data
            .chunks_exact(4)
            .map(|v| i32::from_le_bytes([v[0], v[1], v[2], v[3]]) as f32)
            .collect(),
        "<i8" | "=i8" => data
            .chunks_exact(8)
            .map(|v| i64::from_le_bytes([v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]]) as f32)
            .collect(),
        "|u1" | "u1" => data.iter().map(|&value| f32::from(value)).collect(),
        other => {
            return Err(invalid(format!(
                "dtype {other}, and this reads float32, float64, int32, int64 and uint8"
            )));
        }
    };

    if rows
        .checked_mul(columns)
        .is_none_or(|count| values.len() < count)
    {
        return Err(invalid(format!(
            "header promises {rows}x{columns} values and the file holds {}",
            values.len()
        )));
    }

    Ok(values
        .chunks_exact(columns)
        .take(rows)
        .map(<[f32]>::to_vec)
        .collect())
}

/// Reads one IDX file into rows: dimension zero is the sample, every remaining
/// dimension is flattened into the row. `scale` divides `u8` data by 255, which
/// is right for pixels and wrong for labels.
///
/// NOTE: the whole file is read into memory, like `read_npy` above. MNIST
/// is 45 MB and the rest of the crate wants the dataset resident anyway.
fn read_idx(path: &Path, scale: bool) -> Result<Vec<Vec<f32>>, NetworkError> {
    let bytes = std::fs::read(path)?;
    let name = path.display();
    let invalid = |message: String| NetworkError::InvalidDataset(format!("{name}: {message}"));

    if bytes.starts_with(&[0x1f, 0x8b]) {
        return Err(invalid(
            "gzipped, and this reads the plain file: gunzip it first".into(),
        ));
    }
    // Two zero bytes, a type code, then the number of dimensions.
    if bytes.len() < 4 || bytes[0] != 0 || bytes[1] != 0 {
        return Err(invalid("not an IDX file".into()));
    }
    let dimensions = bytes[3] as usize;
    let header = 4 + dimensions * 4;
    if dimensions == 0 || bytes.len() < header {
        return Err(invalid(format!("a {dimensions}-dimensional header")));
    }

    let shape: Vec<usize> = bytes[4..header]
        .chunks_exact(4)
        .map(|v| u32::from_be_bytes([v[0], v[1], v[2], v[3]]) as usize)
        .collect();
    // A corrupt shape can overflow; the file can never hold that many values.
    if shape
        .iter()
        .try_fold(1usize, |total, dimension| total.checked_mul(*dimension))
        .is_none_or(|total| total > bytes.len())
    {
        return Err(invalid(format!("shape {shape:?} is larger than the file")));
    }
    let rows = shape[0];
    let columns: usize = shape[1..].iter().product::<usize>().max(1);
    if rows == 0 {
        return Err(invalid("no samples".into()));
    }

    // IDX is big-endian throughout, which is the one thing it does not share
    // with .npy.
    let data = &bytes[header..];
    let values: Vec<f32> = match bytes[2] {
        0x08 => data.iter().map(|&value| f32::from(value)).collect(),
        0x09 => data.iter().map(|&value| f32::from(value as i8)).collect(),
        0x0b => data
            .chunks_exact(2)
            .map(|v| f32::from(i16::from_be_bytes([v[0], v[1]])))
            .collect(),
        0x0c => data
            .chunks_exact(4)
            .map(|v| i32::from_be_bytes([v[0], v[1], v[2], v[3]]) as f32)
            .collect(),
        0x0d => data
            .chunks_exact(4)
            .map(|v| f32::from_be_bytes([v[0], v[1], v[2], v[3]]))
            .collect(),
        0x0e => data
            .chunks_exact(8)
            .map(|v| f64::from_be_bytes([v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]]) as f32)
            .collect(),
        other => return Err(invalid(format!("type code {other:#04x}"))),
    };

    if values.len() < rows * columns {
        return Err(invalid(format!(
            "header promises {rows}x{columns} values and the file holds {}",
            values.len()
        )));
    }

    let divisor = if scale && bytes[2] == 0x08 {
        255.0
    } else {
        1.0
    };
    Ok(values
        .chunks_exact(columns)
        .take(rows)
        .map(|row| row.iter().map(|value| value / divisor).collect())
        .collect())
}

/// The per-column mean and deviation [`Dataset::standardize`] measured.
///
/// Save it with the model: a network trained on standardized columns predicts
/// nonsense from a raw row, and the statistics are the only way to put a later
/// row on the same scale. It is `serde`-serializable for that reason.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Standardizer {
    mean: Vec<f32>,
    deviation: Vec<f32>,
}

impl Standardizer {
    /// Applies the statistics to every input row of `dataset`.
    pub fn apply(&self, dataset: &mut Dataset) {
        for row in &mut dataset.inputs {
            self.apply_row(row);
        }
    }

    /// Applies them to one row, which is what a prediction needs.
    ///
    /// ```
    /// # use rusting_brain::Dataset;
    /// # let mut train = Dataset::new(vec![vec![1.0], vec![3.0]], vec![vec![0.0]; 2]);
    /// # let statistics = train.standardize();
    /// let mut row = vec![2.0];
    /// statistics.apply_row(&mut row);
    /// // model.predict(&row)?;
    /// ```
    pub fn apply_row(&self, row: &mut [f32]) {
        for ((value, mean), deviation) in row.iter_mut().zip(&self.mean).zip(&self.deviation) {
            *value = (*value - mean) / deviation;
        }
    }

    pub fn mean(&self) -> &[f32] {
        &self.mean
    }

    pub fn deviation(&self) -> &[f32] {
        &self.deviation
    }
}

/// NOTE: single-line rows only. A quoted field containing a newline needs
/// a reader that owns the line splitting; add one if a real corpus needs it.
fn split_csv_row(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut field = String::new();
    let mut quoted = false;
    let mut characters = line.chars().peekable();

    while let Some(character) = characters.next() {
        match character {
            '"' if quoted => {
                if characters.peek() == Some(&'"') {
                    characters.next();
                    field.push('"');
                } else {
                    quoted = false;
                }
            }
            '"' if field.is_empty() => quoted = true,
            ',' if !quoted => fields.push(std::mem::take(&mut field)),
            _ => field.push(character),
        }
    }

    fields.push(field);
    fields
}

#[derive(Clone, Copy, Debug)]
pub struct DatasetBatch<'a> {
    pub inputs: &'a [Vec<f32>],
    pub targets: &'a [Vec<f32>],
}

/// A source of training batches.
///
/// [`Network::fit`](crate::Network::fit) reads its data through this, and
/// [`Dataset::stream`] is the in-memory implementation of it. A corpus that
/// does not fit in memory implements it instead: keep one batch's worth of rows
/// in a buffer of your own and hand out slices of that buffer.
pub trait BatchSource {
    /// The next batch of the current epoch, or `None` once the epoch is spent.
    ///
    /// The batch borrows the source, so it is dropped before the next call and
    /// a streaming source is free to overwrite the buffer behind it.
    fn next_batch(&mut self) -> Result<Option<DatasetBatch<'_>>, NetworkError>;

    /// Starts an epoch. An in-memory source reshuffles here; a file-backed one
    /// seeks back to the beginning.
    ///
    /// The epoch index is an argument rather than a count kept by the source so
    /// that a shuffle seeded from it repeats: that is what makes a resumed run
    /// see the rows it would have seen.
    fn start_epoch(&mut self, epoch: usize) -> Result<(), NetworkError>;

    /// Discards the next `batches` batches of this epoch, which is how
    /// [`Network::fit_stream_resuming`](crate::Network::fit_stream_resuming)
    /// returns to a saved [`BatchCursor`].
    ///
    /// NOTE: the default reads them and throws them away. That costs the
    /// I/O of the prefix but none of the training, which is the expensive half;
    /// a source that can seek to a batch directly should override this.
    fn skip_batches(&mut self, batches: usize) -> Result<(), NetworkError> {
        for _ in 0..batches {
            if self.next_batch()?.is_none() {
                return Err(NetworkError::InvalidDataset(format!(
                    "the source ran out while skipping to batch {batches} of a resumed epoch"
                )));
            }
        }
        Ok(())
    }
}

/// Where a run stands in its data: the epoch it is in and how many batches of
/// that epoch it has trained on.
///
/// Serialize it beside the model and the optimizer state, and hand it back to
/// [`Network::fit_stream_resuming`](crate::Network::fit_stream_resuming) to
/// carry on from the same row rather than from the top of the epoch.
///
/// The position is exact for a source whose order is a function of the epoch
/// index, which a [`DatasetStream`] with a seed is and one without a seed is
/// not.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct BatchCursor {
    pub epoch: usize,
    pub batch: usize,
}

/// A [`Dataset`] read as batches, from [`Dataset::stream`].
///
/// It holds the dataset mutably because shuffling between epochs rewrites it.
pub struct DatasetStream<'a> {
    dataset: &'a mut Dataset,
    batch_size: usize,
    shuffle: bool,
    seed: Option<u64>,
    cursor: usize,
    /// Which of the original rows sits at each position. Shuffling permutes the
    /// order it finds, so without this every epoch's order would depend on all
    /// the orders before it, and a run resumed at epoch 40 would see rows that
    /// the run it is continuing never would.
    order: Vec<usize>,
}

impl DatasetStream<'_> {
    /// Rearranges the rows so that the original row `target[position]` sits at
    /// each `position`, whatever order they are in now.
    fn reorder(&mut self, target: &[usize]) {
        let mut position = vec![0; self.order.len()];
        for (slot, &original) in self.order.iter().enumerate() {
            position[original] = slot;
        }

        // The rows are moved out of the old vectors rather than cloned: what
        // is left behind is discarded with them.
        let mut inputs = std::mem::take(&mut self.dataset.inputs);
        let mut targets = std::mem::take(&mut self.dataset.targets);
        self.dataset.inputs = target
            .iter()
            .map(|&original| std::mem::take(&mut inputs[position[original]]))
            .collect();
        self.dataset.targets = target
            .iter()
            .map(|&original| std::mem::take(&mut targets[position[original]]))
            .collect();
        self.order = target.to_vec();
    }
}

impl BatchSource for DatasetStream<'_> {
    fn next_batch(&mut self) -> Result<Option<DatasetBatch<'_>>, NetworkError> {
        if self.cursor >= self.dataset.len() {
            return Ok(None);
        }
        let start = self.cursor;
        let end = (start + self.batch_size).min(self.dataset.len());
        self.cursor = end;

        Ok(Some(DatasetBatch {
            inputs: &self.dataset.inputs[start..end],
            targets: &self.dataset.targets[start..end],
        }))
    }

    fn start_epoch(&mut self, epoch: usize) -> Result<(), NetworkError> {
        if self.shuffle {
            let mut target: Vec<usize> = (0..self.order.len()).collect();
            match self.seed.map(|seed| seed + epoch as u64) {
                Some(seed) => target.shuffle(&mut StdRng::seed_from_u64(seed)),
                None => target.shuffle(&mut rand::thread_rng()),
            }
            self.reorder(&target);
        }
        self.cursor = 0;
        Ok(())
    }

    fn skip_batches(&mut self, batches: usize) -> Result<(), NetworkError> {
        // The rows are already here, so skipping is arithmetic.
        self.cursor = (self.cursor + batches * self.batch_size).min(self.dataset.len());
        Ok(())
    }
}

/// Random crops and rotations, drawn afresh for every batch of every epoch.
///
/// Wraps any [`BatchSource`] of images in the flat layout
/// [`from_image_folder`](Dataset::from_image_folder) produces — plane after
/// plane, each plane row-major — and hands out a transformed copy of each
/// batch. The targets pass through untouched.
///
/// The crop is the padded crop small-image training uses: the image moves by
/// up to `padding` pixels in each direction and the uncovered border reads as
/// zero. The rotation turns it about its centre by up to `degrees` either
/// way, sampled bilinearly. Both are one inverse mapping, so an image that is
/// shifted and turned is resampled once rather than twice.
///
/// With a seed, the draws for an epoch depend only on that seed and the epoch
/// index, as the shuffle of a [`DatasetStream`] does, so a resumed run sees
/// the same images.
///
/// ```
/// # use rusting_brain::{Augment, BatchSource, Dataset, TrainConfig};
/// let mut dataset = Dataset::new(vec![vec![1.0; 16]], vec![vec![1.0]]);
/// let config = TrainConfig { batch_size: 1, shuffle: false, ..Default::default() };
/// let mut augmented = Augment::new(dataset.stream(&config), 1, 4, 4)?
///     .random_crop(1)
///     .random_rotation(15.0)
///     .seed(7);
/// augmented.start_epoch(0)?;
/// let batch = augmented.next_batch()?.unwrap();
/// assert_eq!(batch.inputs[0].len(), 16);
/// assert_eq!(batch.targets[0], vec![1.0]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct Augment<S> {
    source: S,
    channels: usize,
    height: usize,
    width: usize,
    padding: usize,
    degrees: f32,
    seed: Option<u64>,
    rng: StdRng,
    inputs: Vec<Vec<f32>>,
}

impl<S: BatchSource> Augment<S> {
    /// Wraps `source`, whose rows are `channels` planes of `height` by `width`.
    /// Nothing is transformed until a crop or a rotation is asked for.
    pub fn new(
        source: S,
        channels: usize,
        height: usize,
        width: usize,
    ) -> Result<Self, NetworkError> {
        if channels == 0 || height == 0 || width == 0 {
            return Err(NetworkError::InvalidDataset(format!(
                "an image of {channels} channels of {height}x{width} has no pixels to augment"
            )));
        }
        Ok(Self {
            source,
            channels,
            height,
            width,
            padding: 0,
            degrees: 0.0,
            seed: None,
            rng: StdRng::from_entropy(),
            inputs: Vec::new(),
        })
    }

    /// Shifts each image by up to `padding` pixels along each axis.
    pub fn random_crop(mut self, padding: usize) -> Self {
        self.padding = padding;
        self
    }

    /// Turns each image by up to `degrees` either way.
    pub fn random_rotation(mut self, degrees: f32) -> Self {
        self.degrees = degrees.abs();
        self
    }

    /// Makes the draws repeat, epoch by epoch.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

/// Writes `row`, an image of `channels` planes of `height` by `width`,
/// shifted by `(shift_x, shift_y)` and turned by `angle` radians, into `out`.
fn transform(
    (channels, height, width): (usize, usize, usize),
    row: &[f32],
    out: &mut [f32],
    (shift_x, shift_y): (f32, f32),
    angle: f32,
) {
    let (sin, cos) = angle.sin_cos();
    let centre_x = (width as f32 - 1.0) / 2.0;
    let centre_y = (height as f32 - 1.0) / 2.0;
    let plane = height * width;
    let pixel = |channel: usize, x: isize, y: isize| {
        if x < 0 || y < 0 || x >= width as isize || y >= height as isize {
            0.0
        } else {
            row[channel * plane + y as usize * width + x as usize]
        }
    };
    for y in 0..height {
        for x in 0..width {
            // Where this output pixel comes from: undo the shift, then the
            // turn about the centre.
            let dx = x as f32 - shift_x - centre_x;
            let dy = y as f32 - shift_y - centre_y;
            let source_x = cos * dx + sin * dy + centre_x;
            let source_y = -sin * dx + cos * dy + centre_y;
            let (left, top) = (source_x.floor(), source_y.floor());
            let (fx, fy) = (source_x - left, source_y - top);
            let (left, top) = (left as isize, top as isize);
            for channel in 0..channels {
                out[channel * plane + y * width + x] = (1.0 - fy)
                    * ((1.0 - fx) * pixel(channel, left, top) + fx * pixel(channel, left + 1, top))
                    + fy * ((1.0 - fx) * pixel(channel, left, top + 1)
                        + fx * pixel(channel, left + 1, top + 1));
            }
        }
    }
}

impl<S: BatchSource> BatchSource for Augment<S> {
    fn next_batch(&mut self) -> Result<Option<DatasetBatch<'_>>, NetworkError> {
        use rand::Rng;
        let Some(batch) = self.source.next_batch()? else {
            return Ok(None);
        };
        let size = self.channels * self.height * self.width;
        let mut inputs = std::mem::take(&mut self.inputs);
        inputs.resize_with(batch.inputs.len(), Vec::new);
        for (row, out) in batch.inputs.iter().zip(&mut inputs) {
            if row.len() != size {
                return Err(NetworkError::InvalidDataset(format!(
                    "a row of {} values is not {} channels of {}x{}",
                    row.len(),
                    self.channels,
                    self.height,
                    self.width
                )));
            }
            let padding = self.padding as i64;
            let shift_x = self.rng.gen_range(-padding..=padding) as f32;
            let shift_y = self.rng.gen_range(-padding..=padding) as f32;
            let angle = if self.degrees > 0.0 {
                self.rng
                    .gen_range(-self.degrees..=self.degrees)
                    .to_radians()
            } else {
                0.0
            };
            out.resize(size, 0.0);
            transform(
                (self.channels, self.height, self.width),
                row,
                out,
                (shift_x, shift_y),
                angle,
            );
        }
        self.inputs = inputs;
        Ok(Some(DatasetBatch {
            inputs: &self.inputs,
            targets: batch.targets,
        }))
    }

    fn start_epoch(&mut self, epoch: usize) -> Result<(), NetworkError> {
        self.rng = match self.seed {
            Some(seed) => StdRng::seed_from_u64(seed.wrapping_add(epoch as u64)),
            None => StdRng::from_entropy(),
        };
        self.source.start_epoch(epoch)
    }
}

/// A JSONL file read one batch at a time, from [`JsonlStream::open`].
///
/// Every line is an object holding an input field and a target field, each one
/// either an array of numbers or a single number:
///
/// ```jsonl
/// {"pixels": [0.0, 0.5, 1.0], "label": [1.0, 0.0]}
/// {"pixels": [1.0, 0.5, 0.0], "label": [0.0, 1.0]}
/// ```
///
/// Only one batch is in memory at a time, so the file can be larger than the
/// machine. Lines that are empty are skipped; any other line that does not
/// parse, or that is missing either field, is an error.
///
/// NOTE: the stream does not shuffle. Shuffling a file this size means
/// either an index of line offsets or a reservoir buffer, and neither is worth
/// writing before something needs it — read a shuffled file, or load it with
/// [`Dataset::from_jsonl`] when it does fit in memory.
pub struct JsonlStream {
    path: std::path::PathBuf,
    lines: std::io::Lines<std::io::BufReader<std::fs::File>>,
    input_field: String,
    target_field: String,
    batch_size: usize,
    inputs: Vec<Vec<f32>>,
    targets: Vec<Vec<f32>>,
    line_number: usize,
}

impl JsonlStream {
    /// Opens `path`, reading `input_field` of every line as the input row and
    /// `target_field` as the target row.
    ///
    /// ```no_run
    /// # use rusting_brain::{JsonlStream, Network, TrainConfig, Activation};
    /// # let mut model = Network::builder().input_size(3).dense(2, Activation::Linear).build();
    /// let mut data = JsonlStream::open("corpus.jsonl", "pixels", "label", 32)?;
    /// model.fit_stream(&mut data, TrainConfig::default())?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn open(
        path: impl AsRef<Path>,
        input_field: &str,
        target_field: &str,
        batch_size: usize,
    ) -> Result<Self, NetworkError> {
        let path = path.as_ref().to_path_buf();
        Ok(Self {
            lines: open_lines(&path)?,
            path,
            input_field: input_field.to_string(),
            target_field: target_field.to_string(),
            batch_size: batch_size.max(1),
            inputs: Vec::new(),
            targets: Vec::new(),
            line_number: 0,
        })
    }
}

fn open_lines(
    path: &Path,
) -> Result<std::io::Lines<std::io::BufReader<std::fs::File>>, NetworkError> {
    use std::io::BufRead;
    Ok(std::io::BufReader::new(std::fs::File::open(path)?).lines())
}

/// Reads one field of a JSONL line as a row of numbers, accepting a single
/// number as the one-element row it stands for.
fn json_row(
    document: &serde_json::Value,
    field: &str,
    path: &Path,
    line_number: usize,
) -> Result<Vec<f32>, NetworkError> {
    let invalid = |message: String| {
        NetworkError::InvalidDataset(format!("{}:{line_number}: {message}", path.display()))
    };

    match document.get(field) {
        Some(serde_json::Value::Array(values)) => values
            .iter()
            .map(|value| {
                value.as_f64().map(|number| number as f32).ok_or_else(|| {
                    invalid(format!(
                        "field `{field}` holds a value that is not a number"
                    ))
                })
            })
            .collect(),
        Some(serde_json::Value::Number(number)) => {
            Ok(vec![number.as_f64().unwrap_or_default() as f32])
        }
        Some(_) => Err(invalid(format!(
            "field `{field}` is neither a number nor an array of numbers"
        ))),
        None => Err(invalid(format!("no field `{field}`"))),
    }
}

impl BatchSource for JsonlStream {
    fn next_batch(&mut self) -> Result<Option<DatasetBatch<'_>>, NetworkError> {
        // Cleared rather than reallocated: the previous batch borrowed these
        // and is gone by the time this runs.
        self.inputs.clear();
        self.targets.clear();

        while self.inputs.len() < self.batch_size {
            let Some(line) = self.lines.next() else { break };
            let line = line?;
            self.line_number += 1;
            if line.trim().is_empty() {
                continue;
            }

            let document: serde_json::Value = serde_json::from_str(&line).map_err(|error| {
                NetworkError::InvalidDataset(format!(
                    "{}:{}: {error}",
                    self.path.display(),
                    self.line_number
                ))
            })?;
            self.inputs.push(json_row(
                &document,
                &self.input_field,
                &self.path,
                self.line_number,
            )?);
            self.targets.push(json_row(
                &document,
                &self.target_field,
                &self.path,
                self.line_number,
            )?);
        }

        if self.inputs.is_empty() {
            return Ok(None);
        }
        Ok(Some(DatasetBatch {
            inputs: &self.inputs,
            targets: &self.targets,
        }))
    }

    fn start_epoch(&mut self, _epoch: usize) -> Result<(), NetworkError> {
        // Reopening rather than seeking: it is the same syscall count and it
        // survives the file having been replaced between epochs.
        self.lines = open_lines(&self.path)?;
        self.line_number = 0;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TrainConfig;

    #[test]
    fn a_seeded_shuffle_permutes_the_rows_in_pairs_and_repeats() {
        let rows = |n: usize| (0..n).map(|i| vec![i as f32]).collect::<Vec<_>>();
        let mut data = Dataset::new(rows(50), rows(50));
        data.shuffle(Some(7));
        assert_eq!(data.inputs, data.targets);
        assert_ne!(data.inputs, rows(50));
        let mut sorted = data.inputs.clone();
        sorted.sort_by(|a, b| a[0].total_cmp(&b[0]));
        assert_eq!(sorted, rows(50));
        let mut again = Dataset::new(rows(50), rows(50));
        again.shuffle(Some(7));
        assert_eq!(again.inputs, data.inputs);
    }

    /// A `.npy` file built by hand, byte for byte as `numpy.save` writes one:
    /// magic, version 1, a u16 header length, then the dict padded to a
    /// multiple of 64 bytes and terminated with a newline.
    fn write_npy(name: &str, descr: &str, shape: &str, data: &[u8]) -> std::path::PathBuf {
        let mut header =
            format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape}, }}");
        while (10 + header.len() + 1) % 64 != 0 {
            header.push(' ');
        }
        header.push('\n');

        let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
        bytes.extend((header.len() as u16).to_le_bytes());
        bytes.extend(header.as_bytes());
        bytes.extend(data);

        let path =
            std::env::temp_dir().join(format!("rusting_brain_{name}_{}.npy", std::process::id()));
        std::fs::write(&path, bytes).unwrap();
        path
    }

    /// An IDX file built by hand: two zero bytes, the type code, the number of
    /// dimensions, then each dimension as a big-endian u32.
    fn write_idx(name: &str, code: u8, shape: &[u32], data: &[u8]) -> std::path::PathBuf {
        let mut bytes = vec![0, 0, code, shape.len() as u8];
        for dimension in shape {
            bytes.extend(dimension.to_be_bytes());
        }
        bytes.extend(data);

        let path =
            std::env::temp_dir().join(format!("rusting_brain_{name}_{}.idx", std::process::id()));
        std::fs::write(&path, bytes).unwrap();
        path
    }

    #[test]
    fn a_jsonl_file_reads_both_whole_and_a_batch_at_a_time() {
        let path =
            std::env::temp_dir().join(format!("rusting_brain_ds_{}.jsonl", std::process::id()));
        std::fs::write(
            &path,
            "{\"x\":[0.0,1.0],\"y\":2.0}\n\n{\"x\":[1.0,0.0],\"y\":3.0}\n{\"x\":[1.0,1.0],\"y\":4.0}\n",
        )
        .unwrap();

        let dataset = Dataset::from_jsonl(&path, "x", "y").unwrap();
        assert_eq!(dataset.len(), 3, "the blank line is skipped, not read");
        assert_eq!(dataset.inputs[2], vec![1.0, 1.0]);
        assert_eq!(
            dataset.targets[0],
            vec![2.0],
            "a scalar target is one column"
        );

        // Two epochs, because the second one only sees anything if the reader
        // reopened the file.
        let mut stream = JsonlStream::open(&path, "x", "y", 2).unwrap();
        for epoch in 0..2 {
            stream.start_epoch(epoch).unwrap();
            let mut rows = Vec::new();
            let mut sizes = Vec::new();
            while let Some(batch) = stream.next_batch().unwrap() {
                sizes.push(batch.inputs.len());
                rows.extend_from_slice(batch.targets);
            }
            assert_eq!(sizes, [2, 1], "epoch {epoch}");
            assert_eq!(rows, dataset.targets, "epoch {epoch}");
        }

        let mut missing = JsonlStream::open(&path, "x", "absent", 2).unwrap();
        let error = missing.next_batch().unwrap_err().to_string();
        assert!(error.contains("no field `absent`"), "the error was {error}");

        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn a_stratified_split_keeps_every_class_on_both_sides() {
        // Nine rows of class 0 and three of class 2: a plain 0.75 split can
        // put every rare row on one side, a stratified one cannot.
        let mut targets = vec![vec![1.0, 0.0, 0.0]; 9];
        targets.extend(vec![vec![0.0, 0.0, 1.0]; 3]);
        let dataset = Dataset::new(vec![vec![0.0]; 12], targets);

        let (train, test) = dataset.split_stratified(0.75);
        assert_eq!((train.len(), test.len()), (9, 3));
        let rare = |set: &Dataset| set.targets.iter().filter(|row| row[2] == 1.0).count();
        assert_eq!((rare(&train), rare(&test)), (2, 1));

        // Single-column targets are the class themselves, one-hot or not.
        let dataset = Dataset::new(
            vec![vec![0.0]; 4],
            vec![vec![0.0], vec![1.0], vec![0.0], vec![1.0]],
        );
        let (train, test) = dataset.split_stratified(0.5);
        assert_eq!(train.targets, vec![vec![0.0], vec![1.0]]);
        assert_eq!(test.targets, vec![vec![0.0], vec![1.0]]);
    }

    #[test]
    fn idx_images_flatten_and_scale_while_labels_stay_indices() {
        // Three 2x2 images, exactly the shape of the MNIST file with smaller
        // numbers: type code 0x08, three dimensions.
        let pixels: Vec<u8> = (0..12).map(|value| value * 20).collect();
        let images = write_idx("images", 0x08, &[3, 2, 2], &pixels);
        let labels = write_idx("labels", 0x08, &[3], &[7, 0, 3]);

        let mut dataset = Dataset::from_idx(&images, &labels).unwrap();
        assert_eq!(dataset.inputs.len(), 3);
        assert_eq!(
            dataset.inputs[0],
            vec![0.0, 20.0 / 255.0, 40.0 / 255.0, 60.0 / 255.0]
        );
        // Labels are u8 too and must not be divided by 255.
        assert_eq!(dataset.targets, vec![vec![7.0], vec![0.0], vec![3.0]]);

        dataset.one_hot_targets(10).unwrap();
        assert_eq!(dataset.targets[0][7], 1.0);

        // Mismatched counts, a gzipped file, and a type code this does not read.
        let two = write_idx("two", 0x08, &[2], &[1, 2]);
        assert!(Dataset::from_idx(&images, &two).is_err());
        let gzipped = write_idx("gz", 0x08, &[1], &[]);
        std::fs::write(&gzipped, [0x1f, 0x8b, 0x08, 0x00]).unwrap();
        assert!(Dataset::from_idx(&gzipped, &labels).is_err());
        let unknown = write_idx("unknown", 0x0a, &[3], &[1, 2, 3]);
        assert!(Dataset::from_idx(&images, &unknown).is_err());

        for path in [images, labels, two, gzipped, unknown] {
            std::fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn npy_arrays_become_rows_and_class_indices_become_one_hot() {
        let inputs: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let x = write_npy("x", "<f4", "(3, 2)", &inputs);
        // Labels as int64, which is what numpy gives a column of class indices.
        let labels: Vec<u8> = [2i64, 0, 1]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let y = write_npy("y", "<i8", "(3,)", &labels);

        let mut dataset = Dataset::from_npy(&x, &y).unwrap();
        assert_eq!(
            dataset.inputs,
            vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]]
        );
        assert_eq!(dataset.targets, vec![vec![2.0], vec![0.0], vec![1.0]]);

        dataset.one_hot_targets(3).unwrap();
        assert_eq!(dataset.targets[0], vec![0.0, 0.0, 1.0]);
        assert_eq!(dataset.targets[1], vec![1.0, 0.0, 0.0]);
        // A class index outside the range named is an error, not a panic.
        assert!(
            Dataset::from_npy(&x, &y)
                .unwrap()
                .one_hot_targets(2)
                .is_err()
        );
        // And a row that is already one-hot cannot be made one-hot again.
        assert!(dataset.one_hot_targets(3).is_err());

        // Mismatched row counts, and a dtype this does not read.
        let short = write_npy("short", "<f4", "(2, 2)", &inputs[..16]);
        assert!(Dataset::from_npy(&short, &y).is_err());
        let complex = write_npy("complex", "<c8", "(3,)", &inputs);
        assert!(Dataset::from_npy(&x, &complex).is_err());

        for path in [x, y, short, complex] {
            std::fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn a_labeled_csv_becomes_one_hot_targets_over_sorted_classes() {
        let (dataset, classes) = Dataset::from_csv_labeled_str(
            "length,width,species\n\
             5.1,3.5,setosa\n\
             7.0,3.2,versicolor\n\
             6.3,3.3,virginica\n\
             4.9,3.0, setosa \n",
        )
        .unwrap();

        assert_eq!(classes, ["setosa", "versicolor", "virginica"]);
        assert_eq!(dataset.len(), 4);
        assert_eq!(dataset.inputs[0], vec![5.1, 3.5]);
        assert_eq!(dataset.targets[0], vec![1.0, 0.0, 0.0]);
        assert_eq!(dataset.targets[2], vec![0.0, 0.0, 1.0]);
        // Whitespace around a label is not a fourth class.
        assert_eq!(dataset.targets[3], dataset.targets[0]);

        // No header, and a row of the wrong width is an error rather than a
        // shorter input row.
        let (headerless, _) = Dataset::from_csv_labeled_str("1.0,2.0,a\n3.0,4.0,b\n").unwrap();
        assert_eq!(headerless.len(), 2);
        assert!(matches!(
            Dataset::from_csv_labeled_str("1.0,2.0,a\n3.0,b\n"),
            Err(NetworkError::InvalidDataset(message)) if message.contains("line 2")
        ));
    }

    #[test]
    fn flipping_mirrors_each_image_row_and_keeps_the_targets() {
        // Two 3x2 images, written as their pixel values so the mirror is
        // readable: rows [1,2,3] and [4,5,6] become [3,2,1] and [6,5,4].
        let mut dataset = Dataset::new(
            vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![0.0; 6]],
            vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        );
        dataset.flip_horizontal(3).unwrap();

        assert_eq!(dataset.len(), 4);
        assert_eq!(dataset.inputs[2], vec![3.0, 2.0, 1.0, 6.0, 5.0, 4.0]);
        assert_eq!(dataset.targets[2], dataset.targets[0]);
        assert_eq!(dataset.targets[3], dataset.targets[1]);
        // Flipping twice is the identity, which is the cheapest check that the
        // mirror is a mirror and not a shuffle.
        let once = dataset.inputs[2].clone();
        let mut again = Dataset::new(vec![once], vec![vec![1.0, 0.0]]);
        again.flip_horizontal(3).unwrap();
        assert_eq!(again.inputs[1], dataset.inputs[0]);

        // A width the row does not divide by is an error, not a silent
        // half-flipped image.
        assert!(dataset.flip_horizontal(4).is_err());
        assert!(dataset.flip_horizontal(0).is_err());
    }

    #[test]
    fn standardizing_centers_each_column_and_survives_a_constant_one() {
        let mut dataset = Dataset::new(
            vec![
                vec![10.0, 5.0, 1_000.0],
                vec![20.0, 5.0, 3_000.0],
                vec![30.0, 5.0, 2_000.0],
            ],
            vec![vec![0.0]; 3],
        );

        let statistics = dataset.standardize();
        assert_eq!(statistics.mean(), &[20.0, 5.0, 2_000.0]);
        // The constant column keeps a scale of one rather than dividing by zero.
        assert_eq!(statistics.deviation()[1], 1.0);

        for column in 0..3 {
            let values: Vec<f32> = dataset.inputs.iter().map(|row| row[column]).collect();
            assert!(values.iter().all(|value| value.is_finite()));
            assert!(values.iter().sum::<f32>().abs() < 1e-5);
        }
        // Two columns that differed by two orders of magnitude now do not.
        assert!((dataset.inputs[0][0] - dataset.inputs[0][2]).abs() < 1e-5);
        assert_eq!(dataset.inputs[1][1], 0.0);
    }

    #[test]
    fn the_same_statistics_scale_a_later_row() {
        let mut train = Dataset::new(vec![vec![1.0], vec![3.0]], vec![vec![0.0]; 2]);
        let statistics = train.standardize();

        let mut row = vec![3.0];
        statistics.apply_row(&mut row);
        assert_eq!(row, train.inputs[1]);

        // Serializable, because the statistics have to outlive the process that
        // measured them.
        let json = serde_json::to_string(&statistics).unwrap();
        assert_eq!(
            serde_json::from_str::<Standardizer>(&json).unwrap(),
            statistics
        );
    }

    #[test]
    fn split_preserves_all_rows() {
        let dataset = Dataset::new(
            vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]],
            vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]],
        );

        let (train, test) = dataset.split(0.5);

        assert_eq!(train.len(), 2);
        assert_eq!(test.len(), 2);
        assert_eq!(train.len() + test.len(), dataset.len());
    }

    #[test]
    fn csv_skips_a_header_and_splits_off_the_targets() {
        let dataset = Dataset::from_csv_str("x1,x2,y\r\n1.0,2.0,0.0\n3.0,4.0,1.0\n\n", 1).unwrap();

        assert_eq!(dataset.inputs, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        assert_eq!(dataset.targets, vec![vec![0.0], vec![1.0]]);
    }

    #[test]
    fn csv_reads_quoted_fields() {
        let dataset = Dataset::from_csv_str("\"1.5\",2.5\n", 1).unwrap();

        assert_eq!(dataset.inputs, vec![vec![1.5]]);
        assert_eq!(dataset.targets, vec![vec![2.5]]);
    }

    #[test]
    fn csv_rejects_a_ragged_row() {
        let error = Dataset::from_csv_str("1.0,2.0\n3.0\n", 1).unwrap_err();

        assert!(matches!(error, NetworkError::InvalidDataset(_)), "{error}");
    }

    #[test]
    fn csv_rejects_text_after_the_first_row() {
        let error = Dataset::from_csv_str("1.0,2.0\nbroken,row\n", 1).unwrap_err();

        assert!(matches!(error, NetworkError::InvalidDataset(_)), "{error}");
    }

    #[test]
    fn csv_without_data_rows_is_empty() {
        let error = Dataset::from_csv_str("x,y\n", 1).unwrap_err();

        assert!(matches!(error, NetworkError::EmptyDataset), "{error}");
    }

    #[test]
    fn csv_needs_an_input_column() {
        let error = Dataset::from_csv_str("1.0,2.0\n", 2).unwrap_err();

        assert!(matches!(error, NetworkError::InvalidDataset(_)), "{error}");
    }

    #[test]
    fn batches_cover_dataset() {
        let dataset = Dataset::new(
            vec![vec![1.0], vec![2.0], vec![3.0]],
            vec![vec![1.0], vec![2.0], vec![3.0]],
        );

        let batches = dataset.batches(2);

        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].inputs.len(), 2);
        assert_eq!(batches[1].inputs.len(), 1);
    }

    #[cfg(feature = "images")]
    #[test]
    fn an_image_folder_becomes_one_hot_rows_of_pixels() {
        let root =
            std::env::temp_dir().join(format!("rusting_brain_images_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        for (class, shade) in [("white", 255u8), ("black", 0)] {
            std::fs::create_dir_all(root.join(class)).unwrap();
            image::GrayImage::from_pixel(4, 4, image::Luma([shade]))
                .save(root.join(class).join("one.png"))
                .unwrap();
        }
        // Not an image, and not an error either.
        std::fs::write(root.join("white").join("notes.txt"), "ignored").unwrap();

        let (dataset, classes) = Dataset::from_image_folder(&root, 2, 2, true).unwrap();

        assert_eq!(classes, ["black", "white"]);
        assert_eq!(dataset.len(), 2);
        // Sorted by class index, so black comes first and its target is [1, 0].
        assert_eq!(dataset.inputs[0], vec![0.0; 4]);
        assert_eq!(dataset.targets[0], vec![1.0, 0.0]);
        assert_eq!(dataset.inputs[1], vec![1.0; 4]);
        assert_eq!(dataset.targets[1], vec![0.0, 1.0]);

        // Three planes of the same square, in RGB.
        let (rgb, _) = Dataset::from_image_folder(&root, 2, 2, false).unwrap();
        assert_eq!(rgb.inputs[1], vec![1.0; 12]);

        std::fs::remove_dir_all(root).unwrap();
    }

    /// A quarter turn of a 3x3 image lands every pixel on its rotated place.
    #[test]
    fn a_quarter_turn_moves_every_pixel_to_its_rotated_place() {
        let image: Vec<f32> = (0..9).map(|value| value as f32).collect();
        let mut out = vec![0.0; 9];
        transform(
            (1, 3, 3),
            &image,
            &mut out,
            (0.0, 0.0),
            std::f32::consts::FRAC_PI_2,
        );
        // Output (x, y) reads source (y, 2 - x): the top row takes the left
        // column, bottom to top.
        let expected = [6.0, 3.0, 0.0, 7.0, 4.0, 1.0, 8.0, 5.0, 2.0];
        for (got, want) in out.iter().zip(expected) {
            assert!((got - want).abs() < 1e-5, "{out:?}");
        }
    }

    #[test]
    fn a_crop_shifts_by_whole_pixels_and_fills_the_border_with_zero() {
        let image: Vec<f32> = (1..=32).map(|value| value as f32).collect(); // 2 planes of 4x4
        let mut dataset = Dataset::new(vec![image.clone(); 20], vec![vec![5.0]; 20]);
        let config = TrainConfig {
            batch_size: 20,
            shuffle: false,
            ..Default::default()
        };
        let mut augmented = Augment::new(dataset.stream(&config), 2, 4, 4)
            .unwrap()
            .random_crop(2)
            .seed(3);
        augmented.start_epoch(0).unwrap();
        let batch = augmented.next_batch().unwrap().unwrap();
        assert_eq!(batch.targets, vec![vec![5.0]; 20]);

        let mut shifts = std::collections::HashSet::new();
        for row in batch.inputs {
            let shift = (-2..=2)
                .flat_map(|sy| (-2..=2).map(move |sx| (sx, sy)))
                .find(|&(sx, sy)| {
                    (0..32).all(|index| {
                        let (channel, y, x) = (index / 16, (index / 4) % 4, index % 4);
                        let (from_x, from_y) = (x as i32 - sx, y as i32 - sy);
                        let want = if (0..4).contains(&from_x) && (0..4).contains(&from_y) {
                            image[channel * 16 + from_y as usize * 4 + from_x as usize]
                        } else {
                            0.0
                        };
                        row[index] == want
                    })
                })
                .unwrap_or_else(|| panic!("{row:?} is not a shifted copy"));
            shifts.insert(shift);
        }
        assert!(shifts.len() > 1, "twenty draws all took one shift");
    }

    #[test]
    fn a_seeded_augment_repeats_an_epoch_and_varies_between_epochs() {
        let image: Vec<f32> = (0..16).map(|value| value as f32).collect();
        let mut dataset = Dataset::new(vec![image; 4], vec![vec![0.0]; 4]);
        let config = TrainConfig {
            batch_size: 4,
            shuffle: false,
            ..Default::default()
        };
        let mut augmented = Augment::new(dataset.stream(&config), 1, 4, 4)
            .unwrap()
            .random_crop(1)
            .random_rotation(20.0)
            .seed(11);
        let mut epoch = |index| {
            augmented.start_epoch(index).unwrap();
            augmented.next_batch().unwrap().unwrap().inputs.to_vec()
        };
        let first = epoch(0);
        assert_eq!(first, epoch(0));
        assert_ne!(first, epoch(1));
    }

    #[test]
    fn an_augment_rejects_a_row_of_the_wrong_size() {
        let mut dataset = Dataset::new(vec![vec![0.0; 15]], vec![vec![0.0]]);
        let config = TrainConfig::default();
        let mut augmented = Augment::new(dataset.stream(&config), 1, 4, 4).unwrap();
        augmented.start_epoch(0).unwrap();
        assert!(augmented.next_batch().is_err());
        assert!(Augment::new(dataset.stream(&config), 0, 4, 4).is_err());
    }
}
