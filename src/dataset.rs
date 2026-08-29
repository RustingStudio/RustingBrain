use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

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

        self.inputs = indexes.iter().map(|&i| self.inputs[i].clone()).collect();
        self.targets = indexes.iter().map(|&i| self.targets[i].clone()).collect();
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

    pub fn batches(&self, batch_size: usize) -> Vec<DatasetBatch<'_>> {
        assert!(batch_size > 0);

        self.inputs
            .chunks(batch_size)
            .zip(self.targets.chunks(batch_size))
            .map(|(inputs, targets)| DatasetBatch { inputs, targets })
            .collect()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DatasetBatch<'a> {
    pub inputs: &'a [Vec<f32>],
    pub targets: &'a [Vec<f32>],
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
