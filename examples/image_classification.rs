//! Trains a dense network to tell two kinds of image apart.
//!
//! ```bash
//! cargo run --release --example image_classification --features images
//! ```
//!
//! The images are generated into a temporary directory first, so the example
//! runs anywhere and needs no download. Replace `build_corpus` with a path to
//! your own `class/*.png` tree and nothing else here changes.
//!
//! This is pixels into a dense network, not a convolutional one: there is no
//! `Conv2d` in the crate, and a 16x16 image flattened to 256 inputs is what
//! `Dataset::from_image_folder` hands over. It is enough for shapes this
//! separable and nothing like enough for photographs.

use rusting_brain::{Activation, Dataset, Loss, Network, Optimizer, TrainConfig};

const SIZE: u32 = 16;
const PER_CLASS: usize = 120;

/// `vertical` and `horizontal` stripe images, with the stripe phase and the
/// contrast varied so the network cannot memorize one bitmap per class.
fn build_corpus(root: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    for (class, vertical) in [("vertical", true), ("horizontal", false)] {
        let directory = root.join(class);
        std::fs::create_dir_all(&directory)?;

        for index in 0..PER_CLASS {
            let phase = index % 4;
            let contrast = 120 + (index % 5) as u8 * 25;
            let image = image::GrayImage::from_fn(SIZE, SIZE, |x, y| {
                let along = if vertical { x } else { y } as usize;
                let lit = (along + phase) % 4 < 2;
                image::Luma([if lit { contrast } else { 20 }])
            });
            image.save(directory.join(format!("{index}.png")))?;
        }
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = std::env::temp_dir().join("rusting_brain_stripes");
    build_corpus(&root)?;

    // Grayscale, so each image is SIZE * SIZE inputs rather than three times
    // that. The class names come back with the dataset because a one-hot index
    // on its own says nothing.
    let (mut dataset, classes) = Dataset::from_image_folder(&root, SIZE, SIZE, true)?;
    println!(
        "{} images of {}x{}, classes {classes:?}",
        dataset.len(),
        SIZE,
        SIZE
    );

    // Shuffle before splitting: the loader returns one class after the other,
    // so an unshuffled split would train on one class and test on the other.
    dataset.shuffle(Some(42));
    // Stratified, so both halves hold both classes in the proportion the corpus
    // has them. With two balanced classes a plain `split` would do; with a rare
    // class it would not, and this costs nothing.
    let (mut train, test) = dataset.split_stratified(0.8);

    // Mirroring after the split, never before: an image and its mirror on
    // opposite sides of the split would make the test accuracy a memory test.
    // A mirrored stripe pattern is still the same stripe pattern, so the label
    // carries over unchanged.
    train.flip_horizontal(SIZE as usize)?;
    println!("{} training images after mirroring", train.len());

    let mut model = Network::builder()
        .input_size((SIZE * SIZE) as usize)
        .dense(32, Activation::Relu)
        .dense(classes.len(), Activation::Softmax)
        .loss(Loss::CrossEntropy)
        .optimizer(Optimizer::adam(0.01))
        .seed(42)
        .build();

    model.fit(
        &train,
        TrainConfig {
            epochs: 40,
            batch_size: 16,
            shuffle: true,
            seed: Some(7),
        },
    )?;

    println!(
        "train {:.1}%   test {:.1}%",
        100.0 * model.accuracy(&train)?,
        100.0 * model.accuracy(&test)?
    );

    // Accuracy is the diagonal of this and hides everything else: which class
    // the misses belong to, and what they were mistaken for.
    println!("\nactual \\ predicted   {}", classes.join("  "));
    for (class, row) in classes.iter().zip(model.confusion_matrix(&test)?) {
        println!("{class:>18}   {row:?}");
    }

    std::fs::remove_dir_all(&root)?;
    Ok(())
}
