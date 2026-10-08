//! Object detection data in the YOLO format: a reader for `images/<split>` and
//! `labels/<split>` directories, and the resizing, letterboxing, mosaic and
//! horizontal flip that ultralytics applies, with the same geometry so that
//! boxes land where its pipeline puts them.

use crate::network::NetworkError;
use std::path::{Path, PathBuf};

/// The grey ultralytics fills padding with.
const PAD: u8 = 114;

/// One labelled box: a class and its centre x, centre y, width and height as
/// fractions of the image.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct YoloLabel {
    pub class: usize,
    pub xywh: [f32; 4],
}

/// Parses a YOLO label file: one `class cx cy w h` line per box.
pub fn parse_yolo_labels(text: &str) -> Result<Vec<YoloLabel>, NetworkError> {
    text.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let bad = || NetworkError::InvalidDataset(format!("bad YOLO label line {line:?}"));
            let fields: Vec<&str> = line.split_whitespace().collect();
            if fields.len() != 5 {
                return Err(bad());
            }
            let class = fields[0].parse().map_err(|_| bad())?;
            let mut xywh = [0.0; 4];
            for (value, field) in xywh.iter_mut().zip(&fields[1..]) {
                *value = field.parse().map_err(|_| bad())?;
            }
            Ok(YoloLabel { class, xywh })
        })
        .collect()
}

/// Every image of `root/images/<split>` (JPEG or PNG, sorted by name) with
/// the labels of `root/labels/<split>/<stem>.txt`; an image without a label
/// file is a background with no boxes.
pub fn read_yolo_split(
    root: &Path,
    split: &str,
) -> Result<Vec<(PathBuf, Vec<YoloLabel>)>, NetworkError> {
    let mut images: Vec<PathBuf> = std::fs::read_dir(root.join("images").join(split))?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<Result<_, _>>()?;
    images.retain(|path| {
        let extension = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        ["jpg", "jpeg", "png"].contains(&extension.to_ascii_lowercase().as_str())
    });
    images.sort();
    let labels = root.join("labels").join(split);
    images
        .into_iter()
        .map(|image| {
            let file = labels
                .join(image.file_stem().unwrap_or_default())
                .with_extension("txt");
            let boxes = match std::fs::read_to_string(&file) {
                Ok(text) => parse_yolo_labels(&text)?,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => Vec::new(),
                Err(error) => return Err(error.into()),
            };
            Ok((image, boxes))
        })
        .collect()
}

/// The padded shape ultralytics validates a batch at: the batch's aspect
/// ratios (`height / width`) scaled to `size`, rounded up to `stride` after
/// adding `pad` strides. Returns `[height, width]`.
pub fn rect_shape(aspects: &[f32], size: usize, stride: usize, pad: f32) -> [usize; 2] {
    let min = aspects.iter().copied().fold(f32::INFINITY, f32::min);
    let max = aspects.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let shape = match (max < 1.0, min > 1.0) {
        (true, _) => [max as f64, 1.0],
        (_, true) => [1.0, 1.0 / min as f64],
        _ => [1.0, 1.0],
    };
    shape.map(|side| (side * size as f64 / stride as f64 + pad as f64).ceil() as usize * stride)
}

/// An RGB image, row-major and interleaved, with its boxes as pixel corners
/// `[x1, y1, x2, y2]` and their classes.
#[derive(Clone, Debug, PartialEq)]
pub struct DetectionImage {
    pub width: usize,
    pub height: usize,
    pub rgb: Vec<u8>,
    pub classes: Vec<usize>,
    pub boxes: Vec<[f32; 4]>,
}

impl DetectionImage {
    /// Takes `labels` in YOLO fractions and turns them into pixel corners.
    pub fn new(width: usize, height: usize, rgb: Vec<u8>, labels: &[YoloLabel]) -> Self {
        let (w, h) = (width as f32, height as f32);
        Self {
            width,
            height,
            rgb,
            classes: labels.iter().map(|label| label.class).collect(),
            boxes: labels
                .iter()
                .map(
                    |YoloLabel {
                         xywh: [x, y, bw, bh],
                         ..
                     }| {
                        [
                            (x - bw / 2.0) * w,
                            (y - bh / 2.0) * h,
                            (x + bw / 2.0) * w,
                            (y + bh / 2.0) * h,
                        ]
                    },
                )
                .collect(),
        }
    }

    /// Decodes an image and resizes it so its long side is `long_side`, the
    /// way ultralytics loads every image before letterboxing or mosaic.
    #[cfg(feature = "images")]
    pub fn open(path: &Path, labels: &[YoloLabel], long_side: usize) -> Result<Self, NetworkError> {
        let image = image::open(path)
            .map_err(|error| NetworkError::InvalidDataset(format!("{}: {error}", path.display())))?
            .into_rgb8();
        let (width, height) = (image.width() as usize, image.height() as usize);
        let image = Self::new(width, height, image.into_raw(), labels);
        let ratio = long_side as f64 / width.max(height) as f64;
        Ok(match ratio == 1.0 {
            true => image,
            false => image.resized(
                ((width as f64 * ratio).ceil() as usize).min(long_side),
                ((height as f64 * ratio).ceil() as usize).min(long_side),
            ),
        })
    }

    /// Bilinear resize with OpenCV's pixel-centre convention (`INTER_LINEAR`),
    /// boxes scaled with it.
    pub fn resized(&self, width: usize, height: usize) -> Self {
        // For each output coordinate, the source index and the weight of the
        // next one, clamped at the edges as OpenCV does.
        let taps = |out: usize, source: usize| -> Vec<(usize, f32)> {
            let scale = source as f64 / out as f64;
            (0..out)
                .map(|i| {
                    let f = (i as f64 + 0.5) * scale - 0.5;
                    let index = f.floor();
                    match (index < 0.0, index >= (source - 1) as f64) {
                        (true, _) => (0, 0.0),
                        (_, true) => (source - 1, 0.0),
                        _ => (index as usize, (f - index) as f32),
                    }
                })
                .collect()
        };
        let (xs, ys) = (taps(width, self.width), taps(height, self.height));
        let mut rgb = vec![0u8; width * height * 3];
        for (row, &(y, wy)) in rgb.chunks_exact_mut(width * 3).zip(&ys) {
            let y1 = (y + 1).min(self.height - 1);
            for (pixel, &(x, wx)) in row.chunks_exact_mut(3).zip(&xs) {
                let x1 = (x + 1).min(self.width - 1);
                for (c, value) in pixel.iter_mut().enumerate() {
                    let at = |yy: usize, xx: usize| self.rgb[(yy * self.width + xx) * 3 + c] as f32;
                    let top = at(y, x) + (at(y, x1) - at(y, x)) * wx;
                    let bottom = at(y1, x) + (at(y1, x1) - at(y1, x)) * wx;
                    *value = (top + (bottom - top) * wy).round() as u8;
                }
            }
        }
        let (sx, sy) = (
            width as f32 / self.width as f32,
            height as f32 / self.height as f32,
        );
        Self {
            width,
            height,
            rgb,
            classes: self.classes.clone(),
            boxes: self
                .boxes
                .iter()
                .map(|[x1, y1, x2, y2]| [x1 * sx, y1 * sy, x2 * sx, y2 * sy])
                .collect(),
        }
    }

    /// Scales the image to fit `height x width` and centres it on grey
    /// padding, as ultralytics' `LetterBox` does. `scaleup: false` only ever
    /// shrinks, which is what validation uses. Returns the image, the scale,
    /// and the left and top padding, which map a prediction back.
    pub fn letterbox(&self, height: usize, width: usize, scaleup: bool) -> (Self, f32, [usize; 2]) {
        let mut ratio = (height as f64 / self.height as f64).min(width as f64 / self.width as f64);
        if !scaleup {
            ratio = ratio.min(1.0);
        }
        // Python's round, ties to even, as ultralytics computes these.
        let unpadded =
            [self.width, self.height].map(|side| (side as f64 * ratio).round_ties_even() as usize);
        let half = [
            (width - unpadded[0]) as f64 / 2.0,
            (height - unpadded[1]) as f64 / 2.0,
        ];
        let [left, top] = half.map(|pad| (pad - 0.1).round_ties_even() as usize);
        let inner = match unpadded == [self.width, self.height] {
            true => self.clone(),
            false => self.resized(unpadded[0], unpadded[1]),
        };
        let mut rgb = vec![PAD; width * height * 3];
        for (y, row) in inner.rgb.chunks_exact(inner.width * 3).enumerate() {
            let start = ((top + y) * width + left) * 3;
            rgb[start..start + row.len()].copy_from_slice(row);
        }
        let ratio = ratio as f32;
        let [dx, dy] = [left as f32, top as f32];
        let image = Self {
            width,
            height,
            rgb,
            classes: self.classes.clone(),
            // Scaled from this image's boxes by the ratio, not taken from the
            // resized one, as ultralytics does.
            boxes: self
                .boxes
                .iter()
                .map(|[x1, y1, x2, y2]| {
                    [
                        x1 * ratio + dx,
                        y1 * ratio + dy,
                        x2 * ratio + dx,
                        y2 * ratio + dy,
                    ]
                })
                .collect(),
        };
        (image, ratio, [left, top])
    }

    /// Mirrors the image and its boxes left to right.
    pub fn flip_horizontal(&mut self) {
        for row in self.rgb.chunks_exact_mut(self.width * 3) {
            row.reverse();
            row.chunks_exact_mut(3).for_each(|pixel| pixel.reverse());
        }
        let w = self.width as f32;
        for b in &mut self.boxes {
            *b = [w - b[2], b[1], w - b[0], b[3]];
        }
    }
}

/// Ultralytics' four-image mosaic: a `2 size` square of grey with the tiles
/// placed top left, top right, bottom left and bottom right of `centre`
/// (`[y, x]`, which ultralytics draws from `size / 2 ..= 3 size / 2`), boxes
/// clipped to the canvas and the ones left with no area dropped. Tiles are
/// expected at a long side of `size`, as [`DetectionImage::open`] gives them.
pub fn mosaic(tiles: [&DetectionImage; 4], size: usize, centre: [usize; 2]) -> DetectionImage {
    let side = 2 * size;
    let [yc, xc] = centre;
    let mut out = DetectionImage {
        width: side,
        height: side,
        rgb: vec![PAD; side * side * 3],
        classes: Vec::new(),
        boxes: Vec::new(),
    };
    for (i, tile) in tiles.into_iter().enumerate() {
        let (w, h) = (tile.width, tile.height);
        // Where the tile goes on the canvas, `a`, and which part of it, `b`.
        let (a, b) = match i {
            0 => {
                let a = [xc.saturating_sub(w), yc.saturating_sub(h), xc, yc];
                (a, [w - (a[2] - a[0]), h - (a[3] - a[1]), w, h])
            }
            1 => {
                let a = [xc, yc.saturating_sub(h), (xc + w).min(side), yc];
                (a, [0, h - (a[3] - a[1]), w.min(a[2] - a[0]), h])
            }
            2 => {
                let a = [xc.saturating_sub(w), yc, xc, (yc + h).min(side)];
                (a, [w - (a[2] - a[0]), 0, w, (a[3] - a[1]).min(h)])
            }
            _ => {
                let a = [xc, yc, (xc + w).min(side), (yc + h).min(side)];
                (a, [0, 0, w.min(a[2] - a[0]), (a[3] - a[1]).min(h)])
            }
        };
        for row in 0..a[3] - a[1] {
            let to = ((a[1] + row) * side + a[0]) * 3;
            let from = ((b[1] + row) * w + b[0]) * 3;
            let len = (a[2] - a[0]) * 3;
            out.rgb[to..to + len].copy_from_slice(&tile.rgb[from..from + len]);
        }
        let [dx, dy] = [a[0] as f32 - b[0] as f32, a[1] as f32 - b[1] as f32];
        for (&class, [x1, y1, x2, y2]) in tile.classes.iter().zip(&tile.boxes) {
            let clip = |v: f32| v.clamp(0.0, side as f32);
            let moved = [clip(x1 + dx), clip(y1 + dy), clip(x2 + dx), clip(y2 + dy)];
            if (moved[2] - moved[0]) * (moved[3] - moved[1]) > 0.0 {
                out.classes.push(class);
                out.boxes.push(moved);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tile(width: usize, height: usize, shade: u8, label: YoloLabel) -> DetectionImage {
        DetectionImage::new(width, height, vec![shade; width * height * 3], &[label])
    }

    #[test]
    fn labels_parse_and_a_missing_file_is_a_background() {
        let labels = parse_yolo_labels("0 0.5 0.25 0.1 0.2\n\n1 0.1 0.1 0.05 0.05\n").unwrap();
        assert_eq!(labels.len(), 2);
        assert_eq!(
            labels[1],
            YoloLabel {
                class: 1,
                xywh: [0.1, 0.1, 0.05, 0.05]
            }
        );
        assert!(parse_yolo_labels("0 0.5 0.5").is_err());

        let root = std::env::temp_dir().join(format!("yolo_split_{}", std::process::id()));
        for dir in ["images/train", "labels/train"] {
            std::fs::create_dir_all(root.join(dir)).unwrap();
        }
        std::fs::write(root.join("images/train/b.jpg"), b"").unwrap();
        std::fs::write(root.join("images/train/a.png"), b"").unwrap();
        std::fs::write(root.join("images/train/notes.txt"), b"").unwrap();
        std::fs::write(root.join("labels/train/a.txt"), "0 0.5 0.5 0.2 0.2\n").unwrap();
        let split = read_yolo_split(&root, "train").unwrap();
        std::fs::remove_dir_all(&root).unwrap();
        let names: Vec<_> = split
            .iter()
            .map(|(path, labels)| (path.file_name().unwrap().to_owned(), labels.len()))
            .collect();
        assert_eq!(names, [("a.png".into(), 1), ("b.jpg".into(), 0)]);
    }

    #[test]
    fn a_flip_mirrors_pixels_and_boxes() {
        let mut image = DetectionImage::new(
            3,
            1,
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9],
            &[YoloLabel {
                class: 0,
                xywh: [0.25, 0.5, 0.1, 1.0],
            }],
        );
        image.flip_horizontal();
        assert_eq!(image.rgb, [7, 8, 9, 4, 5, 6, 1, 2, 3]);
        let expected = [3.0 - 0.9, 0.0, 3.0 - 0.6, 1.0];
        assert!(
            image.boxes[0]
                .iter()
                .zip(expected)
                .all(|(a, b)| (a - b).abs() < 1e-5)
        );
    }

    #[test]
    fn a_mosaic_places_each_tile_around_the_centre_and_drops_boxes_it_cuts_off() {
        let label = |x, y| YoloLabel {
            class: 0,
            xywh: [x, y, 0.1, 0.1],
        };
        let tiles = [
            tile(8, 6, 10, label(0.5, 0.5)),
            tile(8, 6, 20, label(0.95, 0.5)),
            tile(8, 6, 30, label(0.5, 0.5)),
            tile(8, 6, 40, label(0.5, 0.5)),
        ];
        // Size 8 makes a 16 canvas; a centre near the right edge cuts the top
        // right tile to two columns and the bottom right one too, which loses
        // both of their boxes.
        let out = mosaic([&tiles[0], &tiles[1], &tiles[2], &tiles[3]], 8, [7, 14]);
        let at = |y: usize, x: usize| out.rgb[(y * 16 + x) * 3];
        assert_eq!(
            [
                at(6, 13),
                at(6, 14),
                at(7, 13),
                at(7, 14),
                at(0, 0),
                at(15, 0)
            ],
            [10, 20, 30, 40, 114, 114]
        );
        assert_eq!(out.boxes.len(), 2);
        let top_left = out.boxes[0];
        let expected = [6.0 + 3.6, 1.0 + 2.7, 6.0 + 4.4, 1.0 + 3.3];
        assert!(
            top_left
                .iter()
                .zip(expected)
                .all(|(a, b)| (a - b).abs() < 1e-4),
            "{top_left:?}"
        );
    }

    #[test]
    fn a_letterbox_centres_the_scaled_image_on_grey() {
        let image = tile(
            4,
            2,
            50,
            YoloLabel {
                class: 0,
                xywh: [0.5, 0.5, 0.5, 0.5],
            },
        );
        let (out, ratio, [left, top]) = image.letterbox(8, 8, true);
        assert_eq!((ratio, left, top), (2.0, 0, 2));
        assert_eq!(out.rgb[0], 114);
        assert_eq!(out.rgb[(2 * 8) * 3], 50);
        assert_eq!(out.boxes[0], [2.0, 3.0, 6.0, 5.0]);
        // Without scaling up the image keeps its size and is padded all round.
        let (_, ratio, pad) = image.letterbox(8, 8, false);
        assert_eq!((ratio, pad), (1.0, [2, 3]));
        assert_eq!(rect_shape(&[1080.0 / 1920.0], 1280, 32, 0.5), [736, 1312]);
    }

    /// Boxes, shape and pixels against ultralytics 8.4.174 on the CS2 player
    /// dataset, exported by `/home/vasylt/cs2/fixtures/export_k13c_data.py`.
    #[cfg(feature = "images")]
    #[test]
    fn the_pipeline_matches_ultralytics_on_the_player_dataset() {
        let fixture = Path::new("/home/vasylt/cs2/fixtures/k13c_data.json");
        let Ok(text) = std::fs::read_to_string(fixture) else {
            eprintln!("skipped: no {}", fixture.display());
            return;
        };
        let data: serde_json::Value = serde_json::from_str(&text).unwrap();
        let root = Path::new("/home/vasylt/cs2/dataset");
        let labels_of = |split: &str, file: &str| {
            let stem = Path::new(file).file_stem().unwrap().to_str().unwrap();
            parse_yolo_labels(
                &std::fs::read_to_string(root.join(format!("labels/{split}/{stem}.txt"))).unwrap(),
            )
            .unwrap()
        };
        let (mut worst, mut total, mut count) = (0i32, 0i64, 0i64);
        for record in data["val"].as_array().unwrap() {
            let file = record["file"].as_str().unwrap();
            let image = DetectionImage::open(
                &root.join("images/val").join(file),
                &labels_of("val", file),
                1280,
            )
            .unwrap();
            let shape = rect_shape(&[image.height as f32 / image.width as f32], 1280, 32, 0.5);
            assert_eq!(
                shape.to_vec(),
                serde_json::from_value::<Vec<usize>>(record["shape"].clone()).unwrap()
            );
            let (out, _, pad) = image.letterbox(shape[0], shape[1], false);
            assert_eq!(
                pad.map(|p| p as f64).to_vec(),
                serde_json::from_value::<Vec<f64>>(record["ratio_pad"][1].clone()).unwrap()
            );
            let expected: Vec<[f32; 4]> = serde_json::from_value(record["xywhn"].clone()).unwrap();
            assert_eq!(out.boxes.len(), expected.len());
            let (w, h) = (out.width as f32, out.height as f32);
            for ([x1, y1, x2, y2], e) in out.boxes.iter().zip(&expected) {
                let got = [
                    (x1 + x2) / 2.0 / w,
                    (y1 + y2) / 2.0 / h,
                    (x2 - x1) / w,
                    (y2 - y1) / h,
                ];
                assert!(
                    got.iter().zip(e).all(|(a, b)| (a - b).abs() < 1e-5),
                    "{file}: {got:?} against {e:?}"
                );
            }
            for pixel in record["pixels"].as_array().unwrap() {
                let (y, x) = (
                    pixel[0].as_u64().unwrap() as usize,
                    pixel[1].as_u64().unwrap() as usize,
                );
                for c in 0..3 {
                    let diff = out.rgb[(y * out.width + x) * 3 + c] as i32
                        - pixel[2][c].as_i64().unwrap() as i32;
                    worst = worst.max(diff.abs());
                    total += diff.abs() as i64;
                    count += 1;
                }
            }
        }
        // A different JPEG decoder and OpenCV's fixed-point weights move a
        // pixel by a level (measured: worst 1, mean 0.098 over 1536 values).
        let mean = total as f64 / count as f64;
        eprintln!("pixels: worst {worst}, mean {mean:.3} levels");
        assert!(worst <= 2 && mean < 0.3, "worst {worst}, mean {mean}");

        let record = &data["mosaic"];
        let tiles: Vec<DetectionImage> = record["files"]
            .as_array()
            .unwrap()
            .iter()
            .map(|file| {
                let file = file.as_str().unwrap();
                DetectionImage::open(
                    &root.join("images/train").join(file),
                    &labels_of("train", file),
                    1280,
                )
                .unwrap()
            })
            .collect();
        let centre: [usize; 2] = serde_json::from_value(record["centre_yx"].clone()).unwrap();
        let out = mosaic([&tiles[0], &tiles[1], &tiles[2], &tiles[3]], 1280, centre);
        let expected: Vec<[f32; 4]> = serde_json::from_value(record["xyxy"].clone()).unwrap();
        assert_eq!(out.boxes.len(), expected.len());
        for (got, e) in out.boxes.iter().zip(&expected) {
            assert!(
                got.iter().zip(e).all(|(a, b)| (a - b).abs() < 1e-2),
                "{got:?} against {e:?}"
            );
        }
    }
}
