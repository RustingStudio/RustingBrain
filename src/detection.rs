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

    /// ultralytics' `RandomPerspective` with no rotation, shear or
    /// perspective, given its draws: the image scaled by `scale` about its
    /// centre, which lands at `centre` (`[x, y]` in the output), on a
    /// `width x height` grey canvas, bilinear. Boxes follow and are clipped;
    /// those left narrower or shorter than 2 pixels, with under a tenth of
    /// their scaled area, or more elongated than 100:1 are dropped, as its
    /// `box_candidates` does.
    pub fn affine(&self, width: usize, height: usize, scale: f32, centre: [f32; 2]) -> Self {
        let half = [self.width as f32 / 2.0, self.height as f32 / 2.0];
        let mut rgb = vec![PAD; width * height * 3];
        for y in 0..height {
            let sy = (y as f32 - centre[1]) / scale + half[1];
            for x in 0..width {
                let sx = (x as f32 - centre[0]) / scale + half[0];
                let (x0, y0) = (sx.floor(), sy.floor());
                let (fx, fy) = (sx - x0, sy - y0);
                let (x0, y0) = (x0 as isize, y0 as isize);
                for c in 0..3 {
                    // Taps outside the image read the border grey, as
                    // `warpAffine` with a constant border does.
                    let at = |xx: isize, yy: isize| match xx >= 0
                        && yy >= 0
                        && (xx as usize) < self.width
                        && (yy as usize) < self.height
                    {
                        true => self.rgb[(yy as usize * self.width + xx as usize) * 3 + c] as f32,
                        false => PAD as f32,
                    };
                    let top = at(x0, y0) + (at(x0 + 1, y0) - at(x0, y0)) * fx;
                    let bottom = at(x0, y0 + 1) + (at(x0 + 1, y0 + 1) - at(x0, y0 + 1)) * fx;
                    rgb[(y * width + x) * 3 + c] = (top + (bottom - top) * fy).round() as u8;
                }
            }
        }
        let (mut classes, mut boxes) = (Vec::new(), Vec::new());
        for (&class, b) in self.classes.iter().zip(&self.boxes) {
            let map = |v: f32, axis: usize| (v - half[axis]) * scale + centre[axis];
            let moved = [
                map(b[0], 0).clamp(0.0, width as f32),
                map(b[1], 1).clamp(0.0, height as f32),
                map(b[2], 0).clamp(0.0, width as f32),
                map(b[3], 1).clamp(0.0, height as f32),
            ];
            let (w1, h1) = ((b[2] - b[0]) * scale, (b[3] - b[1]) * scale);
            let (w2, h2) = (moved[2] - moved[0], moved[3] - moved[1]);
            let aspect = (w2 / (h2 + 1e-16)).max(h2 / (w2 + 1e-16));
            if w2 > 2.0 && h2 > 2.0 && w2 * h2 / (w1 * h1 + 1e-16) > 0.1 && aspect < 100.0 {
                classes.push(class);
                boxes.push(moved);
            }
        }
        Self {
            width,
            height,
            rgb,
            classes,
            boxes,
        }
    }

    /// ultralytics' `RandomHSV` given its draws: hue, saturation and value
    /// multiplied by `gains` in OpenCV's 8-bit HSV (hue in `0..180`, wrapping).
    pub fn hsv(&mut self, gains: [f32; 3]) {
        for pixel in self.rgb.chunks_exact_mut(3) {
            let [r, g, b] = [pixel[0], pixel[1], pixel[2]].map(f32::from);
            let max = r.max(g).max(b);
            let delta = max - r.min(g).min(b);
            let mut hue = match delta == 0.0 {
                true => 0.0,
                false if max == r => 60.0 * (g - b) / delta,
                false if max == g => 120.0 + 60.0 * (b - r) / delta,
                false => 240.0 + 60.0 * (r - g) / delta,
            };
            if hue < 0.0 {
                hue += 360.0;
            }
            let saturation = match max == 0.0 {
                true => 0.0,
                false => 255.0 * delta / max,
            };
            let hue = ((hue / 2.0).round() * gains[0]).rem_euclid(180.0) * 2.0;
            let saturation = (saturation.round() * gains[1]).clamp(0.0, 255.0) / 255.0;
            let value = (max * gains[2]).clamp(0.0, 255.0);
            // Back to RGB from hue in degrees.
            let chroma = value * saturation;
            let x = chroma * (1.0 - ((hue / 60.0).rem_euclid(2.0) - 1.0).abs());
            let [r, g, b] = match (hue / 60.0) as usize {
                0 => [chroma, x, 0.0],
                1 => [x, chroma, 0.0],
                2 => [0.0, chroma, x],
                3 => [0.0, x, chroma],
                4 => [x, 0.0, chroma],
                _ => [chroma, 0.0, x],
            };
            let m = value - chroma;
            for (out, v) in pixel.iter_mut().zip([r, g, b]) {
                *out = (v + m).round().clamp(0.0, 255.0) as u8;
            }
        }
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

/// Centre, width and height to corners.
pub fn xywh_to_xyxy([x, y, w, h]: [f32; 4]) -> [f32; 4] {
    [x - w / 2.0, y - h / 2.0, x + w / 2.0, y + h / 2.0]
}

/// Corners to centre, width and height.
pub fn xyxy_to_xywh([x1, y1, x2, y2]: [f32; 4]) -> [f32; 4] {
    [(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1]
}

/// Which overlap [`box_iou`] measures.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Overlap {
    Iou,
    Giou,
    Ciou,
}

/// The overlap of `pred` with `target`, both `[x1, y1, x2, y2]`, as
/// ultralytics' `bbox_iou(xywh=False)` computes it, with its gradient with
/// respect to `pred`. CIoU's aspect-ratio weight is held constant in the
/// gradient, as it is there.
pub fn box_iou(pred: [f32; 4], target: [f32; 4], kind: Overlap) -> (f32, [f32; 4]) {
    const EPS: f32 = 1e-7;
    let [x1, y1, x2, y2] = pred;
    let [u1, v1, u2, v2] = target;
    let (w1, h1, w2, h2) = (x2 - x1, y2 - y1 + EPS, u2 - u1, v2 - v1 + EPS);
    let (iw, ih) = (x2.min(u2) - x1.max(u1), y2.min(v2) - y1.max(v1));
    let inter = iw.max(0.0) * ih.max(0.0);
    let union = w1 * h1 + w2 * h2 - inter + EPS;
    let iou = inter / union;
    let (cw, ch) = (x2.max(u2) - x1.min(u1), y2.max(v2) - y1.min(v1));
    // Gradients of the value with respect to each intermediate.
    let mut d_union = -inter / (union * union);
    let (mut d_cw, mut d_ch, mut d_w1, mut d_h1) = (0.0, 0.0, 0.0, 0.0);
    let mut grad = [0.0; 4];
    let value = match kind {
        Overlap::Iou => iou,
        Overlap::Giou => {
            let area = cw * ch + EPS;
            d_union += 1.0 / area;
            let d_area = -union / (area * area);
            (d_cw, d_ch) = (d_area * ch, d_area * cw);
            iou - (area - union) / area
        }
        Overlap::Ciou => {
            let c2 = cw * cw + ch * ch + EPS;
            let (dx, dy) = (u1 + u2 - x1 - x2, v1 + v2 - y1 - y2);
            let rho2 = (dx * dx + dy * dy) / 4.0;
            let k = 4.0 / (std::f32::consts::PI * std::f32::consts::PI);
            let angle = (w2 / h2).atan() - (w1 / h1).atan();
            let v = k * angle * angle;
            let alpha = v / (1.0 - iou + v + EPS);
            let d_c2 = rho2 / (c2 * c2);
            (d_cw, d_ch) = (d_c2 * 2.0 * cw, d_c2 * 2.0 * ch);
            // rho2 falls as the centres meet: d rho2 / d x1 = -dx / 2.
            let d_rho2 = -1.0 / c2;
            grad[0] -= d_rho2 * dx / 2.0;
            grad[2] -= d_rho2 * dx / 2.0;
            grad[1] -= d_rho2 * dy / 2.0;
            grad[3] -= d_rho2 * dy / 2.0;
            let d_atan = alpha * k * 2.0 * angle;
            let norm = w1 * w1 + h1 * h1;
            d_w1 += d_atan * h1 / norm;
            d_h1 -= d_atan * w1 / norm;
            iou - (rho2 / c2 + v * alpha)
        }
    };
    let d_inter = 1.0 / union - d_union;
    d_w1 += d_union * h1;
    d_h1 += d_union * w1;
    if iw >= 0.0 && ih >= 0.0 {
        let (d_iw, d_ih) = (d_inter * ih.max(0.0), d_inter * iw.max(0.0));
        if x2 < u2 {
            grad[2] += d_iw;
        }
        if x1 > u1 {
            grad[0] -= d_iw;
        }
        if y2 < v2 {
            grad[3] += d_ih;
        }
        if y1 > v1 {
            grad[1] -= d_ih;
        }
    }
    if x2 > u2 {
        grad[2] += d_cw;
    }
    if x1 < u1 {
        grad[0] -= d_cw;
    }
    if y2 > v2 {
        grad[3] += d_ch;
    }
    if y1 < v1 {
        grad[1] -= d_ch;
    }
    grad[2] += d_w1;
    grad[0] -= d_w1;
    grad[3] += d_h1;
    grad[1] -= d_h1;
    (value, grad)
}

/// ultralytics' distribution focal loss for one side of a box: the cross
/// entropy of `logits` (one per bin) against the two bins either side of
/// `target`, each weighted by how close the target is to it. Returns the loss
/// and its gradient with respect to the logits.
pub fn distribution_focal_loss(logits: &[f32], target: f32) -> (f32, Vec<f32>) {
    let target = target.clamp(0.0, logits.len() as f32 - 1.01);
    let left = target as usize;
    let (left_weight, right_weight) = (left as f32 + 1.0 - target, target - left as f32);
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = logits.iter().map(|&l| (l - max).exp()).sum();
    let log_p = |i: usize| logits[i] - max - sum.ln();
    let loss = -(log_p(left) * left_weight + log_p(left + 1) * right_weight);
    let mut grad: Vec<f32> = logits.iter().map(|&l| (l - max).exp() / sum).collect();
    grad[left] -= left_weight;
    grad[left + 1] -= right_weight;
    (loss, grad)
}

/// The centre of every grid cell of each `(height, width)` feature map in
/// pixels, row by row, with the stride of its map: ultralytics'
/// `make_anchors` multiplied out by the stride.
pub fn make_anchors(maps: &[(usize, usize)], strides: &[f32]) -> (Vec<[f32; 2]>, Vec<f32>) {
    let mut points = Vec::new();
    let mut point_strides = Vec::new();
    for (&(height, width), &stride) in maps.iter().zip(strides) {
        for y in 0..height {
            for x in 0..width {
                points.push([(x as f32 + 0.5) * stride, (y as f32 + 0.5) * stride]);
                point_strides.push(stride);
            }
        }
    }
    (points, point_strides)
}

/// The box `[left, top, right, bottom]` distances from `point` describe.
pub fn ltrb_to_box([px, py]: [f32; 2], [l, t, r, b]: [f32; 4]) -> [f32; 4] {
    [px - l, py - t, px + r, py + b]
}

/// The distances from `point` to the sides of `xyxy`, clamped to
/// `0..max - 0.01` as the distribution focal loss's target.
pub fn box_to_ltrb([px, py]: [f32; 2], [x1, y1, x2, y2]: [f32; 4], max: f32) -> [f32; 4] {
    [px - x1, py - y1, x2 - px, y2 - py].map(|d| d.clamp(0.0, max - 0.01))
}

/// ultralytics' task-aligned assigner, which picks the anchors each labelled
/// box trains and the soft class targets they train towards.
#[derive(Clone, Debug)]
pub struct TaskAlignedAssigner {
    /// Candidates per labelled box.
    pub topk: usize,
    /// Exponent of the class score in the alignment metric.
    pub alpha: f32,
    /// Exponent of the CIoU in the alignment metric.
    pub beta: f32,
    /// Labelled boxes narrower or shorter than this take anchors as if they
    /// were this wide or tall.
    pub min_side: f32,
}

/// What [`TaskAlignedAssigner::assign`] gives each anchor.
#[derive(Clone, Debug, PartialEq)]
pub struct Assignment {
    pub foreground: Vec<bool>,
    /// The labelled box each anchor trains towards; 0 for background.
    pub gt_index: Vec<usize>,
    pub boxes: Vec<[f32; 4]>,
    /// Soft class targets, anchors by classes, zero for background.
    pub scores: Vec<f32>,
}

impl TaskAlignedAssigner {
    /// YOLOv8 and YOLO11's settings for the given strides.
    pub fn yolo(strides: &[f32]) -> Self {
        Self {
            topk: 10,
            alpha: 0.5,
            beta: 6.0,
            min_side: strides[strides.len().min(2) - 1],
        }
    }

    /// Assigns one image's anchors. `scores` are the predicted class
    /// probabilities, anchors by `num_classes`; `pred_boxes`, `points` and
    /// `gt_boxes` are in pixels.
    pub fn assign(
        &self,
        scores: &[f32],
        num_classes: usize,
        pred_boxes: &[[f32; 4]],
        points: &[[f32; 2]],
        gt_classes: &[usize],
        gt_boxes: &[[f32; 4]],
    ) -> Assignment {
        const EPS: f32 = 1e-9;
        let anchors = points.len();
        let mut assignment = Assignment {
            foreground: vec![false; anchors],
            gt_index: vec![0; anchors],
            boxes: vec![[0.0; 4]; anchors],
            scores: vec![0.0; anchors * num_classes],
        };
        if gt_boxes.is_empty() {
            return assignment;
        }
        let gts = gt_boxes.len();
        let mut overlaps = vec![0.0f32; gts * anchors];
        let mut metric = vec![0.0f32; gts * anchors];
        let mut positive = vec![false; gts * anchors];
        for (g, (&gt, &class)) in gt_boxes.iter().zip(gt_classes).enumerate() {
            let [cx, cy, w, h] = xyxy_to_xywh(gt);
            let [x1, y1, x2, y2] =
                xywh_to_xyxy([cx, cy, w.max(self.min_side), h.max(self.min_side)]);
            let inside: Vec<bool> = points
                .iter()
                .map(|&[px, py]| px - x1 > EPS && py - y1 > EPS && x2 - px > EPS && y2 - py > EPS)
                .collect();
            let row = g * anchors..(g + 1) * anchors;
            for a in (0..anchors).filter(|&a| inside[a]) {
                let overlap = box_iou(gt, pred_boxes[a], Overlap::Ciou).0.max(0.0);
                overlaps[row.start + a] = overlap;
                metric[row.start + a] =
                    scores[a * num_classes + class].powf(self.alpha) * overlap.powf(self.beta);
            }
            // The top candidates by metric, ties to the lower anchor, kept
            // only where the anchor lies inside the box.
            let mut order: Vec<usize> = (0..anchors).collect();
            order.sort_by(|&a, &b| metric[row.start + b].total_cmp(&metric[row.start + a]));
            for &a in order.iter().take(self.topk).filter(|&&a| inside[a]) {
                positive[row.start + a] = true;
            }
        }
        for a in 0..anchors {
            let claims = (0..gts).filter(|&g| positive[g * anchors + a]).count();
            if claims > 1 {
                // The labelled box it overlaps most wins it.
                let best = (0..gts).fold(0, |best, g| {
                    match overlaps[g * anchors + a] > overlaps[best * anchors + a] {
                        true => g,
                        false => best,
                    }
                });
                (0..gts).for_each(|g| positive[g * anchors + a] = g == best);
            }
            if let Some(g) = (0..gts).find(|&g| positive[g * anchors + a]) {
                assignment.foreground[a] = true;
                assignment.gt_index[a] = g;
            }
        }
        // Scale each box's metrics so its best anchor scores its best CIoU.
        let mut norm = vec![0.0f32; anchors];
        for g in 0..gts {
            let row = g * anchors..(g + 1) * anchors;
            let kept = |values: &[f32]| {
                values[row.clone()]
                    .iter()
                    .zip(&positive[row.clone()])
                    .map(|(&v, &p)| if p { v } else { 0.0 })
                    .fold(0.0f32, f32::max)
            };
            let (best_metric, best_overlap) = (kept(&metric), kept(&overlaps));
            for a in (0..anchors).filter(|&a| positive[row.start + a]) {
                norm[a] = norm[a].max(metric[row.start + a] * best_overlap / (best_metric + EPS));
            }
        }
        for a in (0..anchors).filter(|&a| assignment.foreground[a]) {
            let g = assignment.gt_index[a];
            assignment.boxes[a] = gt_boxes[g];
            assignment.scores[a * num_classes + gt_classes[g]] = norm[a];
        }
        // ultralytics hands background anchors the first box too.
        for a in (0..anchors).filter(|&a| !assignment.foreground[a]) {
            assignment.boxes[a] = gt_boxes[0];
        }
        assignment
    }
}

/// One kept detection.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Detection {
    pub xyxy: [f32; 4],
    pub score: f32,
    pub class: usize,
}

/// ultralytics' `non_max_suppression` for one image with one label per box:
/// each `xywh` box takes its best class of `num_classes` scores if that is
/// over `conf`, then from the highest score down a box is dropped when it
/// overlaps a kept box of its class by more than `iou`. At most `max_det`,
/// highest score first.
pub fn non_max_suppression(
    xywh: &[[f32; 4]],
    scores: &[f32],
    num_classes: usize,
    conf: f32,
    iou: f32,
    max_det: usize,
) -> Vec<Detection> {
    let mut candidates: Vec<Detection> = xywh
        .iter()
        .zip(scores.chunks_exact(num_classes))
        .filter_map(|(&b, row)| {
            let class =
                (0..num_classes).fold(0, |best, c| if row[c] > row[best] { c } else { best });
            (row[class] > conf).then(|| Detection {
                xyxy: xywh_to_xyxy(b),
                score: row[class],
                class,
            })
        })
        .collect();
    candidates.sort_by(|a, b| b.score.total_cmp(&a.score));
    // Classes are kept apart by moving each one's boxes 7680 pixels along
    // both axes, as ultralytics does; the rounding that adds is kept too.
    let shifted = |d: &Detection| d.xyxy.map(|v| v + d.class as f32 * 7680.0);
    let area = |[x1, y1, x2, y2]: [f32; 4]| (x2 - x1) * (y2 - y1);
    let mut kept: Vec<Detection> = Vec::new();
    for candidate in candidates {
        let b = shifted(&candidate);
        let overlaps = kept.iter().any(|k| {
            let a = shifted(k);
            let inter = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0)
                * (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
            inter / (area(a) + area(b) - inter) > iou
        });
        if !overlaps {
            kept.push(candidate);
            if kept.len() == max_det {
                break;
            }
        }
    }
    kept
}

/// Which detections are true positives at each of the ten IoU thresholds
/// 0.5, 0.55, ..., 0.95, matched as ultralytics' validator matches them.
fn match_detections(detections: &[Detection], labels: &[(usize, [f32; 4])]) -> Vec<[bool; 10]> {
    let mut correct = vec![[false; 10]; detections.len()];
    // IoU with the class mismatches zeroed, labels by detections.
    let area = |[x1, y1, x2, y2]: [f32; 4]| (x2 - x1) * (y2 - y1);
    let iou: Vec<Vec<f32>> = labels
        .iter()
        .map(|&(class, a)| {
            detections
                .iter()
                .map(|d| {
                    let b = d.xyxy;
                    let inter = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0)
                        * (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
                    let iou = inter / (area(a) + area(b) - inter + 1e-7);
                    if d.class == class { iou } else { 0.0 }
                })
                .collect()
        })
        .collect();
    for (t, threshold) in iou_thresholds().into_iter().enumerate() {
        let mut matches: Vec<(usize, usize, f32)> = (0..labels.len())
            .flat_map(|l| (0..detections.len()).map(move |d| (l, d)))
            .filter(|&(l, d)| iou[l][d] >= threshold)
            .map(|(l, d)| (l, d, iou[l][d]))
            .collect();
        matches.sort_by(|a, b| b.2.total_cmp(&a.2));
        // Each detection's best match, in detection order; then each label's
        // first, in that order, which is the lowest detection index.
        let mut by_detection: Vec<(usize, usize, f32)> = Vec::new();
        for m in matches {
            if !by_detection.iter().any(|k| k.1 == m.1) {
                by_detection.push(m);
            }
        }
        by_detection.sort_by_key(|m| m.1);
        let mut seen_labels = Vec::new();
        for (l, d, _) in by_detection {
            if !seen_labels.contains(&l) {
                seen_labels.push(l);
                correct[d][t] = true;
            }
        }
    }
    correct
}

/// `torch.linspace(0.5, 0.95, 10)` in single precision.
fn iou_thresholds() -> [f32; 10] {
    let step = (0.95f32 - 0.5) / 9.0;
    std::array::from_fn(|i| match i < 5 {
        true => 0.5 + i as f32 * step,
        false => 0.95 - (9 - i) as f32 * step,
    })
}

/// COCO's 101-point interpolated average precision of one precision-recall
/// curve, as ultralytics' `compute_ap`.
fn average_precision(recall: &[f64], precision: &[f64]) -> f64 {
    let mut mrec = vec![0.0];
    mrec.extend_from_slice(recall);
    mrec.extend([recall.last().copied().unwrap_or(1.0), 1.0]);
    let mut mpre = vec![1.0];
    mpre.extend_from_slice(precision);
    mpre.extend([0.0, 0.0]);
    for i in (0..mpre.len() - 1).rev() {
        mpre[i] = mpre[i].max(mpre[i + 1]);
    }
    // numpy's interp: the last point at or below x, joined to the next.
    let interp = |x: f64| {
        let j = mrec.partition_point(|&r| r <= x).saturating_sub(1);
        match j + 1 >= mrec.len() || x == mrec[mrec.len() - 1] {
            true => mpre[mrec.len() - 1],
            false => mpre[j] + (mpre[j + 1] - mpre[j]) / (mrec[j + 1] - mrec[j]) * (x - mrec[j]),
        }
    };
    let xs: Vec<f64> = (0..=100)
        .map(|i| if i == 100 { 1.0 } else { i as f64 * 0.01 })
        .collect();
    let ys: Vec<f64> = xs.iter().map(|&x| interp(x)).collect();
    (0..100)
        .map(|i| (xs[i + 1] - xs[i]) * (ys[i] + ys[i + 1]) / 2.0)
        .sum()
}

/// One image's detections and its labelled `(class, xyxy)` boxes.
pub type LabelledDetections = (Vec<Detection>, Vec<(usize, [f32; 4])>);

/// mAP50 and mAP50-95 as ultralytics' validator reports them, over images
/// given as their detections and their labelled `(class, xyxy)` boxes.
/// Classes without a label are left out of the mean; with no labels at all
/// both are zero.
pub fn mean_average_precision(images: &[LabelledDetections]) -> [f32; 2] {
    let mut rows: Vec<(f32, usize, [bool; 10])> = Vec::new();
    let mut label_classes: Vec<usize> = Vec::new();
    for (detections, labels) in images {
        let correct = match labels.is_empty() {
            true => vec![[false; 10]; detections.len()],
            false => match_detections(detections, labels),
        };
        rows.extend(
            detections
                .iter()
                .zip(correct)
                .map(|(d, c)| (d.score, d.class, c)),
        );
        label_classes.extend(labels.iter().map(|l| l.0));
    }
    rows.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut classes = label_classes.clone();
    classes.sort_unstable();
    classes.dedup();
    if classes.is_empty() {
        return [0.0, 0.0];
    }
    let mut sums = [0.0f64; 2];
    for &class in &classes {
        let labelled = label_classes.iter().filter(|&&c| c == class).count() as f64;
        let hits: Vec<&[bool; 10]> = rows.iter().filter(|r| r.1 == class).map(|r| &r.2).collect();
        if hits.is_empty() {
            continue;
        }
        for t in 0..10 {
            let (mut tp, mut recall, mut precision) = (0.0, Vec::new(), Vec::new());
            for (i, hit) in hits.iter().enumerate() {
                tp += hit[t] as u8 as f64;
                recall.push(tp / (labelled + 1e-16));
                precision.push(tp / (i + 1) as f64);
            }
            let ap = average_precision(&recall, &precision);
            sums[1] += ap;
            if t == 0 {
                sums[0] += ap;
            }
        }
    }
    let n = classes.len() as f64;
    [(sums[0] / n) as f32, (sums[1] / (10.0 * n)) as f32]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_identity_affine_and_unit_hsv_keep_the_image() {
        let rgb = (0..40 * 20 * 3).map(|v| (v * 7 % 256) as u8).collect();
        let labels = [YoloLabel {
            class: 0,
            xywh: [0.5, 0.5, 0.5, 0.5],
        }];
        let image = DetectionImage::new(40, 20, rgb, &labels);
        assert_eq!(image.affine(40, 20, 1.0, [20.0, 10.0]), image);
        let mut tinted = image.clone();
        tinted.hsv([1.0, 1.0, 1.0]);
        for (a, b) in tinted.rgb.iter().zip(&image.rgb) {
            assert!(a.abs_diff(*b) <= 3, "{a} vs {b}");
        }
        // Halved about the centre, the box halves and stays centred; pushed
        // off the canvas it is dropped.
        let half = image.affine(40, 20, 0.5, [20.0, 10.0]);
        assert_eq!(half.boxes, vec![[15.0, 7.5, 25.0, 12.5]]);
        let gone = image.affine(40, 20, 1.0, [80.0, 10.0]);
        assert!(gone.boxes.is_empty());
    }

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

    fn load_ops() -> Option<serde_json::Value> {
        let fixture = Path::new("/home/vasylt/cs2/fixtures/k13b_ops.json");
        let Ok(text) = std::fs::read_to_string(fixture) else {
            eprintln!("skipped: no {}", fixture.display());
            return None;
        };
        Some(serde_json::from_str(&text).unwrap())
    }

    fn get<T: serde::de::DeserializeOwned>(value: &serde_json::Value) -> T {
        serde_json::from_value(value.clone()).unwrap()
    }

    #[test]
    fn overlap_gradients_match_finite_differences() {
        let pairs = [
            ([10.0, 12.0, 50.0, 70.0], [20.0, 5.0, 60.0, 55.0]),
            ([0.0, 0.0, 30.0, 10.0], [40.0, 20.0, 70.0, 90.0]),
            ([15.0, 15.0, 35.0, 45.0], [10.0, 10.0, 40.0, 50.0]),
        ];
        // CIoU's gradient holds its aspect weight constant, so only the
        // ultralytics fixture checks it.
        for kind in [Overlap::Iou, Overlap::Giou] {
            for (pred, target) in pairs {
                let (_, grad) = box_iou(pred, target, kind);
                for i in 0..4 {
                    let (mut up, mut down) = (pred, pred);
                    up[i] += 0.01;
                    down[i] -= 0.01;
                    let numeric =
                        (box_iou(up, target, kind).0 - box_iou(down, target, kind).0) / 0.02;
                    assert!(
                        (numeric - grad[i]).abs() < 2e-4,
                        "{kind:?} side {i}: {numeric} vs {}",
                        grad[i]
                    );
                }
            }
        }
    }

    /// Against ultralytics 8.4.174, exported by
    /// `/home/vasylt/cs2/fixtures/export_k13b_ops.py`.
    #[test]
    fn detection_ops_match_ultralytics() {
        let Some(data) = load_ops() else { return };

        let (preds, targets): (Vec<[f32; 4]>, Vec<[f32; 4]>) =
            (get(&data["iou"]["pred"]), get(&data["iou"]["target"]));
        let mut worst = [0.0f32; 2];
        for (kind, key) in [
            (Overlap::Iou, "iou"),
            (Overlap::Giou, "giou"),
            (Overlap::Ciou, "ciou"),
        ] {
            let values: Vec<f32> = get(&data["iou"][key]["value"]);
            let grads: Vec<[f32; 4]> = get(&data["iou"][key]["grad"]);
            for i in 0..preds.len() {
                let (value, grad) = box_iou(preds[i], targets[i], kind);
                worst[0] = worst[0].max((value - values[i]).abs());
                for j in 0..4 {
                    worst[1] = worst[1].max((grad[j] - grads[i][j]).abs());
                }
            }
        }
        assert!(
            worst[0] < 1e-5 && worst[1] < 1e-5,
            "overlap worst value {} grad {}",
            worst[0],
            worst[1]
        );

        let logits: Vec<Vec<f32>> = get(&data["dfl"]["logits"]);
        let targets: Vec<[f32; 4]> = get(&data["dfl"]["target"]);
        let losses: Vec<f32> = get(&data["dfl"]["loss"]);
        let grads: Vec<Vec<f32>> = get(&data["dfl"]["grad"]);
        for (b, sides) in targets.iter().enumerate() {
            let mut loss = 0.0;
            for (s, &target) in sides.iter().enumerate() {
                let (l, grad) = distribution_focal_loss(&logits[b * 4 + s], target);
                loss += l / 4.0;
                for (g, e) in grad.iter().zip(&grads[b * 4 + s]) {
                    assert!((g / 4.0 - e).abs() < 1e-6, "dfl grad {g} vs {e}");
                }
            }
            assert!(
                (loss - losses[b]).abs() < 1e-5,
                "dfl loss {loss} vs {}",
                losses[b]
            );
        }

        let a = &data["assign"];
        let points: Vec<[f32; 2]> = get(&a["points"]);
        let (maps, strides) = ([(8, 8), (4, 4), (2, 2)], [8.0, 16.0, 32.0]);
        assert_eq!(make_anchors(&maps, &strides).0, points);
        let scores: Vec<Vec<f32>> = get(&a["pred_scores"]);
        let assignment = TaskAlignedAssigner::yolo(&strides).assign(
            &scores.concat(),
            2,
            &get::<Vec<[f32; 4]>>(&a["pred_boxes"]),
            &points,
            &get::<Vec<usize>>(&a["gt_cls"]),
            &get::<Vec<[f32; 4]>>(&a["gt"]),
        );
        assert_eq!(assignment.foreground, get::<Vec<bool>>(&a["fg"]));
        assert_eq!(assignment.gt_index, get::<Vec<usize>>(&a["gt_idx"]));
        assert_eq!(assignment.boxes, get::<Vec<[f32; 4]>>(&a["target_boxes"]));
        let expected: Vec<Vec<f32>> = get(&a["target_scores"]);
        for (got, e) in assignment.scores.iter().zip(expected.concat()) {
            assert!((got - e).abs() < 1e-5, "assigned score {got} vs {e}");
        }

        let scores: Vec<Vec<f32>> = get(&data["nms"]["scores"]);
        let kept = non_max_suppression(
            &get::<Vec<[f32; 4]>>(&data["nms"]["xywh"]),
            &scores.concat(),
            2,
            0.001,
            0.7,
            300,
        );
        let expected: Vec<[f32; 6]> = get(&data["nms"]["kept"]);
        assert_eq!(kept.len(), expected.len());
        for (d, e) in kept.iter().zip(&expected) {
            assert_eq!(
                (d.xyxy, d.score, d.class),
                ([e[0], e[1], e[2], e[3]], e[4], e[5] as usize)
            );
        }

        let images: Vec<_> = data["map"]["images"]
            .as_array()
            .unwrap()
            .iter()
            .map(|image| {
                let dets: Vec<[f32; 4]> = get(&image["dets"]);
                let (classes, confs): (Vec<usize>, Vec<f32>) =
                    (get(&image["det_cls"]), get(&image["det_conf"]));
                let detections: Vec<Detection> = (0..dets.len())
                    .map(|i| Detection {
                        xyxy: dets[i],
                        score: confs[i],
                        class: classes[i],
                    })
                    .collect();
                let labels: Vec<[f32; 4]> = get(&image["labels"]);
                let labels: Vec<(usize, [f32; 4])> = get::<Vec<usize>>(&image["label_cls"])
                    .into_iter()
                    .zip(labels)
                    .collect();
                assert_eq!(
                    match_detections(&detections, &labels),
                    get::<Vec<[bool; 10]>>(&image["correct"])
                );
                (detections, labels)
            })
            .collect();
        let [map50, map50_95] = mean_average_precision(&images);
        let expected = [
            data["map"]["map50"].as_f64().unwrap(),
            data["map"]["map50_95"].as_f64().unwrap(),
        ];
        assert!(
            (map50 as f64 - expected[0]).abs() < 1e-6
                && (map50_95 as f64 - expected[1]).abs() < 1e-6,
            "mAP {map50} {map50_95} vs {expected:?}"
        );
    }
}
