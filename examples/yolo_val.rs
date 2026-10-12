//! Validates a YOLO11s checkpoint on a YOLO-format dataset's `val` split the
//! way ultralytics' validator does (rect batches of 16 at imgsz 1280, stride
//! 32, pad 0.5; conf 0.001, NMS IoU 0.7, at most 300 boxes) and prints mAP50
//! and mAP50-95. Images run one at a time on the device.
//!
//! ```bash
//! cargo run --release --features cuda,images --example yolo_val -- \
//!     /home/vasylt/Code/games/cs2/fixtures/k13d_player_v1.safetensors /home/vasylt/Code/games/cs2/dataset 1 [tf32]
//! ```
//!
//! The model runs fused, as ultralytics validates it; `tf32` runs mixed
//! precision, the GEMMs on the tensor cores in bf16 with an FP32 accumulator.
//! It prints the time of each stage per image.

use rand::SeedableRng;
use rand::rngs::StdRng;
use rusting_brain::detection::{DetectionImage, read_yolo_split};
use rusting_brain::param::CudaDevice;
use rusting_brain::safetensors::SafeTensors;
use rusting_brain::yolo::{Yolo11, Yolo11Scale, validate};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let weights = args.get(1).map_or(
        "/home/vasylt/Code/games/cs2/fixtures/k13d_player_v1.safetensors",
        |a| a,
    );
    let root = Path::new(args.get(2).map_or("/home/vasylt/Code/games/cs2/dataset", |a| a));
    let classes: usize = args.get(3).map_or(Ok(1), |a| a.parse())?;
    let device = match args.get(4).map(String::as_str) {
        Some("tf32") => CudaDevice::with_mixed_precision(0, 8192)?,
        _ => CudaDevice::new(0, 8192)?,
    };
    let mut model = Yolo11::new(Yolo11Scale::S, classes, &mut StdRng::seed_from_u64(0))?;
    model.load_ultralytics(&mut SafeTensors::open(weights)?)?;
    let mut model = model.fuse();
    model.to_cuda_on(&device)?;
    let images = read_yolo_split(root, "val")?
        .into_iter()
        .map(|(path, labels)| DetectionImage::open(&path, &labels, 1280))
        .collect::<Result<Vec<_>, _>>()?;
    let v = validate(&mut model, &device, &images, 1280)?;
    println!(
        "{} images: mAP50 {:.4}, mAP50-95 {:.4}; {:.1} ms/image (model {:.1} ms)",
        images.len(),
        v.map50,
        v.map50_95,
        v.total_ms,
        v.model_ms,
    );
    let [upload, forward, download, decode, nms] = v.stages_ms;
    println!(
        "per image: upload {upload:.2} ms, forward {forward:.2} ms, download {download:.2} ms, decode {decode:.2} ms, NMS {nms:.2} ms"
    );
    Ok(())
}
