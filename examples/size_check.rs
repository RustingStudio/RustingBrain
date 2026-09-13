use rusting_brain::{Precision, TransformerLm};

fn main() {
    let mut model = TransformerLm::builder().seed(1).build().unwrap();
    let counts = model.parameter_counts();
    let dir = std::env::temp_dir();
    let (json, f32p, q8p) = (
        dir.join("rb_size.json"),
        dir.join("rb_size.f32.rbw"),
        dir.join("rb_size.q8.rbw"),
    );

    model.save_json(&json).unwrap();
    model.save_bin(&f32p, Precision::F32).unwrap();
    model.save_bin(&q8p, Precision::Q8).unwrap();

    println!("{:.1}M params", counts.total as f64 / 1e6);
    for (name, path) in [("json", &json), ("f32 ", &f32p), ("q8  ", &q8p)] {
        let bytes = std::fs::metadata(path).unwrap().len();
        println!(
            "{name}: {:>7.1} MB   {:.2} bytes/param",
            bytes as f64 / 1e6,
            bytes as f64 / counts.total as f64
        );
        std::fs::remove_file(path).ok();
    }
}
