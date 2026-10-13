use std::io::Read;
fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().unwrap();
    let megabytes: usize = args.next().unwrap().parse().unwrap();
    let mut bytes = Vec::new();
    std::fs::File::open(&path)
        .unwrap()
        .take((megabytes << 20) as u64)
        .read_to_end(&mut bytes)
        .unwrap();
    let text = String::from_utf8_lossy(&bytes);
    let lines: Vec<&str> = text.split_inclusive('\n').collect();
    let start = std::time::Instant::now();
    let json =
        rusting_brain::Bpe::train_json_split_digits(&lines, 32_000, &["<|endoftext|>"]).unwrap();
    println!(
        "{megabytes} MB, {} lines: {:.1} s, json {} bytes",
        lines.len(),
        start.elapsed().as_secs_f64(),
        json.len()
    );
}
