use crate::gpu_matrix::{GpuMatrix, gpu_dot};
use crate::matrix::Matrix;
use cudarc::cublas::CudaBlas;
use cudarc::driver::CudaContext;
use std::panic::{self, AssertUnwindSafe};
use std::time::{Duration, Instant};

pub fn run_cpu_benchmark() -> f64 {
    println!("{}", "=".repeat(80));
    println!("{:^80}", "CPU MATRIX MULTIPLICATION BENCHMARK");
    println!("{}", "=".repeat(80));
    println!("CPU: {}\n", cpu_name());

    let cpu_sizes = vec![
        (64, 64, 64, "Tiny Square"),
        (128, 128, 128, "Small Square"),
        (256, 256, 256, "Medium Square"),
        (384, 384, 384, "Large Square"),
        (512, 512, 512, "Extra Large Square"),
        (256, 1024, 256, "Tall (256x1024x256)"),
        (1024, 256, 1024, "Wide (1024x256x1024)"),
    ];

    println!(
        "{:<20} {:>15} {:>15} {:>15} {:>15}",
        "Configuration", "Size", "GFLOPS", "Time (ms)", "Performance"
    );
    println!("{}", "-".repeat(85));

    let mut max_gflops = 0.0;

    for (m, k, n, name) in cpu_sizes {
        let ops = 2.0 * m as f64 * k as f64 * n as f64;
        let gflops = ops / 1e9;

        let a = Matrix::random(m, k);
        let b = Matrix::random(k, n);
        let mut c = Matrix::new(m, n);

        for _ in 0..3 {
            a.dot(&b, &mut c);
        }

        let iterations = 20;
        let mut times = Vec::with_capacity(iterations);

        for _ in 0..iterations {
            let start = Instant::now();
            a.dot(&b, &mut c);
            times.push(start.elapsed());
        }

        let avg_time = mean_duration(&times);
        let std_dev = std_duration(&times);
        let avg_gflops = gflops / avg_time.as_secs_f64();

        if avg_gflops > max_gflops {
            max_gflops = avg_gflops;
        }

        let perf_indicator = get_performance_indicator(avg_gflops);

        println!(
            "{:<20} {:>7}x{:<3}x{:<3} {:>13.2} {:>12.3} ±{:<6.2} {:>12}",
            name,
            m,
            k,
            n,
            avg_gflops,
            avg_time.as_secs_f64() * 1000.0,
            std_dev.as_secs_f64() * 1000.0,
            perf_indicator
        );
    }

    print_cpu_summary();
    max_gflops
}

pub fn run_gpu_benchmark() -> Result<f64, String> {
    catch_cuda_panic(run_gpu_benchmark_inner)?
}

fn run_gpu_benchmark_inner() -> Result<f64, String> {
    println!("\n{}", "=".repeat(80));
    println!("{:^80}", "GPU MATRIX MULTIPLICATION BENCHMARK");
    println!("{}", "=".repeat(80));

    let dev = CudaContext::new(0).map_err(|err| {
        format!(
            "CUDA driver/device initialization failed: {err:?}\n\
             Check `nvidia-smi` and verify that the NVIDIA driver is installed."
        )
    })?;
    let stream = dev
        .new_stream()
        .map_err(|err| format!("failed to create CUDA stream: {err:?}"))?;
    let blas = CudaBlas::new(stream.clone()).map_err(|err| {
        format!(
            "cuBLAS initialization failed: {err:?}\n\
             Install CUDA/cuBLAS runtime libraries or add their directory to LD_LIBRARY_PATH."
        )
    })?;

    let gpu_name = dev
        .name()
        .unwrap_or_else(|_| "NVIDIA CUDA device".to_string());
    let compute_capability = dev
        .compute_capability()
        .map(|(major, minor)| format!("{major}.{minor}"))
        .unwrap_or_else(|_| "unknown".to_string());

    println!("GPU: {gpu_name}");
    println!("Compute capability: {compute_capability}");
    println!("CUDA binding feature: {}", cuda_feature_name());
    println!("cuBLAS: dynamically loaded\n");

    let gpu_sizes = vec![
        (512, 512, 512, "512 Square", "Cache warmup"),
        (1024, 1024, 1024, "1K Square", "Good utilization"),
        (2048, 2048, 2048, "2K Square", "Compute bound"),
        (4096, 4096, 4096, "4K Square", "Memory heavy"),
        (1024, 4096, 1024, "Transformer FFN", "ML workload"),
        (4096, 1024, 4096, "Transformer Proj", "ML workload"),
        (8192, 1024, 1024, "Wide 8Kx1K", "Bandwidth test"),
        (1024, 8192, 1024, "Tall 1Kx8K", "Bandwidth test"),
    ];

    println!(
        "{:<20} {:>15} {:>15} {:>15} {:>15} {:>12}",
        "Configuration", "Size", "GFLOPS", "Time (ms)", "Utilization", "BW (GB/s)"
    );
    println!("{}", "-".repeat(100));

    let mut max_gflops = 0.0;

    for (m, k, n, name, _description) in gpu_sizes {
        let ops = 2.0 * m as f64 * k as f64 * n as f64;
        let gflops = ops / 1e9;

        let cpu_a = Matrix::random(m, k);
        let cpu_b = Matrix::random(k, n);

        let gpu_a = GpuMatrix::from_cpu(stream.clone(), dev.clone(), &cpu_a);
        let gpu_b = GpuMatrix::from_cpu(stream.clone(), dev.clone(), &cpu_b);
        let mut gpu_c = GpuMatrix::new(stream.clone(), dev.clone(), m, n);

        for _ in 0..5 {
            gpu_dot(&blas, &gpu_a, &gpu_b, &mut gpu_c);
        }
        stream.synchronize().unwrap();

        let iterations = 100;
        let mut times = Vec::with_capacity(iterations);

        for _ in 0..iterations {
            let start = Instant::now();
            gpu_dot(&blas, &gpu_a, &gpu_b, &mut gpu_c);
            stream.synchronize().unwrap();
            times.push(start.elapsed());
        }

        let avg_time = mean_duration(&times);
        let std_dev = std_duration(&times);
        let avg_gflops = gflops / avg_time.as_secs_f64();
        let utilization = (avg_gflops / 8200.0) * 100.0;

        if avg_gflops > max_gflops {
            max_gflops = avg_gflops;
        }

        let bytes = (m * k + k * n + m * n) as f64 * 4.0;
        let bandwidth = bytes / avg_time.as_secs_f64() / 1e9;

        let util_indicator = match utilization as i32 {
            u if u > 80 => "Fast",
            u if u > 50 => "Good",
            u if u > 20 => "Fair",
            _ => "Slow",
        };

        println!(
            "{:<20} {:>4}x{:<4}x{:<4} {:>13.2} {:>10.3} ±{:<6.2} {:>6.1}% {:>4} {:>8.1}",
            name,
            m,
            k,
            n,
            avg_gflops,
            avg_time.as_secs_f64() * 1000.0,
            std_dev.as_secs_f64() * 1000.0,
            utilization,
            util_indicator,
            bandwidth
        );
    }

    print_gpu_summary();
    Ok(max_gflops)
}

fn mean_duration(durations: &[Duration]) -> Duration {
    let sum: Duration = durations.iter().sum();
    sum / durations.len() as u32
}

fn std_duration(durations: &[Duration]) -> Duration {
    let mean = mean_duration(durations);
    let variance: f64 = durations
        .iter()
        .map(|d| {
            let diff = d.as_secs_f64() - mean.as_secs_f64();
            diff * diff
        })
        .sum::<f64>()
        / durations.len() as f64;
    Duration::from_secs_f64(variance.sqrt())
}

fn get_performance_indicator(gflops: f64) -> &'static str {
    match gflops as i32 {
        g if g > 4 => "Excellent",
        g if g > 3 => "Good",
        g if g > 2 => "Fair",
        g if g > 1 => "Poor",
        _ => "Very Poor",
    }
}

fn print_cpu_summary() {
    println!("\n{}", "=".repeat(80));
    println!("CPU Performance Summary:");
    println!(
        "• CPU is usually best for small matrices where GPU transfer/setup overhead dominates"
    );
    println!("• Performance depends on CPU generation, memory bandwidth, and matrix shape");
    println!("• For real inference, benchmark with the batch sizes your application uses");
}

fn print_gpu_summary() {
    println!("\n{}", "=".repeat(80));
    println!("GPU Performance Summary:");
    println!("• Best for matrices ≥ 1024x1024");
    println!("• Memory bandwidth: ~40 GB/s (good for Ampere architecture)");
    println!("• Sweet spot: 1024-4096 size range");
    println!("\nOptimization Tips:");
    println!("• Batch small matrices together");
    println!("• Use FP16 for 2x speedup");
    println!("• Keep data on GPU between operations");
    println!("• For sizes <512, consider CPU (overhead matters)");
}

pub fn run_benchmarks() {
    println!("{}", "=".repeat(80));
    println!("{:^80}", "RustingBrain Performance Benchmarks");
    println!("{}", "=".repeat(80));

    let cpu_max_gflops = run_cpu_benchmark();
    let gpu_result = run_gpu_benchmark();

    let gpu_max_gflops = gpu_result.unwrap_or_else(|err| {
        print_gpu_setup_help(&err);
        0.0
    });

    println!("\n{}", "=".repeat(80));
    println!("{:^80}", "BENCHMARK COMPLETE");
    println!("{}", "=".repeat(80));
    println!("\n Summary:");
    println!(
        "   CPU: Up to {:.2} GFLOPS (matrixmultiply)",
        cpu_max_gflops
    );
    if gpu_max_gflops > 0.0 {
        println!("   GPU: Up to {:.2} GFLOPS (cuBLAS)", gpu_max_gflops);
        println!(
            "   Speedup: ~{:.1}x for large matrices",
            gpu_max_gflops / cpu_max_gflops
        );
    } else {
        println!("   GPU: skipped because CUDA/cuBLAS is not fully available at runtime");
    }
}

fn cpu_name() -> String {
    std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|contents| {
            contents.lines().find_map(|line| {
                line.strip_prefix("model name")
                    .and_then(|line| line.split_once(':'))
                    .map(|(_, name)| name.trim().to_string())
            })
        })
        .unwrap_or_else(|| "unknown CPU".to_string())
}

fn catch_cuda_panic<T>(f: impl FnOnce() -> T) -> Result<T, String> {
    let old_hook = panic::take_hook();
    panic::set_hook(Box::new(|_| {}));
    let result = panic::catch_unwind(AssertUnwindSafe(f));
    panic::set_hook(old_hook);

    match result {
        Ok(value) => Ok(value),
        Err(payload) => {
            if let Some(message) = payload.downcast_ref::<String>() {
                Err(message.clone())
            } else if let Some(message) = payload.downcast_ref::<&str>() {
                Err((*message).to_string())
            } else {
                Err("CUDA runtime panicked during initialization".to_string())
            }
        }
    }
}

fn print_gpu_setup_help(error: &str) {
    println!("\nGPU benchmark skipped:");
    println!("{error}");
    println!("\nTo fix CUDA runtime discovery:");
    println!("  1. Confirm the NVIDIA driver works: nvidia-smi");
    println!("  2. Confirm cuBLAS is installed: ldconfig -p | grep libcublas");
    println!(
        "  3. If needed, install CUDA Toolkit from https://developer.nvidia.com/cuda-downloads"
    );
    println!("  4. If installed outside linker paths, run:");
    println!("     export LD_LIBRARY_PATH=/opt/cuda/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH");
    println!("  5. See INSTALL_CUDA.md for the full guide.");
}

fn cuda_feature_name() -> &'static str {
    if cfg!(feature = "cuda-13-1") {
        "cuda-13-1"
    } else if cfg!(feature = "cuda-13-0") {
        "cuda-13-0"
    } else if cfg!(feature = "cuda-12-8") {
        "cuda-12-8"
    } else if cfg!(feature = "cuda-12-6") {
        "cuda-12-6"
    } else if cfg!(feature = "cuda-12-4") {
        "cuda-12-4"
    } else if cfg!(feature = "cuda-12-0") {
        "cuda-12-0"
    } else if cfg!(feature = "cuda-11-8") {
        "cuda-11-8"
    } else {
        "cuda"
    }
}
