//main.rs
mod matrix;
mod network;
mod xor;
mod complex;
mod tf_compare;
mod gpu_matrix; 
mod gpu_test; 
mod tensor;

fn main() {
    // xor::xor();

    // complex::complex_example();

    // tf_compare::tensorflow_like_example();
    // tf_compare::learning_sanity_test();
    // tf_compare::large_model_learning_test();

    gpu_test::run_benchmarks();
}
