mod matrix;
mod network;
mod xor;
mod complex;
mod tf_compare;

fn main() {
    xor::xor();

    complex::complex_example();

    tf_compare::tensorflow_like_example();
}
