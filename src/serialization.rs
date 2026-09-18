//! JSON save/load for both model types.
//!
//! JSON is readable and portable but costs roughly ten bytes per weight. A
//! transformer past a few million parameters should use
//! [`TransformerLm::save_bin`](crate::TransformerLm::save_bin), which writes
//! F32 or quantized I8 and carries optimizer state alongside.

use crate::network::{Network, NetworkError};
use crate::transformer::TransformerLm;
use std::path::Path;

pub fn save_json<P: AsRef<Path>>(model: &Network, path: P) -> Result<(), NetworkError> {
    model.save_json(path)
}

pub fn load_json<P: AsRef<Path>>(path: P) -> Result<Network, NetworkError> {
    Network::load_json(path)
}

pub fn save_transformer_json<P: AsRef<Path>>(
    model: &TransformerLm,
    path: P,
) -> Result<(), NetworkError> {
    model.save_json(path)
}

pub fn load_transformer_json<P: AsRef<Path>>(path: P) -> Result<TransformerLm, NetworkError> {
    TransformerLm::load_json(path)
}
