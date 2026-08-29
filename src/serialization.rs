use crate::network::{Network, NetworkError};
use std::path::Path;

pub fn save_json<P: AsRef<Path>>(model: &Network, path: P) -> Result<(), NetworkError> {
    model.save_json(path)
}

pub fn load_json<P: AsRef<Path>>(path: P) -> Result<Network, NetworkError> {
    Network::load_json(path)
}
