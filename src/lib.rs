#![no_std]
extern crate alloc;
use alloc::string::{String, ToString};

mod instrument;

pub fn validate(wasm: &[u8], deterministic: bool) -> Result<(), String> {
    _ = instrument::instrument(wasm, deterministic).map_err(|e| e.to_string())?;
    Ok(())
}
