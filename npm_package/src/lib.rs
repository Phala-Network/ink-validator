#![allow(non_snake_case)]

use wasm_bindgen::prelude::*;

// When the `wee_alloc` feature is enabled, use `wee_alloc` as the global
// allocator.
#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

/// Validates the given WebAssembly (wasm) binary to check for unsupported instructions.
///
/// @param {Uint8Array} wasmBytes - The binary to be validated.
/// @param {boolean} indeterministic - A flag indicating whether or not indeterministic instructions are allowed. Set to true for IndeterministicInkCode resources.
/// @returns {string | undefined} - undefined if the binary is valid. an error description if the binary contains unsupported instructions.
#[wasm_bindgen]
pub fn validate(wasmBytes: &[u8], indeterministic: bool) -> Option<String> {
    match validator::validate(wasmBytes, !indeterministic) {
        Ok(_) => None,
        Err(e) => Some(e.to_string()),
    }
}

/// Same as the function validate except that the first argument is a string of hex encoded wasm binary.
///
#[wasm_bindgen]
pub fn validateHex(wasmHex: &str, indeterministic: bool) -> Option<String> {
    let wasm_hex = wasmHex.trim_start_matches("0x");
    let Ok(wasm_bytes) = hex::decode(wasm_hex) else {
        return Some("Failed to decode hexed wasm".to_string());
    };
    validate(&wasm_bytes, indeterministic)
}
