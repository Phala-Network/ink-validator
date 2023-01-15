use clap::Parser;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
struct Contract {
    source: ContractSource,
}

#[derive(Serialize, Deserialize, Debug)]
struct ContractSource {
    wasm: String,
}

#[derive(Parser, Debug)]
#[clap(about = "Validate given ink contract", version, author)]
struct Args {
    /// Whether allow indeterministic instructions
    #[arg(short = 'i', long)]
    indeterministic: bool,

    /// The target wasm binary or contract
    file: String,
}

fn load_wasm_bin(file: &str) -> Vec<u8> {
    if file.ends_with(".wasm") {
        std::fs::read(file).expect("Failed to read wasm file")
    } else if file.ends_with(".contract") {
        let json = std::fs::read_to_string(file).expect("Failed to read contract file");
        let contract_info: Contract =
            serde_json::from_str(&json).expect("Failed to parse contract file");
        let hexed_wasm = contract_info.source.wasm.trim_start_matches("0x");
        hex::decode(hexed_wasm).expect("Failed to hex decode wasm")
    } else {
        panic!("Only .wasm and .contract files are supported");
    }
}

fn main() {
    let args = Args::parse();
    let wasm = load_wasm_bin(&args.file);
    validator::validate(&wasm, !args.indeterministic).expect("Invalid contract");
}
