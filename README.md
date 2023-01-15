# ink-validate
ink-validate is a command line tool to validate ink contracts.

## Installation

To install ink-validate, you can clone the repository and build the tool using cargo:
```bash
git clone https://github.com/kvinwang/ink-validator.git
cd ink-validator/cli
cargo install --path .
```

## Example
To validate a contract called "contract.wasm", use the following command:
```bash
ink-validate contract.wasm
```

To validate a contract called "contract.wasm" and allow indeterministic instructions, use the following command:
```bash
ink-validate -i contract.wasm
```

## License
ink-validate is released under the MIT License.