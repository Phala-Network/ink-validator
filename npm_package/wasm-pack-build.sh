#!/bin/sh
wasm-pack build --target $1 --out-dir pkg/$1 --out-name index --release
rm pkg/$1/.gitignore
rm pkg/$1/README.md
rm pkg/$1/package.json