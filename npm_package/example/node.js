const { validate } = require("@kvinwang/ink-validator");
const fs = require('fs');

function validateFile(filename) {
    const wasmBin = fs.readFileSync(filename); 
    const result = validate(wasmBin, false);
    if (typeof result === 'string') {
        console.error(`${filename}: Invalid`);
        console.error(result);
    } else {
        console.error(`${filename}: Valid`);
    }
}
validateFile("valid.wasm");
validateFile("invalid.wasm");