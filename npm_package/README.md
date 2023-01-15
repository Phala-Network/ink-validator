## ink validator

This npm package supply a function to validate given compiled WASM binary to be a valid phat contract.

## Usage

```js
const { validate } = require("@kvinwang/ink_validator");
const fs = require('fs');

const wasmBin = fs.readFileSync("qjs.wasm"); 
const result = validate(wasmBin, false);
if (typeof result === 'string') {
  console.log("Invalid contract:", result);
}
```
