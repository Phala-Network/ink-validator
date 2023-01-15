import { validate, validateHex } from "@kvinwang/ink-validator";
window.validateInk = function (code, indeterministic) {
    if (typeof code === "string") {
        return validateHex(code, indeterministic);
    } else {
        return validate(code, indeterministic);
    }
}
