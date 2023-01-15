import { validate, validateHex } from "@kvinwang/ink-validator";
window.validateInk = function (code, allowIndeterministic) {
    if (typeof code === "string") {
        return validateHex(code, allowIndeterministic);
    } else {
        return validate(code, allowIndeterministic);
    }
}