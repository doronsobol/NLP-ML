"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const utils_1 = require("../utils");
utils_1.utills.writeTextFile('test.txt', 'Im a test2');
utils_1.utills.readTextFile("test.txt").then((data) => {
    console.log("DATA: " + data);
});
//# sourceMappingURL=test-read-write-file.js.map