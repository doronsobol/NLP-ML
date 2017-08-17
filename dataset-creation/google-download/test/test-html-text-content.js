"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const utils_1 = require("../utils");
var test;
(function (test) {
    utils_1.utills.readTextFile("./examples/test-html-provider.txt").then((html) => {
        /*
        const dom = utills.getDom(html);
        var doc = dom.window.document;

        var element= doc.querySelector("html");
        //var text= element.textContent;
        var text= element.innerText || element.textContent;
        //console.log(element.textContent);
        console.log(text);
        */
        var $ = utils_1.utills.getJQuery(html);
        var element = $("body");
        var text = element.text();
        console.log(text);
    });
})(test || (test = {}));
//# sourceMappingURL=test-html-text-content.js.map