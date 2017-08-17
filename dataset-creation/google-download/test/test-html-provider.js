"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const html_provider_1 = require("../web-downloaders/html-provider");
const utils_1 = require("../utils");
//var url = 'https://stackoverflow.com/questions/6968448/where-is-body-in-a-nodejs-http-get-response';
var url = 'http://www.morfix.co.il/search';
html_provider_1.HtmlProvider.getHtml(url).then((data) => {
    utils_1.utills.writeTextFile('./examples/test-html-provider.txt', data).then(() => {
        console.log("done");
        utils_1.utills.readTextFile("./examples/test-html-provider.txt").then((html) => {
            console.log(html_provider_1.HtmlProvider.getHtmlContent(html));
        });
    });
});
//# sourceMappingURL=test-html-provider.js.map