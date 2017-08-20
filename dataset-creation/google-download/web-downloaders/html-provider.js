"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const utils_1 = require("../utils");
class HtmlProvider {
    static getHtml(url) {
        return new Promise((resolve, reject) => {
            var protocol;
            if (url.startsWith('https'))
                protocol = this.https;
            else
                protocol = this.http;
            protocol.get(url, (res) => {
                var body = '';
                res.on('data', function (chunk) {
                    body += chunk;
                });
                res.on('end', function () {
                    //console.log(body);
                    resolve(body);
                });
            });
        });
    }
    static getHtmlContent(html) {
        var $ = utils_1.utills.getJQuery(html);
        var element = $("body");
        var text = element.text();
        return text;
    }
}
HtmlProvider.http = require('http');
HtmlProvider.https = require('https');
exports.HtmlProvider = HtmlProvider;
//# sourceMappingURL=html-provider.js.map