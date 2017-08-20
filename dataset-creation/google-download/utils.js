"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
class utills {
    static writeTextFile(filepath, output) {
        return new Promise((resolve, reject) => {
            this.fs.writeFile(filepath, output, function (err) {
                if (err) {
                    //return console.error(err);
                    reject(err);
                }
                //console.log("File created!");
                resolve();
            });
        });
    }
    static readTextFile(filepath) {
        return new Promise((resolve, reject) => {
            this.fs.readFile(filepath, function (err, data) {
                if (err) {
                    //return console.error(err);
                    reject(err);
                }
                //console.log("Asynchronous read: " + data.toString());
                resolve(data.toString());
            });
        });
    }
    /************* Dom ***************/
    static getDom(html) {
        const jsdom = require("jsdom");
        const { JSDOM } = jsdom;
        const dom = new JSDOM(html);
        return dom;
    }
    static getJQuery(html) {
        var dom = this.getDom(html);
        var $ = require('jquery')(dom.window);
        return $;
    }
}
/************* Files ***************/
utills.fs = require('fs');
exports.utills = utills;
//# sourceMappingURL=utils.js.map