export class utils {
    /************* Files ***************/
    private static fs = require('fs');

    public static writeTextFile(filepath: string, output: string):Promise<void> {
        return new Promise<void>((resolve, reject) => {
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

    public static readTextFile(filepath): Promise<string> {
        return new Promise<string>((resolve, reject) => {
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
    
    public static getDom(html:string):any{
        const jsdom = require("jsdom");
        const { JSDOM } = jsdom;
        const dom = new JSDOM(html);
        return dom;
    }

    public static getJQuery(html:string){
        var dom = this.getDom(html);
        var $ = require('jquery')(dom.window);
        return $;
    }
}