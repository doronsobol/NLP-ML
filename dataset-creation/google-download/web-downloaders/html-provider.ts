import { utils } from "../utils";

export class HtmlProvider {
    private static http = require('http');
    private static https = require('https');

    public static getHtml(url: string): Promise<string> {
        return new Promise<string>((resolve, reject) => {
            var protocol;
            if(url.startsWith('https'))
                protocol=this.https;
            else
                protocol=this.http;
            
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

    public static getHtmlContent(html:string):string{
        var $=utils.getJQuery(html);
        var element=$("body");
        var text= element.text();
        return text; 
    }
}
