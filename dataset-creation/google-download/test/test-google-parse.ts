import { HtmlProvider } from '../web-downloaders/html-provider';
import { utils } from "../utils";

/*
var url = 'https://www.google.co.il/search?q=search';

HtmlProvider.getHtml(url).then((data) => {
    utills.writeTextFile('./examples/test-google-parse-search.txt', data).then(() => {
        console.log("done");
    });
});
*/

module test {


    utils.readTextFile("./examples/test-google-parse-search.txt").then((html) => {

        var $ = utils.getJQuery(html);
        

        /*
        var element= $("div.g > a").each((i,element) => {
           console.log($(element).attr("href").toString());
        });
*/
        var element = $("div.g>h3.r>a:first-child").each((i, element) => {
            var href: string = $(element).attr("href").toString();
            var startIndex = href.indexOf("http");
            var endIndex = href.indexOf("&sa");
            if (startIndex != -1 && endIndex != -1) {
                var url = href.substring(startIndex, endIndex);
                console.log(url);
            }
            else {
                console.log("NONE! " + href);
            }
        });
    });

}