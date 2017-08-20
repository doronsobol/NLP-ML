import { HtmlProvider } from '../web-downloaders/html-provider';
import { utils } from "../utils";

//var url = 'https://stackoverflow.com/questions/6968448/where-is-body-in-a-nodejs-http-get-response';
var url = 'http://www.morfix.co.il/search';

HtmlProvider.getHtml(url).then((data) => {
    utils.writeTextFile('./examples/test-html-provider.txt', data).then(() => {
        console.log("done");

        utils.readTextFile("./examples/test-html-provider.txt").then((html) => {
            console.log(HtmlProvider.getHtmlContent(html));
        });

    });
});




