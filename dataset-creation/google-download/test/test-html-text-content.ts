import { utils } from "../utils";

module test {


    utils.readTextFile("./examples/test-html-provider.txt").then((html) => {
        /*
        const dom = utills.getDom(html); 
        var doc = dom.window.document;

        var element= doc.querySelector("html");
        //var text= element.textContent;
        var text= element.innerText || element.textContent;
        //console.log(element.textContent);
        console.log(text); 
        */



        var $=utils.getJQuery(html);
        var element=$("body");
        var text= element.text();
        console.log(text); 

    });

}
