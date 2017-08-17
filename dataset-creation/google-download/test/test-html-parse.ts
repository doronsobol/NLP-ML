var html = "<html><head><title>titleTest</title></head><body><a href='test0'>test01</a><a href='test1'>test02</a><a href='test2'>test03</a></body></html>";
//var html= `<!DOCTYPE html><p>Hello world</p>`; // "Hello world"

const jsdom = require("jsdom");
const { JSDOM } = jsdom;
const dom = new JSDOM(html);
var doc=dom.window.document;
//console.log(doc.querySelector("html").textContent); 

doc.querySelectorAll("a").forEach(element => {
    console.log(element);
});


