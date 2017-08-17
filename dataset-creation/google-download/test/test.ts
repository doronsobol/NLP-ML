var http= require('https');
var url= 'https://stackoverflow.com/questions/6968448/where-is-body-in-a-nodejs-http-get-response';
//var url='https://www.google.co.il/search?q=search';
http.get(url, (res) => {
  // Do stuff with response
  res.on("data", function(chunk) {
    console.log("BODY: " + chunk);
    //util.handler(chunk);

  });
});

class util{
  static handler(html){
    console.log("html: ");
    
  }
}


