import { utils } from "../utils";

utils.writeTextFile('test.txt', 'Im a test2');
utils.readTextFile("test.txt").then((data)=>{
    console.log("DATA: "+data);
});