/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package coreferenceparser;

import java.io.File;
import java.io.IOException;
import java.nio.file.FileStore;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 *
 * @author matan
 */
public class CoreferenceParser {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        // TODO code application logic here

        String testText = "Barack Obama was born in Hawaii.  He is the president. Obama was elected in 2008.";
        //CoreferenceEnrich.test(testText);
        //CoreferenceEnrich.replaceCoreferences(testText);
        //CoreferenceEnrich.replaceCoreferences(testText, "He is the president.");
        //CoreferenceEnrich.replaceCoreferences(testText, "Obama was elected in 2008.");
        //CoreferenceEnrich.replaceCoreferences(testText, "He is the president. Obama was elected in 2008.");

        ReadFileTest();
    }

    public static void ReadFileTest() {
        //String path="./_test_data/test1.txt";
        //String path="./_test_data/test_1276.txt";
        String path="./_test_data/test_52263.txt";

        
        try {
            List<String> lines = Files.readAllLines(Paths.get(path));
            String pharse= lines.remove(0);
            String content = String.join(System.lineSeparator(), lines);
            //String res= CoreferenceEnrich.replaceCoreferences(content, pharse);
            String res= CoreferenceEnrich.replaceCoreferences_naive(content, pharse);
            System.out.println();
            System.out.println("Pharse: "+pharse);
            System.out.println("Result: "+res);

        } catch (Exception e) {
            // LOGGER.error("Failed to load file.", e);
            System.out.println(e);
        }

    }
}
