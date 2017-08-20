package coreferenceparser;

import edu.stanford.nlp.coref.CorefCoreAnnotations;
import edu.stanford.nlp.coref.data.CorefChain;
import edu.stanford.nlp.coref.data.CorefChain.CorefMention;

import edu.stanford.nlp.coref.data.Mention;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.*;
import edu.stanford.nlp.util.CoreMap;

import java.util.*;

public class CoreferenceEnrich {

    public static void test(String text) {
        Annotation document = new Annotation(text);
        Properties props = new Properties();
        //props.setProperty("annotators", "tokenize,ssplit,pos,lemma,ner,parse,mention,coref");
        props.setProperty("annotators", "tokenize,ssplit,pos,lemma,ner,parse,mention,dcoref");
        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
        pipeline.annotate(document);
        System.out.println("---");
        System.out.println("coref chains");
        for (CorefChain cc : document.get(CorefCoreAnnotations.CorefChainAnnotation.class).values()) {
            System.out.println("\t" + cc);
            System.out.println("getRepresentativeMention " + cc.getRepresentativeMention());
            System.out.println("getRepresentativeMention " + cc.getRepresentativeMention().position);
        }
        for (CoreMap sentence : document.get(CoreAnnotations.SentencesAnnotation.class)) {
            System.out.println("---");
            System.out.println("mentions");
            for (Mention m : sentence.get(CorefCoreAnnotations.CorefMentionsAnnotation.class)) {
                System.out.println("\t" + m);
                System.out.println("\t" + m.headIndex);
            }
        }
    }

    public static String replaceCoreferences(String text) {
        Properties props = new Properties();
        props.put("annotators", "tokenize, ssplit, pos, lemma, ner, parse, dcoref");
        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
        Annotation doc = new Annotation(text);
        pipeline.annotate(doc);

        Map<Integer, CorefChain> corefs = doc.get(CorefCoreAnnotations.CorefChainAnnotation.class);
        List<CoreMap> sentences = doc.get(CoreAnnotations.SentencesAnnotation.class);

        List<String> resolved = new ArrayList<String>();

        for (CoreMap sentence : sentences) {

            List<CoreLabel> tokens = sentence.get(CoreAnnotations.TokensAnnotation.class);

            for (CoreLabel token : tokens) {

                Integer corefClustId = token.get(CorefCoreAnnotations.CorefClusterIdAnnotation.class);
                System.out.println(token.word() + " --> corefClusterID = " + corefClustId);

                CorefChain chain = corefs.get(corefClustId);
                System.out.println("matched chain = " + chain);

                if (chain == null) {
                    resolved.add(token.word());
                    System.out.println("Adding the same word " + token.word());
                } else {

                    int sentINdx = chain.getRepresentativeMention().sentNum - 1;
                    System.out.println("sentINdx :" + sentINdx);
                    CoreMap corefSentence = sentences.get(sentINdx);
                    List<CoreLabel> corefSentenceTokens = corefSentence.get(TokensAnnotation.class);
                    String newwords = "";
                    CorefMention reprMent = chain.getRepresentativeMention();
                    System.out.println("reprMent :" + reprMent);
                    System.out.println("Token index " + token.index());
                    System.out.println("Start index " + reprMent.startIndex);
                    System.out.println("End Index " + reprMent.endIndex);
                    if (token.index() <= reprMent.startIndex || token.index() >= reprMent.endIndex) {

                        for (int i = reprMent.startIndex; i < reprMent.endIndex; i++) {
                            CoreLabel matchedLabel = corefSentenceTokens.get(i - 1);
                            resolved.add(matchedLabel.word().replace("'s", ""));
                            System.out.println("matchedLabel : " + matchedLabel.word());
                            newwords += matchedLabel.word() + " ";

                        }
                    } else {
                        resolved.add(token.word());
                        System.out.println("token.word() : " + token.word());
                    }

                    System.out.println("converting " + token.word() + " to " + newwords);
                }

                System.out.println();
                System.out.println();
                System.out.println("-----------------------------------------------------------------");

            }

        }

        String resolvedStr = "";
        System.out.println();
        for (String str : resolved) {
            resolvedStr += str + " ";
        }
        System.out.println(resolvedStr);
        return resolvedStr;
    }

    public static String replaceCoreferences_naive(String corpusText, String text) {
        Properties props = new Properties();
        props.put("annotators", "tokenize, ssplit, pos, lemma, ner, parse, dcoref");
        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
        Annotation doc = new Annotation(corpusText);
        pipeline.annotate(doc);

        Map<Integer, CorefChain> corefs = doc.get(CorefCoreAnnotations.CorefChainAnnotation.class);
        List<CoreMap> sentences = doc.get(CoreAnnotations.SentencesAnnotation.class);

        List<String> resolved = new ArrayList<String>();
        List<Integer> culsters_inUse = new ArrayList<Integer>();
        String remainText = text;

        for (CoreMap sentence : sentences) {
            if (remainText.isEmpty()) {
                break;
            }
            if (remainText.startsWith(sentence.toString())) {
                remainText = remainText.substring(sentence.toString().length()).trim();
            } else {
                remainText = text;
                resolved = new ArrayList<String>();
                culsters_inUse = new ArrayList<Integer>();
                continue;
            }

            List<CoreLabel> tokens = sentence.get(CoreAnnotations.TokensAnnotation.class);

            for (CoreLabel token : tokens) {

                Integer corefClustId = token.get(CorefCoreAnnotations.CorefClusterIdAnnotation.class);
                System.out.println(token.word() + " --> corefClusterID = " + corefClustId);

                if (culsters_inUse.contains(corefClustId)) {
                    System.out.println("corefClusterID already replaced.");
                    resolved.add(token.word());
                    System.out.println("Adding the same word " + token.word());
                    continue;
                }

                CorefChain chain = corefs.get(corefClustId);
                System.out.println("matched chain = " + chain);

                if (chain == null) {
                    resolved.add(token.word());
                    System.out.println("Adding the same word " + token.word());
                } else {

                    int sentINdx = chain.getRepresentativeMention().sentNum - 1;
                    System.out.println("sentINdx :" + sentINdx);
                    CoreMap corefSentence = sentences.get(sentINdx);
                    List<CoreLabel> corefSentenceTokens = corefSentence.get(TokensAnnotation.class);
                    String newwords = "";
                    CorefMention reprMent = chain.getRepresentativeMention();
                    System.out.println("reprMent :" + reprMent);
                    System.out.println("Token index " + token.index());
                    System.out.println("Start index " + reprMent.startIndex);
                    System.out.println("End Index " + reprMent.endIndex);
                    if (token.index() <= reprMent.startIndex || token.index() >= reprMent.endIndex) {

                        for (int i = reprMent.startIndex; i < reprMent.endIndex; i++) {
                            CoreLabel matchedLabel = corefSentenceTokens.get(i - 1);
                            resolved.add(matchedLabel.word().replace("'s", ""));
                            System.out.println("matchedLabel : " + matchedLabel.word());
                            newwords += matchedLabel.word() + " ";

                        }
                    } else {
                        resolved.add(token.word());
                        System.out.println("token.word() : " + token.word());
                    }

                    System.out.println("converting " + token.word() + " to " + newwords);

                    culsters_inUse.add(corefClustId);
                }

                System.out.println();
                System.out.println();
                System.out.println("-----------------------------------------------------------------");

            }

        }

        String resolvedStr = "";
        System.out.println();
        for (String str : resolved) {
            resolvedStr += str + " ";
        }
        System.out.println(resolvedStr);
        return resolvedStr;
    }

    public static String replaceCoreferences(String corpusText, String text) {
        Properties props = new Properties();
        props.put("annotators", "tokenize, ssplit, pos, lemma, ner, parse, dcoref");
        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
        Annotation doc = new Annotation(corpusText);
        pipeline.annotate(doc);

        Map<Integer, CorefChain> corefs = doc.get(CorefCoreAnnotations.CorefChainAnnotation.class);
        List<CoreMap> sentences = doc.get(CoreAnnotations.SentencesAnnotation.class);

        Annotation text_doc = new Annotation(text);
        pipeline.annotate(text_doc);
        List<CoreMap> text_sentences = text_doc.get(CoreAnnotations.SentencesAnnotation.class);
        int nextTextSentenceIndex = 0;

        List<String> resolved = new ArrayList<String>();
        List<Integer> culsters_inUse = new ArrayList<Integer>();

        for (CoreMap sentence : sentences) {
            if (nextTextSentenceIndex >= text_sentences.size()) {
                break;
            }
            if (sentence.toString().equals(text_sentences.get(nextTextSentenceIndex).toString())) {
                nextTextSentenceIndex++;
            } else {
                nextTextSentenceIndex = 0;
                resolved = new ArrayList<String>();
                culsters_inUse = new ArrayList<Integer>();
                continue;
            }

            List<CoreLabel> tokens = sentence.get(CoreAnnotations.TokensAnnotation.class);

            for (CoreLabel token : tokens) {

                Integer corefClustId = token.get(CorefCoreAnnotations.CorefClusterIdAnnotation.class);
                System.out.println(token.word() + " --> corefClusterID = " + corefClustId);

                if (culsters_inUse.contains(corefClustId)) {
                    System.out.println("corefClusterID already replaced.");
                    resolved.add(token.word());
                    System.out.println("Adding the same word " + token.word());
                    continue;
                }

                CorefChain chain = corefs.get(corefClustId);
                System.out.println("matched chain = " + chain);

                if (chain == null) {
                    resolved.add(token.word());
                    System.out.println("Adding the same word " + token.word());
                } else {

                    int sentINdx = chain.getRepresentativeMention().sentNum - 1;
                    System.out.println("sentINdx :" + sentINdx);
                    CoreMap corefSentence = sentences.get(sentINdx);
                    List<CoreLabel> corefSentenceTokens = corefSentence.get(TokensAnnotation.class);
                    String newwords = "";
                    CorefMention reprMent = chain.getRepresentativeMention();
                    System.out.println("reprMent :" + reprMent);
                    System.out.println("Token index " + token.index());
                    System.out.println("Start index " + reprMent.startIndex);
                    System.out.println("End Index " + reprMent.endIndex);
                    if (token.index() <= reprMent.startIndex || token.index() >= reprMent.endIndex) {

                        for (int i = reprMent.startIndex; i < reprMent.endIndex; i++) {
                            CoreLabel matchedLabel = corefSentenceTokens.get(i - 1);
                            resolved.add(matchedLabel.word().replace("'s", ""));
                            System.out.println("matchedLabel : " + matchedLabel.word());
                            newwords += matchedLabel.word() + " ";

                        }

                    } else {
                        newwords=token.word();
                        resolved.add(token.word());
                        System.out.println("token.word() : " + token.word());
                    }

                    System.out.println("converting " + token.word() + " to " + newwords);

                    culsters_inUse.add(corefClustId);
                }

                System.out.println();
                System.out.println();
                System.out.println("-----------------------------------------------------------------");

            }

        }

        String resolvedStr = "";
        System.out.println();
        for (String str : resolved) {
            resolvedStr += str + " ";
        }
        System.out.println(resolvedStr);
        return resolvedStr;
    }
}
