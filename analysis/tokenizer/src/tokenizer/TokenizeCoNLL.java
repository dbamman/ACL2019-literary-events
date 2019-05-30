package tokenizer;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.List;
import java.util.Properties;

import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;

public class TokenizeCoNLL {

	public static String readText(String infile) {
		BufferedReader in1;
		StringBuffer buffer = new StringBuffer();

		try {
			in1 = new BufferedReader(new InputStreamReader(new FileInputStream(
					infile), "UTF-8"));
			String str1;

			while ((str1 = in1.readLine()) != null) {
				try {
					buffer.append(str1.trim() + " ");

				} catch (Exception e) {
					e.printStackTrace();
				}
				
				
			}
			in1.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
		return buffer.toString();
	}

	public static void process(String doc, String outFile) throws Exception {

		StanfordCoreNLP pipeline;
		Annotation document = new Annotation(doc);
		Properties props = new Properties();
		props.put("annotators", "tokenize, ssplit");

		pipeline = new StanfordCoreNLP(props);
		pipeline.annotate(document);

		List<CoreMap> sentences = document.get(SentencesAnnotation.class);
		try {
			OutputStreamWriter out = new OutputStreamWriter(
					new FileOutputStream(outFile), "UTF-8");

			for (CoreMap sentence : sentences) {

				String text = "";
				for (CoreLabel token : sentence.get(TokensAnnotation.class)) {

					String whitespaceAfter = token.after();
					String original = token.originalText();
					text += original + whitespaceAfter;
					out.write(token.originalText() + "\tO\n");
				}
				out.write("\n");
			}
			out.close();
		} catch (Exception e) {
			e.printStackTrace();
		}

	}

	public static void main(String[] args) throws Exception {
		String path=args[0];
		String output=args[1];
		
		String doc = readText(path);
		process(doc, output);
	}
}
