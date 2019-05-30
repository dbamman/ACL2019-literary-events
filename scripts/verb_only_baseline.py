import sys,os
import spacy
import event_eval
import event_reader

if __name__ == "__main__":

	outputFile=sys.argv[1]

	nlp = spacy.load('en', disable=['ner,parser'])
	nlp.remove_pipe('ner')
	nlp.remove_pipe('parser')

	train_folder = "../data/bert/train"
	dev_folder = "../data/bert/dev"
	test_folder = "../data/bert/test"

	testSentences, _ = event_reader.prepare_annotations_from_folder(test_folder)
	test_metadata=event_reader.convert_to_index(testSentences)

	golds=[]
	preds=[]

	for sentence in testSentences:
		tokens_list=[word[0] for word in sentence]
		tokens = nlp.tokenizer.tokens_from_list(tokens_list)
		nlp.tagger(tokens)
		for idx, token in enumerate(tokens):
			pred=0
			if token.tag_.startswith("V"):
				pred=1
			preds.append(pred)
			label=sentence[idx][1]
			golds.append(label)

	out=open(outputFile, "w", encoding="utf-8")
	for idx, pred in enumerate(preds):
		out.write("%s\t%s\n" % ('\t'.join([str(x) for x in test_metadata[idx]]), pred))
	out.close()


