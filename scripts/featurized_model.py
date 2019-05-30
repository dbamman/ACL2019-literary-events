import os, sys
import random
import numpy as np
from scipy import sparse
from sklearn import linear_model
from sklearn.metrics import f1_score
import spacy
import nltk
from nltk.corpus import wordnet as wn
import event_reader, event_eval

nlp = spacy.load('en')

train_folder = "../data/bert/train"
dev_folder = "../data/bert/dev"
test_folder = "../data/bert/test"

count_yes={}
uncount_yes={}

embeddings={}

def read_countability(filename):
	with open(filename) as file:
		for line in file:
			cols=line.rstrip().split("\\")
			word=cols[1]
			if len(word.split(" ")) > 1:
				continue
			val=None
			cat=int(cols[3])

			# celex cat 1 = noun
			if cat != 1:
				continue

			count=cols[4]
			uncount=cols[5]
			
			if count == "Y":
				count_yes[word]=1
			if uncount == "Y":
				uncount_yes[word]=1

def read_embeddings(filename):
	with open(filename) as file:
		for line in file:
			cols=line.rstrip().split(" ")
			word=cols[0]
			embedding=[float(x) for x in cols[1:]]
			embeddings[word]=embedding

def get_pos(idx, spacy_tokens):
	feats={}
	tag=spacy_tokens[idx].tag_
	feats["POS_%s" % tag]=1

	return feats

def get_embedding(idx, spacy_tokens):
	feats={}
	word=spacy_tokens[idx].text
	if word in embeddings:
		for idx,f in enumerate(embeddings[word]):
			feats["EMB_%s" % idx]=f
	return feats

def lookup_wordnet(word):
	cat=None
	if word.tag_.startswith("V"):
		cat=wn.VERB
	if word.tag_.startswith("N"):
		cat=wn.NOUN
	if word.tag_.startswith("J"):
		cat=wn.ADJ
	if word.tag_.startswith("R"):
		cat=wn.ADV
	
	if cat != None:
		synsets=wn.synsets(word.lemma_, pos=cat)

		return synsets
	return None

def get_wordnet(idx, spacy_tokens):
	feats={}
	word=spacy_tokens[idx]
	synsets=lookup_wordnet(word)
	if synsets != None:
		for synset in synsets:
			feats[synset]=1

			hypernyms=synset.hypernyms()

			for hypernym in hypernyms[:3]:
				feats["HYP_%s" % hypernym]=1

	return feats

def get_context(idx, spacy_tokens):
	feats={}
	word=spacy_tokens[idx]

	word1=word2=word3=""
	pos1=pos2=pos3=""

	if idx > 0:
		orig3=spacy_tokens[idx-1]
		word3=orig3.lemma_
		pos3=orig3.tag_.lower()
	if idx > 1:
		orig2=spacy_tokens[idx-2]
		word2=orig3.lemma_		
		pos2=orig2.tag_.lower()
	if idx > 2:
		orig1=spacy_tokens[idx-1]
		word1=orig3.lemma_
		pos1=orig1.tag_.lower()

	feats["L_CONTEXT_%s_%s_%s" % (str(word1).lower(), str(word2).lower(), str(word3).lower())]=1
	feats["L_CONTEXT_POS_%s_%s_%s" % (pos1.lower(), pos2.lower(), pos3.lower())]=1

	feats["L_CONTEXT_BAG_%s" % str(word1).lower()]=1
	feats["L_CONTEXT_BAG_%s" % str(word2).lower()]=1
	feats["L_CONTEXT_BAG_%s" % str(word3).lower()]=1
	
	feats["L_CONTEXT_1_%s" % str(word1).lower()]=1
	feats["L_CONTEXT_2_%s" % str(word2).lower()]=1
	feats["L_CONTEXT_3_%s" % str(word3).lower()]=1
	

	word1=word2=word3=""
	pos1=pos2=pos3=""

	if idx < len(spacy_tokens) -1:
		orig1=spacy_tokens[idx+1]
		word1=orig1.lemma_
		pos1=orig1.tag_.lower()

	if idx < len(spacy_tokens) -2:
		orig2=spacy_tokens[idx+2]
		word2=orig2.lemma_
		pos1=orig2.tag_.lower()
	if idx < len(spacy_tokens) -3:
		orig3=spacy_tokens[idx+3]
		word3=orig3.lemma_
		pos3=orig3.tag_.lower()

	feats["R_CONTEXT_%s_%s_%s" % (str(word1).lower(), str(word2).lower(), str(word3).lower())]=1
	feats["R_CONTEXT_POS_%s_%s_%s" % (pos1.lower(), pos2.lower(), pos3.lower())]=1

	feats["R_CONTEXT_BAG_%s" % str(word1).lower()]=1
	feats["R_CONTEXT_BAG_%s" % str(word2).lower()]=1
	feats["R_CONTEXT_BAG_%s" % str(word3).lower()]=1
	
	feats["R_CONTEXT_1_%s" % str(word1).lower()]=1
	feats["R_CONTEXT_2_%s" % str(word2).lower()]=1
	feats["R_CONTEXT_3_%s" % str(word3).lower()]=1

	return feats

def get_nsubj_features(idx, spacy_tokens):

	""" 
	from Reiter and Frank 2010; Suh 2006; includes lots of info about the syntactic subject:
		-- bare plurals ("dogs like ice creams")
		-- word
		-- pos
		-- determiner type (a, the, etc.)
		-- countability
		-- wordnet synset
	"""
	feats={}
	word=spacy_tokens[idx]

	for child in word.children:
		if child.dep_ == "nsubj":
			feats["NSUBJ_%s" % child.lemma_]=1
			feats["NSUBJ_POS_%s" % child.tag_]=1

			if child.lemma_ in count_yes and child.lemma_ in uncount_yes:
				feats["COUNTABILITY_AMBIG"]=1
			elif child.lemma_ in count_yes:
				feats["COUNTABILITY_COUNT"]=1
			elif child.lemma_ in uncount_yes:
				feats["COUNTABILITY_UNCOUNT"]=1	

			feats["NSUBJ_POS_%s" % child.tag_]=1
			synsets=lookup_wordnet(child)
			if synsets != None:
				for synset in synsets:
					feats["NSUBJ_WORDNET_%s" % synset]=1

			for grandchild in child.children:
				if grandchild.tag_ == "DT":
					feats["NSUBJ_DT_%s" % grandchild.lemma_]=1

			if (child.tag_ == "NNS" or child.tag_ == "NNPS"):
				flag=False
				for grandchild in child.children:
					if grandchild.tag_ == "DT" or grandchild.tag_ == "PRP$" or grandchild.tag_ == "CD":
						flag=True
				if not flag:
					feats["BARE_PLURAL_NSUBJ"]=1
	return feats


def get_dep(idx, spacy_tokens):
	feats={}
	word=spacy_tokens[idx]
	deprel=word.dep_.lower()
	head=word.head

	
	feats["DEPREL_HEAD_%s_%s" % (deprel, head.text)]=1
	feats["HEAD_%s" % (head.text)]=1
	feats["DEPREL%s" % deprel]=1
	feats["HEAD_TAG%s" % head.tag_]=1

	return feats

def get_lemma(idx, spacy_tokens):
	feats={}
	word=spacy_tokens[idx]

	feats["LEMMA_%s" % word.lemma_.lower()]=1

	return feats

def get_word(idx, spacy_tokens):
	feats={}
	word=spacy_tokens[idx].text

	feats["UNIGRAM_%s" % word.lower()]=1

	return feats

def featurize(idx, spacy_tokens, feature_functions):
	feats={}
	
	for function in feature_functions:
		feats.update(function(idx, spacy_tokens))

	return feats

def create_feature_vocab(all_feats):
	feature_vocab={}
	idx=0
	for k in all_feats:
		feature_vocab[k]=idx
		idx+=1
	return feature_vocab

def featurize_data(data, feature_functions, train=False):
	all_feats={}

	featurized=[]

	for s_idx, sentence in enumerate(data):
		if s_idx % 1000 == 0:
			print(s_idx)

		tokens_list=[word[0] for word in sentence]
		tokens = nlp.tokenizer.tokens_from_list(tokens_list)
		nlp.tagger(tokens)
		nlp.parser(tokens)

		for idx, (_, label, _, _, _) in enumerate(sentence):
			feats=featurize(idx, tokens, feature_functions)
			if train:
				all_feats.update(feats)

			featurized.append((feats, label))

	return featurized, all_feats


def create_sparse(raw_data, vocab):
	V=len(vocab)
	X=sparse.lil_matrix((len(raw_data), V))
	Y=[]
	for idx,(feats, label) in enumerate(raw_data):
		for f in feats:
			if f in vocab:
				X[idx,vocab[f]]=feats[f]
		Y.append(label)

	return X, Y

def train_eval(trainSentences, devSentences, testSentences, outputFile, feature_functions):

	featurized_train, trainingFeats=featurize_data(trainSentences, feature_functions, train=True)
	featurized_dev, _=featurize_data(devSentences, feature_functions, train=False)
	featurized_test, _=featurize_data(testSentences, feature_functions, train=False)

	feature_vocab=create_feature_vocab(trainingFeats)

	train_X, train_Y=create_sparse(featurized_train, feature_vocab)
	dev_X, dev_Y=create_sparse(featurized_dev, feature_vocab)
	test_X, test_Y=create_sparse(featurized_test, feature_vocab)

	Cs=[0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10]

	bestModel=None
	bestC=None
	bestF1=0
	for C in Cs:
		logreg = linear_model.LogisticRegression(C=C, solver='lbfgs', penalty='l2', max_iter=1000)
		logreg.fit(train_X, train_Y)
		preds=logreg.predict(dev_X)

		# select regularization strength based on F1 for dev data
		f1=f1_score(dev_Y, preds)
		if f1 > bestF1:
			bestF1=f1
			bestModel=logreg
			bestC=C
		print ("Dev F1: %.3f, C: %s" % (f1, C))

	print(bestModel)
	final_test_preds=bestModel.predict(test_X)

	out=open(outputFile, "w", encoding="utf-8")
	for idx, pred in enumerate(final_test_preds):
		out.write("%s\t%s\n" % ('\t'.join([str(x) for x in test_metadata[idx]]), pred))
	out.close()



if __name__ == "__main__":

	read_countability("../data/esl.cd")
	read_embeddings("../data/guten.vectors.txt")

	trainSentences, _ = event_reader.prepare_annotations_from_folder(train_folder)
	devSentences, _ = event_reader.prepare_annotations_from_folder(dev_folder)
	testSentences, _ = event_reader.prepare_annotations_from_folder(test_folder)

	test_metadata=event_reader.convert_to_index(testSentences)

	functions=[get_word, get_context, get_dep, get_lemma, get_wordnet, get_pos, get_nsubj_features, get_embedding]
	train_eval(trainSentences, devSentences, testSentences, "../results/featurized.preds.txt", functions)

	functions=[get_context, get_dep, get_lemma, get_wordnet, get_pos, get_nsubj_features, get_embedding]
	train_eval(trainSentences, devSentences, testSentences, "../results/featurized.word_ablation.preds.txt", functions)

	functions=[get_word, get_context, get_dep, get_wordnet, get_pos, get_nsubj_features, get_embedding]
	train_eval(trainSentences, devSentences, testSentences, "../results/featurized.lemma_ablation.preds.txt", functions)

	functions=[get_word, get_context, get_dep, get_lemma, get_wordnet, get_nsubj_features, get_embedding]
	train_eval(trainSentences, devSentences, testSentences, "../results/featurized.pos_ablation.preds.txt", functions)

	functions=[get_word, get_dep, get_lemma, get_wordnet, get_pos, get_nsubj_features, get_embedding]
	train_eval(trainSentences, devSentences, testSentences, "../results/featurized.context_ablation.preds.txt", functions)

	functions=[get_word, get_context, get_lemma, get_wordnet, get_pos, get_nsubj_features, get_embedding]
	train_eval(trainSentences, devSentences, testSentences, "../results/featurized.syntax_ablation.preds.txt", functions)

	functions=[get_word, get_context, get_dep, get_lemma, get_pos, get_nsubj_features, get_embedding]
	train_eval(trainSentences, devSentences, testSentences, "../results/featurized.wordnet_ablation.preds.txt", functions)

	functions=[get_word, get_context, get_dep, get_lemma, get_wordnet, get_pos, get_nsubj_features]
	train_eval(trainSentences, devSentences, testSentences, "../results/featurized.embedding_ablation.preds.txt", functions)

	functions=[get_word, get_context, get_dep, get_lemma, get_wordnet, get_pos, get_embedding]
	train_eval(trainSentences, devSentences, testSentences, "../results/featurized.bare_plurals_ablation.preds.txt", functions)
