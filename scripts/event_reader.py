import os
import numpy as np

def load_embeddings(filename, vocab_size, word_embedding_dim):
	# 0 is padding value so start with 1 (for UNK)

	vocab={}
	embeddings=np.zeros((vocab_size, word_embedding_dim))

	with open(filename, encoding="utf-8") as file:
		for idx,line in enumerate(file):
			# start with 2; 0 for zero padding, 1 for _UNK_

			if idx + 2 >= vocab_size:
				break

			cols=line.rstrip().split(" ")
			val=np.array(cols[1:])
			word=cols[0]
			embeddings[idx+2]=val
			vocab[word]=idx+2

	return embeddings, vocab

def read_annotations(filename, documents_are_sequences=False, useBERT=False):

	""" Read tsv data and return sentences and [word, tag, sentenceID, filename] list """

	with open(filename, encoding="utf-8") as f:
		sentence = []
		sentences = []
		sentenceID=0
		for line in f:
			if len(line) > 0:
				if line == '\n':
					sentenceID+=1
	
					if documents_are_sequences == False:	# each individual sentences is its own sequence
						sentences.append(sentence)
						sentence = []
					else:
						continue

				else:
					data=[]
					split_line = line.rstrip().split('\t')

					data.append(split_line[0])
					data.append(1 if split_line[1] == "EVENT" else 0)

					data.append(sentenceID)
					data.append(filename)

					if useBERT:

						bert=np.array(split_line[2].split(" "), dtype=float)
						data.append(bert)
					else:
						data.append(None)

					sentence.append(data)
		
		if len(sentence) > 0:
			sentences.append(sentence)

	return sentences



def convert_to_index(sentences):

	""" Index which sentences come from which books """

	words=[]
	for idx, sentence in enumerate(sentences):
		for word, label, sid, book, bert in sentence:
			words.append([book, sid, word, label])

	return words

def prepare_annotations_from_file(filename, documents_are_sequences=False, useBERT=False):

	""" Read a single file of annotations, returning:
		-- a list of sentences, each a list of [word, label, sentenceID, filename]
		-- a dict specifying the start and end sentences for the file
	"""

	sentences = []
	book_index = {}
	total_sentence_count = 0
	annotations = read_annotations(filename, documents_are_sequences, useBERT)
	sentences += annotations
	book_sentence_count = len(annotations)
	book_index[filename] = [total_sentence_count, total_sentence_count + book_sentence_count]
	total_sentence_count += book_sentence_count
	return sentences,book_index

def prepare_annotations_from_folder(folder, documents_are_sequences=False, useBERT=False):

	""" Read folder of annotations, returning:
		-- a list of sentences, each a list of [word, label, sentenceID, filename]
		-- a dict specifying the start and end sentences for each file
	"""

	sentences = []
	book_index = {}
	total_sentence_count = 0
	for filename in os.listdir(folder):
		if filename.endswith(".tsv"):
			annotations = read_annotations(os.path.join(folder,filename), documents_are_sequences, useBERT)
			sentences += annotations
			book_sentence_count = len(annotations)
			book_index[filename] = [total_sentence_count, total_sentence_count + book_sentence_count]
			total_sentence_count += book_sentence_count
	print("num sentences: %s" % len(sentences))
	return sentences,book_index
