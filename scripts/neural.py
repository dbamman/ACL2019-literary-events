import sys, re, os, math, argparse
from time import time
import numpy as np
from numpy import array
from keras.models import Model
from keras.layers import Embedding, Concatenate
from keras.layers import Dense, Input, Masking, Dropout, TimeDistributed, LSTM, Bidirectional
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, Callback
import event_reader, event_eval


train_folder = "../data/bert/train"
dev_folder = "../data/bert/dev"
test_folder = "../data/bert/test"

embeddings={}
vocab_size=100000
word_embedding_dim=100
pos_embedding_dim=5
batch_size=32
window=20
lstm_size=100
feng_cnn_filters=100

max_character_length=52
char_embedding_dim=50

num_epochs=1000
devC = devX = devP = devW = devY = devL = devL = None

BERT_SIZE=3072

# https://medium.com/@thongonary/how-to-compute-f1-score-for-each-epoch-in-keras-a1acd17715a2
# updated for keras 2.0
class BatchF1(Callback):
	def on_train_begin(self, logs={}):
		self.val_f1s = []
		self.val_recalls = []
		self.val_precisions = []
	def on_epoch_end(self, epoch, logs={}):
		predictions=np.asarray(self.model.predict([self.validation_data[0], self.validation_data[1], self.validation_data[2], self.validation_data[3]])).reshape(-1)
		val_predict = (predictions).round()
		truth=np.array(self.validation_data[4])

		val_targ = truth.reshape(-1)

		_val_f1, _val_precision, _val_recall, _, _, _=event_eval.check_f1_two_lists(val_targ, val_predict)
		self.val_f1s.append(_val_f1)
		self.val_f1s.append(_val_precision)
		self.val_f1s.append(_val_recall)

		print ("— val_f1: %f — val_precision: %f — val_recall %f" %(_val_f1, _val_precision, _val_recall))
		return

class GeneratorF1(Callback):
	def on_train_begin(self, logs={}):
		self.val_f1s = []
		self.val_recalls = []
		self.val_precisions = []
	def on_epoch_end(self, epoch, logs={}):

		preds=[]
		gold=[]
		metric_dev_generator = single_generator(devC, devX, devP, devW, devY, devL)

		for step in range(len(devL)):
			batch, y=next(metric_dev_generator)

			probs=self.model.predict_on_batch(batch)

			_, length, _=y.shape
			for i in range(length):
				preds.append(probs[0][i][0] >= 0.5)
				gold.append(y[0][i][0])

		_val_f1, _val_precision, _val_recall, _, _, _=event_eval.check_f1_two_lists(gold, preds)

		self.val_f1s.append(_val_f1)
		self.val_f1s.append(_val_precision)
		self.val_f1s.append(_val_recall)

		print ("— val_f1: %f — val_precision: %f — val_recall %f" %(_val_f1, _val_precision, _val_recall))
		return


def pad_length(data, max_length, zero_val):

	words=[]
	for word in data:
		words.append(word)
	
	for i in range(max_length-len(words)):
		words.append(zero_val)

	words=words[:max_length]

	return np.array(words)

def char_cnn():

	char_cnn_input = Input(shape=(max_character_length, char_embedding_dim))
	cnn2=Conv1D(filters=25, kernel_size=2, strides=1, padding="same", activation="relu")(char_cnn_input)
	cnn3=Conv1D(filters=25, kernel_size=3, strides=1, padding="same", activation="relu")(char_cnn_input)
	cnn4=Conv1D(filters=25, kernel_size=4, strides=1, padding="same", activation="relu")(char_cnn_input)
	cnn5=Conv1D(filters=25, kernel_size=5, strides=1, padding="same", activation="relu")(char_cnn_input)
	maxpool2=GlobalMaxPooling1D()(cnn2)
	maxpool3=GlobalMaxPooling1D()(cnn3)
	maxpool4=GlobalMaxPooling1D()(cnn4)
	maxpool5=GlobalMaxPooling1D()(cnn5)
	
	concat=Concatenate()([maxpool2, maxpool3, maxpool4, maxpool5])
	model = Model(inputs=[char_cnn_input], outputs=concat)
	return model

def sentence_cnn():

	embedding_concat=Input(shape=(None,word_embedding_dim+pos_embedding_dim))

	cnn2=Conv1D(filters=feng_cnn_filters, kernel_size=2, strides=1, padding="same", activation="tanh")(embedding_concat)
	cnn3=Conv1D(filters=feng_cnn_filters, kernel_size=3, strides=1, padding="same", activation="tanh")(embedding_concat)

	maxpool2=GlobalMaxPooling1D()(cnn2)
	maxpool3=GlobalMaxPooling1D()(cnn3)

	concat=Concatenate()([maxpool2, maxpool3])

	model = Model(inputs=[embedding_concat], outputs=concat)

	return model


def event_cnn(embeddings, sentenceCNN, charCNN, uniLSTM, useBERT):

	word_character_sequence_input = Input(shape=(None,max_character_length), dtype='int32')

	word_sequence_input = Input(shape=(None,window*2+1,), dtype='int32')
	position_sequence_input = Input(shape=(None,window*2+1,), dtype='int32')

	cnn_word_embedding_layer = Embedding(vocab_size,
									word_embedding_dim,
									weights=[embeddings],
									trainable=False, mask_zero=False)

	word_embedding_layer = Embedding(vocab_size,
									word_embedding_dim,
									weights=[embeddings],
									trainable=False, mask_zero=True)
	position_embedding_layer = Embedding(100,
									pos_embedding_dim,
									trainable=True)

	char_embedding_layer = Embedding(100,
									char_embedding_dim,
									trainable=True)

	char_embedded_sequences = char_embedding_layer(word_character_sequence_input)
	char_cnn_model=char_cnn()
	char_cnn_output = TimeDistributed(char_cnn_model)(char_embedded_sequences)

	char_cnn_output = Dropout(0.5)(char_cnn_output)

	
	word_embeddings=cnn_word_embedding_layer(word_sequence_input)
	position_embeddings=position_embedding_layer(position_sequence_input)
	sentence_concat=Concatenate()([word_embeddings, position_embeddings])

	sentence_cnn_model=sentence_cnn()

	sentence_cnn_output=TimeDistributed(sentence_cnn_model)(sentence_concat)

	orig_embeddings = None
	if useBERT:
		orig_sequence_input = Input(shape=(None,BERT_SIZE), dtype='float')
		orig_embeddings = Masking(mask_value=0.0)(orig_sequence_input)
	else:
		orig_sequence_input = Input(shape=(None,), dtype='int32')
		orig_embeddings=word_embedding_layer(orig_sequence_input)

	embedded_sequences=None

	# input to LSTM is word embeddings + character CNN output
	if charCNN:
		embedded_sequences=Concatenate()([char_cnn_output,orig_embeddings])
	else:
		embedded_sequences=orig_embeddings

	lstm = None

	if uniLSTM:
		lstm = LSTM(lstm_size, return_sequences=True, recurrent_dropout=0.25, dropout=0.5)(embedded_sequences)
	else:
		lstm = Bidirectional(LSTM(lstm_size, return_sequences=True, recurrent_dropout=0.25, dropout=0.5), merge_mode='concat')(embedded_sequences)

	concat=None

	# make a prediction based on output of LSTM and entire sequence CNN
	if sentenceCNN:
		concat=Concatenate()([sentence_cnn_output,lstm])
	else:
		concat=lstm

	preds=Dense(1, activation='sigmoid')(concat)
	model = Model(inputs=[word_character_sequence_input,orig_sequence_input, word_sequence_input, position_sequence_input], outputs=preds)

	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
	
	print (model.summary())
	return model


def get_distance(one, two):
	distance=one-two
	if distance < -5 and distance >= -10:
		distance=-5
	elif distance < -10 and distance >= -20:
		distance=-6
	elif distance < -20:
		distance=-7

	if distance > 5 and distance <= 10:
		distance=5
	elif distance > 10 and distance <= 20:
		distance=6
	elif distance > 20:
		distance=7

	return distance+7	

def get_char_vocab():
	char2Idx = {"PADDING":0, "UNKNOWN":1}
	for c in " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-_()[]{}!?:;#'\"/\\%$`&=*+@^~|":
		char2Idx[c] = len(char2Idx)
	
	return char2Idx

def transform_examples(data, vocab, useBERT):

	""" Transform data from list of [word, label, sentenceID, filename] lists to:
		-- list of subword characters 
		-- words in window around target
		-- positions of words in window around target
		-- target word
		-- labels
		-- sentence lengths
	"""

	maxlen=0
	char_vocab=get_char_vocab()

	X=[]
	positions=[]
	Y=[]
	orig=[]
	all_chars=[]
	lengths=[]
	
	for sentence in data:
		
		lengths.append(len(sentence))
		words=[]
		if len(sentence) > maxlen:
			maxlen=len(sentence)
		sentence_chars=[]

		labels=[]

		for w,label, _, _, bert in sentence:

			## get subword character list

			o = w.replace( u'\u2018', u"'").replace( u'\u2019', u"'").replace( u'\u201c', u'"').replace( u'\u201d', u'"')
			chars=list(o)

			char_ids=[]
			for char in chars:
				char_id = char_vocab["UNKNOWN"]
				if char in char_vocab:
					char_id=char_vocab[char]

				char_ids.append(char_id)

			char_ids=char_ids[:max_character_length]
			for idx in range(len(char_ids), max_character_length):
				char_ids.append(0)
			
			sentence_chars.append(char_ids)

			## add target word id

			if useBERT:
				# print(bert)
				words.append(bert)
			else:
				if w.lower() in vocab:
					words.append(vocab[w.lower()])
				else:
					words.append(1)

			# add label for target

			labels.append(label)

		## for sentence CNN, get words in window around target, along with their position IDs

		X_s=[]
		P_s=[]

		for idx,(word,_, _, _, _) in enumerate(sentence):

			pos=[]
			pwords=[]

			start=max(idx-window,0)
			end=min(idx+window, len(sentence))

			for idx2 in range(start, end):
				distance=get_distance(idx, idx2)
				
				pos.append(distance)
				ww,_, _, _, _=sentence[idx2]
				ww=ww.lower()
				if ww in vocab:
					pwords.append(vocab[ww])
				else:
					pwords.append(1)

			P_s.append(pad_length(pos, window*2+1,99))
			X_s.append(pad_length(pwords, window*2+1,0))

		# append sentence information to total dataset
		all_chars.append(sentence_chars)
		X.append(X_s)
		positions.append(P_s)
		orig.append(words)
		Y.append(labels)

	print("Max sentence length: %s" % maxlen)

	return all_chars, X, positions, orig, Y, lengths

def pad_all(all_chars, X, positions, orig, Y, lengths, max_sequence_length, useBERT):
	print("Using bert: ", useBERT)
	for idx, length in enumerate(lengths):

		if length > max_sequence_length:
			print ("%s is longer than max length %s" % (length, max_sequence_length))
			sys.exit(1)
		for i in range(length, max_sequence_length):
			all_chars[idx].append(np.zeros(max_character_length))
			X[idx].append(np.zeros(window*2+1))
			positions[idx].append(np.zeros(window*2+1))
			if useBERT:
				orig[idx].append(np.zeros(BERT_SIZE))
			else:
				orig[idx].append(0)

			Y[idx].append(0)

	Y=np.array(Y)
	Y=np.expand_dims(Y, axis=-1)
	print("WORDS: ", np.array(orig).shape)
	return np.array(all_chars), np.array(X), np.array(positions), np.array(orig), np.array(Y), lengths

def single_generator(C, X, P, W, Y, L):
	while True:
		for idx, length in enumerate(L):
			thisC=C[idx][:length]	# sequence of sequence of char ids (one per word)
			thisX=X[idx][:length]	# sequence of sequence of word ids
			thisP=P[idx][:length]	# sequence of sequence of word positions
			thisW=W[idx][:length]	# sequence of word ids
			thisY=Y[idx][:length]	# sequence of word labels (0/1)

			thisY=np.array(thisY)
			thisY=np.expand_dims(thisY, axis=-1)
			thisY=np.expand_dims(thisY, axis=0)

			thisC=np.array(thisC)
			thisC=np.expand_dims(thisC, axis=0)
			
			thisW=np.array(thisW)
			thisW=np.expand_dims(thisW, axis=0)

			thisX=np.array(thisX)
			thisX=np.expand_dims(thisX, axis=0)

			thisP=np.array(thisP)
			thisP=np.expand_dims(thisP, axis=0)

			yield ([thisC, thisW, thisX, thisP], thisY)

def predict_file_batch(filename, embeddingsFile, existingModel, predictionsFile, documents_are_sequences, sentenceCNN, charCNN, pad, uniLSTM, useBERT):
	embeddings, vocab=event_reader.load_embeddings(embeddingsFile, vocab_size, word_embedding_dim)

	testSentences, testBookIndex = event_reader.prepare_annotations_from_file(filename, documents_are_sequences=documents_are_sequences, useBERT=useBERT)
	
	testC, testX, testP, testW, testY, testL = transform_examples(testSentences, vocab, useBERT)

	test_metadata=event_reader.convert_to_index(testSentences)

	max_sequence_length=0
	for length in testL:
		if length > max_sequence_length:
			max_sequence_length=length
	
	print("max l: ", max_sequence_length)
	model = event_cnn(embeddings, sentenceCNN, charCNN, uniLSTM, useBERT)

	model.load_weights(existingModel)
	predictionFile=predictionsFile
	out=open(predictionFile, "w", encoding="utf-8")

	if pad:
		testC, testX, testP, testW, testY, testL=pad_all(testC, testX, testP, testW, testY, testL, max_sequence_length, useBERT)
		model.predict([testC, testW, testX, testP])

		probs=model.predict([testC, testW, testX, testP], 
						 batch_size=128
					   )
		c=0
		lastSent=None
		for step in range(len(testL)):
			for i in range(testL[step]):
				sid=test_metadata[c][1]
				if lastSent != sid and lastSent != None:
					out.write("\n")

				w_book, w_sid, w_word, _ = test_metadata[c]
				label="O"
				if probs[step][i][0] > 0.5:
					label="EVENT"
				out.write("%s\t%s\n" % (w_word, label))
				lastSent=sid
				c+=1

	else:
		test_generator = single_generator(testC, testX, testP, testW, testY, testL)
	
		c=0
		for step in range(len(testL)):
			batch, y=next(test_generator)

			probs=model.predict_on_batch(batch)

			_, length, _=y.shape

			for i in range(length):
				sid=test_metadata[c][1]
				
				w_book, w_sid, w_word, _ = test_metadata[c]
				label="O"
				if probs[0][i][0] > 0.5:
					label="EVENT"
				out.write("%s\t%s\n" % (w_word, label))

				c+=1
			
			out.write("\n")
			

	out.close()


def test(embeddingsFile, existingModel, predictionsFile, documents_are_sequences, sentenceCNN, charCNN, uniLSTM, useBERT):
	embeddings, vocab=event_reader.load_embeddings(embeddingsFile, vocab_size, word_embedding_dim)

	testSentences, testBookIndex = event_reader.prepare_annotations_from_folder(test_folder, documents_are_sequences, useBERT)
	
	testC, testX, testP, testW, testY, testL = transform_examples(testSentences, vocab, useBERT)

	test_generator = single_generator(testC, testX, testP, testW, testY, testL)

	test_metadata=event_reader.convert_to_index(testSentences)

	model = event_cnn(embeddings, sentenceCNN, charCNN, uniLSTM, useBERT)

	model.load_weights(existingModel)
	predictionFile=predictionsFile
	out=open(predictionFile, "w", encoding="utf-8")
	gold=[]
	preds=[]
	c=0
	for step in range(len(testL)):
		batch, y=next(test_generator)

		probs=model.predict_on_batch(batch)

		_, length, _=y.shape
		for i in range(length):
			out.write("%s\t%s\t%s\t%.20f\n" % ('\t'.join([str(x) for x in test_metadata[c]]), int(probs[0][i][0] > 0.5), y[0][i][0], probs[0][i][0]))

			preds.append(probs[0][i][0] >= 0.5)
			gold.append(y[0][i][0])
			c+=1

	f, p, r, correct, trials, trues=event_eval.check_f1_two_lists(gold, preds)

	print ("precision: %.3f %s/%s" % (p, correct, trials))
	print ("recall: %.3f %s/%s" % (r, correct, trues))
	print ("F: %.3f" % f)
		
	event_eval.check_f1_two_lists(gold, preds)
	out.close()


def train(embeddingsFile, writePath, documents_are_sequences, sentenceCNN, charCNN, pad, uniLSTM, useBERT):

	global devC, devX, devP, devW, devY, devL

	embeddings, vocab=event_reader.load_embeddings(embeddingsFile, vocab_size, word_embedding_dim)

	trainSentences, trainBookIndex = event_reader.prepare_annotations_from_folder(train_folder, documents_are_sequences, useBERT)
	devSentences, devBookIndex = event_reader.prepare_annotations_from_folder(dev_folder, documents_are_sequences, useBERT)
	
	devC, devX, devP, devW, devY, devL = transform_examples(devSentences, vocab, useBERT)
	trainC, trainX, trainP, trainW, trainY, trainL = transform_examples(trainSentences, vocab, useBERT)

	max_sequence_length=0
	for length in trainL:
		if length > max_sequence_length:
			max_sequence_length=length
	for length in devL:
		if length > max_sequence_length:
			max_sequence_length=length

	print("Max length: %s" % max_sequence_length)

	model = event_cnn(embeddings, sentenceCNN, charCNN, uniLSTM, useBERT)


	tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
	checkpoint = ModelCheckpoint("%s.hdf5" % writePath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
	early_stopping = EarlyStopping(monitor='val_loss',
							  min_delta=0,
							  patience=15,
							  verbose=0, mode='auto')


	if pad:
		trainC, trainX, trainP, trainW, trainY, trainL=pad_all(trainC, trainX, trainP, trainW, trainY, trainL, max_sequence_length, useBERT)
		devC, devX, devP, devW, devY, devL=pad_all(devC, devX, devP, devW, devY, devL, max_sequence_length, useBERT)
		
		batchF1=BatchF1()

		model.fit([trainC, trainW, trainX, trainP], trainY, 
						 validation_data=([devC, devW, devX, devP], devY),
						 epochs=num_epochs,
						 batch_size=batch_size,
						 callbacks=[batchF1,tensorboard,checkpoint,early_stopping]
					   )

	else:
		train_generator = single_generator(trainC, trainX, trainP, trainW, trainY, trainL)
		dev_generator = single_generator(devC, devX, devP, devW, devY, devL)
		generatorF1 = GeneratorF1()

		model.fit_generator(train_generator, 
						 steps_per_epoch=len(trainL),
						 validation_data=dev_generator,
						 validation_steps=len(devL),
						 epochs=num_epochs,
						 callbacks=[generatorF1,tensorboard,checkpoint,early_stopping]
					   )

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('-m','--mode', help='{train,test}', required=True)
	parser.add_argument('-e','--embeddings', help='pre-trained embeddings', required=True)
	parser.add_argument('-x','--existingModel', help='trained model', required=False)
	parser.add_argument('-r','--fileToPredict', help='conll file to predict', required=False)
	parser.add_argument('-w','--writePath', help='writePath', required=False)
	parser.add_argument('-p','--predictionsFile', help='predictionsFile', required=False)
	parser.add_argument('-b','--batchsize', help='batchsize', required=False)
	parser.add_argument('-d','--documents_are_sequences', action="store_true", default=False, help='documents_are_sequences', required=False)
	parser.add_argument('-f','--sentenceCNN', action="store_true", default=False, help='include sentence-level CNN', required=False)
	parser.add_argument('-u','--uniLSTM', action="store_true", default=False, help='one-directional LSTM', required=False)
	parser.add_argument('-t','--bert', action="store_true", default=False, help='use pretrained contextual vectors', required=False)
	parser.add_argument('-c','--charCNN', action="store_true", default=False, help='include subword character CNN', required=False)
	parser.add_argument('-a','--pad', action="store_true", default=False, help='pad sequences to fixed length', required=False)

	args = vars(parser.parse_args())
	print(args)

	mode=args["mode"]
	embeddingsFile=args["embeddings"]
	if args["batchsize"] != None:
		batch_size=int(args["batchsize"])
	documents_are_sequences=bool(args["documents_are_sequences"])
	sentenceCNN=bool(args["sentenceCNN"])
	charCNN=bool(args["charCNN"])
	useBERT=bool(args["bert"])
	uniLSTM=bool(args["uniLSTM"])
	pad=bool(args["pad"])
	print ("das", documents_are_sequences)

	if mode == "train":
		writePath=args["writePath"]
		train(embeddingsFile, writePath, documents_are_sequences, sentenceCNN, charCNN, pad, uniLSTM, useBERT)
	elif mode == "test":
		existingModel=args["existingModel"]
		predictionsFile=args["predictionsFile"]
		test(embeddingsFile, existingModel, predictionsFile, documents_are_sequences, sentenceCNN, charCNN, uniLSTM, useBERT)
	elif mode == "predict":
		existingModel=args["existingModel"]
		predictionsFile=args["predictionsFile"]
		fileToPredict=args["fileToPredict"]
		predict_file_batch(fileToPredict, embeddingsFile, existingModel, predictionsFile, documents_are_sequences, sentenceCNN, charCNN, pad, uniLSTM, useBERT)




