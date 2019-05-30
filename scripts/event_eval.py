import sys
import numpy as np

def read_data(filename):

	data={}
	with open(filename, encoding="utf-8") as file:
		for line in file:
			cols=line.rstrip().split("\t")
			book=cols[0]
			sentence=cols[1]
			word=cols[2]
			truth=int(cols[3])
			pred=int(cols[4])
			key="%s.%s" % (book, sentence)

			if key not in data:
				data[key]=[]

			data[key].append([truth, pred])

	new_data=[]
	for target in data:
		new_data.append(data[target])
	return new_data

def check_f1_two_lists(gold, preds):

	correct=0.
	trials=0.
	trues=0.
	for j in range(len(preds)):
		if preds[j] == 1:
			trials+=1
		if gold[j] == 1:
			trues+=1
		if preds[j] == gold[j] and preds[j] == 1:
			correct+=1

	p=0
	if trials > 0:
		p=correct/trials
	r=0
	if trues > 0:
		r=correct/trues

	f=0
	if (p+r) > 0:
		f=(2*p*r)/(p+r)

	print ("precision: %.3f %s/%s" % (p, correct, trials))
	print ("recall: %.3f %s/%s" % (r, correct, trues))
	print ("F: %.3f" % f)

	return f, p, r, correct, trials, trues


def check_f1(data):
	correct=0.
	trials=0.
	trues=0.

	for sentence in data:
		for word in sentence:
			truth=word[0]
			pred=word[1]

			if pred == 1:
				trials+=1
			if truth == 1:
				trues+=1
			if pred == truth and pred == 1:
				correct+=1

	p=0.
	if trials > 0:
		p=correct/trials
	r=0.
	if trues > 0:
		r=correct/trues

	f=0.
	if (p+r) > 0:
		f=(2*p*r)/(p+r)

	return f, p, r, correct, trials, trues

def bootstrap(data, resamples):
	precs=[]
	recalls=[]
	f1s=[]
	for b in range(resamples):
		choice=np.random.choice(data, size=len(data), replace=True)
		f, p, r, correct, trials, trues=check_f1(choice)
		precs.append(p)
		recalls.append(r)
		f1s.append(f)

	prec_lower, prec_median, prec_upper=100*np.percentile(precs, [2.5, 50, 97.5])
	rec_lower, rec_median, rec_upper=100*np.percentile(recalls, [2.5, 50, 97.5])
	f_lower, f_median, f_upper=100*np.percentile(f1s, [2.5, 50, 97.5])

	print("LaTeX:")
	print("""MODEL&%.1f {\small [%.1f-%.1f]}&%.1f {\small [%.1f-%.1f]}&%.1f {\small [%.1f-%.1f]} \\\\ \hline\n""" % (prec_median, prec_lower, prec_upper, rec_median, rec_lower, rec_upper, f_median, f_lower, f_upper))

	print("%s\t%s" % ("Precision", '\t'.join(["%.3f" % x for x in np.percentile(precs, [2.5, 50, 97.5])])))
	print("%s\t%s" % ("Recall", '\t'.join(["%.3f" % x for x in np.percentile(recalls, [2.5, 50, 97.5])])))
	print("%s\t%s" % ("F1", '\t'.join(["%.3f" % x for x in np.percentile(f1s, [2.5, 50, 97.5])])))

if __name__ == "__main__":

	data=read_data(sys.argv[1])
	bootstrap(data, 10000)

