import sys
from collections import defaultdict

def proc(filenames):
	preds=defaultdict(int)
	golds=defaultdict(int)
	n=len(filenames)

	keys={}
	for filename in filenames:

		with open(filename, encoding="utf-8") as file:
			for idx, line in enumerate(file):
				cols=line.rstrip().split("\t")
				gold=int(cols[3])
				pred=int(cols[4])

				golds[idx]=gold

				preds[idx]+=float(pred)/n
				keys[idx]="%s\t%s\t%s" % (cols[0], cols[1], cols[2])

	for idx in sorted(preds.keys()):
		prediction=0
		if preds[idx] >= 0.5:
			prediction=1

		print("%s\t%s\t%s" % (keys[idx], golds[idx], prediction))

proc(sys.argv[1:])

