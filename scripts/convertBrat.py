import sys, os
import re
from os import listdir
from os.path import isfile, join

def read_splits(trainFile, devFile, testFile):
	train={}
	dev={}
	test={}

	with open(trainFile) as file:
		for line in file:
			train[line.rstrip()]=1
	with open(devFile) as file:
		for line in file:
			dev[line.rstrip()]=1
	with open(testFile) as file:
		for line in file:
			test[line.rstrip()]=1

	return train, dev, test

def read_ann(filename):
	events={}

	with open(filename, encoding="utf-8") as file:
		for line in file:
			cols=line.rstrip().split("\t")
			if len(cols) < 3:
				continue
				
			idd=cols[0]
			parts=cols[1].split(" ")
			term=cols[2]
			cat=parts[0]
			if cat != "EVENT":
				continue

			start=int(parts[1])
			end=int(parts[2])


			events[start]=(cat, term, start, end)

	return events


def read_txt(filename, anns, outfile):

	out=open(outfile, "w", encoding="utf-8")
	with open(filename, encoding="utf-8") as file:
		text=file.read()

	p = re.compile("[\n ]")

	spaces=[]
	space_kind=[]
	spaces.append(-1)
	space_kind.append("START")
	for m in p.finditer(text):
		spaces.append(m.start())
		space_kind.append(m.group())

	for idx, loc in enumerate(spaces):
		if space_kind[idx] == "\n":
			out.write("\n")
		start=loc+1
		end=len(text)
		if idx < len(spaces)-1:
			end=spaces[idx+1]

		token=text[start:end]
		label="O"

		if start in anns:
			label="EVENT"

		if len(token) > 0:
			out.write ("%s\t%s\n" % (token, label))

	out.close()

if __name__ == "__main__":

	folder="../data/brat"
	outdirr="../data/tsv"

	os.makedirs("%s/%s" % (outdirr, "train"))
	os.makedirs("%s/%s" % (outdirr, "dev"))
	os.makedirs("%s/%s" % (outdirr, "test"))
	
	train, dev, test=read_splits("../data/train.ids", "../data/dev.ids", "../data/test.ids")

	onlyfiles = [f for f in listdir(folder) if isfile(join(folder, f))]

	for filename in onlyfiles:
		if filename.endswith(".txt"):
			base=re.sub(".txt", "", filename.split("/")[-1])
			print(base)

			textFile="%s/%s.txt" % (folder, base)
			annFile="%s/%s.ann" % (folder, base)

			outDir=None
			if base in train:
				outDir=os.path.join("../data/tsv", "train")
			elif base in dev:
				outDir=os.path.join("../data/tsv", "dev")
			elif base in test:
				outDir=os.path.join("../data/tsv", "test")

			outFile=os.path.join(outDir, "%s.tsv" % base)

			anns=read_ann(annFile)
			read_txt(textFile, anns, outFile)


