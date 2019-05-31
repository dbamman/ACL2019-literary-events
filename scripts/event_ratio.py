import sys, re, csv, os
from os import listdir
from os.path import isfile, join
import numpy as np
from math import sqrt 

def get_ratio(filename):
	events=0.
	tokens=0.

	if not os.path.isfile(filename):
		print ("%s missing!" % filename)
		return None, None

	with open(filename) as file:

		for line in file:
			cols=line.rstrip().split("\t")
			if len(cols) < 2:
				continue

			if cols[1] == "EVENT":
				events+=1
			tokens+=1

	return events, tokens

def proc(filename, label, out):

	ratios=[]
	distances=[]
	with open(filename) as csv_file:
		csv_reader = csv.reader(csv_file)
		id_col=author_col=title_col=None
		for idx,row in enumerate(csv_reader):
			if idx == 0:
				id_col=row.index("Book_ID")
				author_col=row.index("Author")
				title_col=row.index("Title")

				continue

			idd=row[id_col]
			author=row[author_col]
			title=row[title_col]
			path="../analysis/data/gutenberg_clean_tokenized_bert_predictions/%s.txt" % idd
			events, tokens=get_ratio(path)
			
			if events is not None:
				ratio=events/tokens
				distance=tokens/events
				ratios.append(ratio)
				distances.append(distance)
				out.write("%s\t%s\t%.5f\t%s\t%s\n" % (idd, label, ratio, author, title))

	ratio_stderr=np.std(ratios)/sqrt(len(ratios))
	ratio_lower=np.mean(ratios) - (1.96*ratio_stderr)
	ratio_upper=np.mean(ratios) + (1.96*ratio_stderr)
	
	distance_stderr=np.std(distances)/sqrt(len(distances))
	distance_lower=np.mean(distances) - (1.96*distance_stderr)
	distance_upper=np.mean(distances) + (1.96*distance_stderr)

	print("%s&%.1f {\\small [%.1f-%.1f]}& %.1f {\\small [%.1f-%.1f]} \\\\" % (label, np.mean(ratios)*100, ratio_lower*100, ratio_upper*100, np.mean(distances), distance_lower, distance_upper))

outfile="../analysis/results/prestige.txt"
with open(outfile, "w") as out:
	out.write("text\tprestige\tval\tauthor\ttitle\n")
	proc("../analysis/metadata/high_prestige.csv", "High prestige", out)
	proc("../analysis/metadata/low_prestige.csv", "Low prestige", out)

outfile="../analysis/results/popularity.txt"
with open(outfile, "w") as out:
	out.write("text\tpopularity\tval\tauthor\ttitle\n")
	proc("../analysis/metadata/high_popularity.csv", "High popularity", out)
	proc("../analysis/metadata/low_popularity.csv", "Low popularity", out)




