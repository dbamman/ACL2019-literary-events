# get skipgram embeddings trained on 15K texts from Project Gutenberg

wget http://people.ischool.berkeley.edu/~dbamman/guten.vectors.txt.gz
gunzip guten.vectors.txt.gz
mv guten.vectors.txt ../data/