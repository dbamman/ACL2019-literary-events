##
# Below follows the steps for identifying events in texts from Project Gutenberg; 
# to skip all steps, download processed data here:
# http://people.ischool.berkeley.edu/~dbamman/data/gutenberg_clean_tokenized_bert_predictions.tar.gz
##

# Remove Gutenberg boilerplate
mkdir ../analysis/data/gutenberg_clean/

for i in `ls ../analysis/data/gutenberg/`
do 
	python filter_boilerplate.py ../analysis/data/gutenberg/$i ../analysis/data/gutenberg_clean/$i
done

# Tokenize texts using PTB tokenization
cd ../analysis/tokenizer

mkdir ../data/gutenberg_clean_tokenized/

for i in `ls ../data/gutenberg_clean/`
do 
	./runjava tokenizer/TokenizeCoNLL ../data/gutenberg_clean/$i ../data/gutenberg_clean_tokenized/$i
done

# get BERT embeddings for tokens (this takes a long time)

mkdir ../data/gutenberg_clean_tokenized_bert/

cd ../../scripts/

for i in `ls ../analysis/data/gutenberg_clean_tokenized/`
do
	python return_bert_features.py --file ../analysis/data/gutenberg_clean_tokenized/$i --model_path=bert-base-cased --output=../analysis/data/gutenberg_clean_tokenized_bert/$i
done


# Tag events with best model
BEST_MODEL=../results/bert/bert.3.hdf5

mkdir ../analysis/data/gutenberg_clean_tokenized_bert_predictions/

for i in `ls ../analysis/data/gutenberg_clean_tokenized_bert/`
do
	python neural.py --existingModel $BEST_MODEL --predictionsFile ../analysis/data/gutenberg_clean_tokenized_bert_predictions/$i --fileToPredict ../analysis/data/gutenberg_clean_tokenized_bert/$i --mode predict -e ../data/guten.vectors.txt --bert
done

# calculate popularity/prestige measures
python event_ratio.py > ../analysis/results/prestige.popularity.txt

