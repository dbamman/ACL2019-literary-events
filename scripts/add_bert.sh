for i in `ls ../data/tsv/train/`
do echo $i
python return_bert_features.py --file ../data/tsv/train/$i --model_path=bert-base-cased --output=../data/bert/train/$i
done

for i in `ls ../data/tsv/dev/`
do echo $i
python return_bert_features.py --file ../data/tsv/dev/$i --model_path=bert-base-cased --output=../data/bert/dev/$i
done

for i in `ls ../data/tsv/test/`
do echo $i
python return_bert_features.py --file ../data/tsv/test/$i --model_path=bert-base-cased --output=../data/bert/test/$i
done
