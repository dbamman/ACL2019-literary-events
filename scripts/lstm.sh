MODEL=lstm

RESULTS_DIR=../results/$MODEL
mkdir -p $RESULTS_DIR

for i in 0 1 2 3 4
	do
		python3 neural.py -m train --writePath $RESULTS_DIR/$MODEL.${i} -e ../data/guten.vectors.txt --pad --batchsize 16 --uniLSTM
		python3 neural.py -m test --existingModel $RESULTS_DIR/$MODEL.${i}.hdf5 -e ../data/guten.vectors.txt --pad --batchsize 16 --uniLSTM --predictionsFile $RESULTS_DIR/$MODEL.${i}.test.preds.txt
	done

python aggregate_preds.py $RESULTS_DIR/$MODEL.0.test.preds.txt $RESULTS_DIR/$MODEL.1.test.preds.txt $RESULTS_DIR/$MODEL.2.test.preds.txt $RESULTS_DIR/$MODEL.3.test.preds.txt $RESULTS_DIR/$MODEL.4.test.preds.txt > $RESULTS_DIR/$MODEL.aggregate.test.preds.txt
python3 event_eval.py $RESULTS_DIR/$MODEL.aggregate.test.preds.txt > $RESULTS_DIR/$MODEL.aggregate.test.preds.results

