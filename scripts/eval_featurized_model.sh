python featurized_model.py
python event_eval.py ../results/featurized.preds.txt > ../results/featurized.preds.bootstrap.txt
python event_eval.py ../results/featurized.word_ablation.preds.txt > ../results/featurized.word_ablation.preds.bootstrap.txt
python event_eval.py ../results/featurized.lemma_ablation.preds.txt > ../results/featurized.lemma_ablation.preds.bootstrap.txt
python event_eval.py ../results/featurized.pos_ablation.preds.txt > ../results/featurized.pos_ablation.preds.bootstrap.txt
python event_eval.py ../results/featurized.context_ablation.preds.txt > ../results/featurized.context_ablation.preds.bootstrap.txt
python event_eval.py ../results/featurized.syntax_ablation.preds.txt > ../results/featurized.syntax_ablation.preds.bootstrap.txt
python event_eval.py ../results/featurized.wordnet_ablation.preds.txt > ../results/featurized.wordnet_ablation.preds.bootstrap.txt
python event_eval.py ../results/featurized.embedding_ablation.preds.txt > ../results/featurized.embedding_ablation.preds.bootstrap.txt
python event_eval.py ../results/featurized.bare_plurals_ablation.preds.txt > ../results/featurized.bare_plurals_ablation.preds.bootstrap.txt