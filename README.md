# ACL2019-literary-events

Code to support Sims, Park and Bamman (2019), "Literary Event Detection" (ACL).


 
## Setup

```sh
pip install spacy==2.1.4
pip install torch==1.1.0
pip install pytorch-pretrained-bert==0.6.2

python -m spacy download en_core_web_sm
python -m spacy download en
```

## 1. Prepare data

```sh
cd scripts/
python convertBrat.py
./add_bert.sh
./get_gutenberg_embeddings.sh
```

## 2. Verb-only baseline
```sh
cd scripts/
./eval_verb_model.sh
```

## 3. Featurized model

Download CELEX2 ([LDC96L14](https://catalog.ldc.upenn.edu/LDC96L14)) and place the `celex2/english/esl/esl.cd` file in the `data/` directory.

```sh
cd scripts/
./eval_featurized_model.sh
```

## 4. Neural models

```sh
cd scripts/
./lstm.sh
./bilstm.sh
./bilstm.charCNN.sh
./bilstm.sentenceCNN.sh
./bilstm.documentContext.sh
./bert.sh
./bert.charCNN.sh
```

## 5. Analysis
```sh
cd scripts/
./analyze_gutenberg.sh
```