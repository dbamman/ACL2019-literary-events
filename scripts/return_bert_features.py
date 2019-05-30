import pdb
import torch
import os, sys, argparse, pickle, numpy, time
from extract_bert_features import *
from pytorch_pretrained_bert.tokenization import BertTokenizer


parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, default=None)
parser.add_argument('--model_path', type=str,
                    help="Select a BERT pre-trained model from the following:"
                    "bert-base-uncased, bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
parser.add_argument('--output', type=str)

args = parser.parse_args()

model_path = args.model_path

def get_bert_representations(model_path, sentence_list, max_batch=512):
    t0 = time.time()
    # Order sentences by length
    lengths = [len(l) for l in sentence_list]
    ordering = np.argsort(lengths)
    ordered_list = [None for i in range(len(sentence_list))]
    for i, ind in enumerate(ordering):
            ordered_list[i] = sentence_list[ind]

    ordered_berts = []

    # Get representations for the ordered list to reduce padding required
    i = 0
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False, do_basic_tokenize=False)

    while i < len(sentence_list):
            if i > 0:
                    print("Processing %d of %d, total time: %f" %(i, len(sentence_list), time.time()-t0))
            sents = ordered_list[i:i+max_batch]
            max_len = max([sum([len(tokenizer.tokenize(s)) for s in sent]) for sent in sents])
            max_len += 10
            print("Max length: " + str(max_len))
            result = return_berts(input_sents=sents, model_path=model_path, max_seq_length=max_len)
            result = [np.array(r) for r in result]
            ordered_berts += result
            i += max_batch
    print("Computed Berts in %f seconds." % (time.time()-t0))

    # Put the representations back in the original ordering of the sentences
    berts = [None for i in range(len(sentence_list))]
    for i, ind in enumerate(ordering):
            berts[ind] = ordered_berts[i]
            assert(berts[ind].shape[0] == len(sentence_list[ind]))

    return berts

def read_sentences(filenames, case_sensitive_embeddings=True):

    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False, do_basic_tokenize=False)

    word_seqs = []
    word_seq = []

    orig_lines=[]
    orig_line=[]

    for filename in filenames:
      with open(filename, encoding="utf-8") as f:
        for line in f:
          cols=line.rstrip().split("\t")
          if len(cols) < 2:
            if len(word_seq) > 0:

              # The BERT max sentence length (in word pieces) is 512 tokens, so let's chop up longer
              # sentences into smaller chunks
              word_pieces_total=sum([len(tokenizer.tokenize(s)) for s in word_seq])

              if word_pieces_total > 500:
                print("sequence longer than 500: word pieces - %s, words - %s" % (word_pieces_total, len(word_seq)))
                wp_count=0
                w_cur=[]
                o_cur=[]
                for idx, word in enumerate(word_seq):
                  wp=tokenizer.tokenize(word)
                  if wp_count + len(wp) > 500:
                    word_seqs.append(w_cur)
                    orig_lines.append(o_cur)
                    w_cur=[]
                    o_cur=[]
                    wp_count=0

                  w_cur.append(word)
                  o_cur.append(orig_line[idx])

                  wp_count+=len(wp)

                if len(w_cur) > 0:
                    word_seqs.append(w_cur)
                    orig_lines.append(o_cur)
                    w_cur=[]
                    o_cur=[]
                    wp_count=0

              else:
                word_seqs.append(word_seq)
                orig_lines.append(orig_line)

              word_seq =[]
              orig_line =[]
            continue

        
          word = re.sub(" ", "_", cols[0])
          word = re.sub(" ", "_", word)
          # for BERT word piece tokenization
          word = re.sub("^##", "-##", word)

          if case_sensitive_embeddings:
            word=word.lower()

          word_seq.append(word)
          orig_line.append(line.rstrip())

    if len(word_seq) > 0:
      
      word_pieces_total=sum([len(tokenizer.tokenize(s)) for s in word_seq])

      if word_pieces_total > 500:
        print("sequence longer than 500: word pieces - %s, words - %s" % (word_pieces_total, len(word_seq)))
        wp_count=0
        w_cur=[]
        o_cur=[]
        for idx, word in enumerate(word_seq):
          wp=tokenizer.tokenize(word)
          if wp_count + len(wp) > 500:
            word_seqs.append(w_cur)
            orig_lines.append(o_cur)
            w_cur=[]
            o_cur=[]
            wp_count=0

          w_cur.append(word)
          o_cur.append(orig_line[idx])

          wp_count+=len(wp)

        if len(w_cur) > 0:
            word_seqs.append(w_cur)
            orig_lines.append(o_cur)
            w_cur=[]
            o_cur=[]
            wp_count=0
      else:
          word_seqs.append(word_seq)
          orig_lines.append(orig_line)

    return word_seqs, orig_lines


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('')
print("********************************************")
print("Running on: {}".format(device))
print("Using the following model: {}".format(model_path))
print("********************************************")
print('')

sents, orig_lines = read_sentences([args.file])

berts = get_bert_representations(model_path, sents)

with open(args.output, "w", encoding="utf-8") as out:
    for idx, bert in enumerate(berts):
        orig=orig_lines[idx]
        for i in range(len(orig)):
            out.write("%s\t%s\n" % (orig[i], ' '.join(str(x) for x in bert[i])))
        out.write("\n")
