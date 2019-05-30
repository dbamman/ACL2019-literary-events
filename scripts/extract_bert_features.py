# >>> Code has been heavily adapted and extended from original source (see copyright below) <<<

#coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Extract pre-computed feature vectors from a PyTorch BERT model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import logging
import json
import re
import pickle
import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel


class InputExample(object):

    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


def get_retokenized_embeds(input_sents, output_embeds, concat):
    """Averages subword token embeddings and either concatenates or averages
       model layer weights associated with each of the original input tokens"""
    
    all_embeds_list = []
    
    for sent_idx in range(len(output_embeds)):
    
        embeds_list = []
        features = output_embeds[sent_idx]['features'][1:-1]
        for feature in features:
            layers = feature['layers']

            if concat:
                embed = np.concatenate([np.array(layer['values']) for layer in layers])
            else:
                embed = np.sum([np.array(layer['values']) for layer in layers],axis=0)

            embeds_list.append(embed)

        tokens = [features[i]['token'] for i in range(len(features))]
        
        subword_list = [0 for token in tokens]
        for idx in range(len(tokens)):
            if tokens[idx].startswith('##'):
                subword_list[idx] = 2
                if subword_list[idx-1] != 2:
                    subword_list[idx-1] = 1
    
        sub_index_lists = [] 
        for idx in range(len(tokens)):
            if subword_list[idx] == 1:
                curr_list = [idx]
                idx2 = idx + 1
                while subword_list[idx2] == 2:
                    curr_list.append(idx2)
                    if idx2 == len(tokens)-1:
                        break
                    idx2 +=1
                sub_index_lists.append(curr_list)  

        subword_list[:] = [x for x in subword_list if x != 2]
        
        for idx in range(len(subword_list)):
            if subword_list[idx] == 1:
                subword_list[idx] = sub_index_lists.pop(0)
                
        count = 0
        for idx in range(len(subword_list)):
            if type(subword_list[idx]) == list:
                count += len(subword_list[idx])
            else:
                subword_list[idx] = count
                count += 1
        
        final_embeds_list = []
        for idx in range(len(subword_list)):
            if type(subword_list[idx]) == list:
                embed = np.sum([embeds_list[pos] for pos in subword_list[idx]],axis=0)/len(subword_list[idx])
                final_embeds_list.append(embed)
            else:
                final_embeds_list.append(embeds_list[subword_list[idx]])

        if len(input_sents[sent_idx]) != len(final_embeds_list):
            print (sent_idx, input_sents[sent_idx], len(input_sents[sent_idx]), len(final_embeds_list), subword_list)

        assert len(input_sents[sent_idx]) == len(final_embeds_list)
       
        all_embeds_list.append(final_embeds_list)

    return all_embeds_list



def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputFeature`s."""

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0      0   0    1  1  1   1  1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)
        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length

        #if ex_index < 5:
            #logger.info("*** Example ***")
            #logger.info("unique_id: %s" % (example.unique_id))
            #logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            #logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            #logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            #logger.info(
                #"input_type_ids: %s" % " ".join([str(x) for x in input_type_ids])) 

        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def read_examples(input_sents):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    unique_id = 0
    if any(isinstance(el, list) for el in input_sents):
        for sent in input_sents:
            text_a = ' '.join(sent).strip()
            text_b = None
            examples.append(
                InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
            unique_id += 1
    else:
        text_a = ' '.join(input_sents).strip()
        text_b = None
        examples.append(
                InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
    return examples


def return_berts(input_sents=None, model_path=None, lower_case=False,layers="-1,-2,-3,-4",max_seq_length=128,
            batch_size=8,local_rank=-1,concat=True):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    n_gpu = torch.cuda.device_count()
        
    layer_indexes = [int(x) for x in layers.split(",")]

    if model_path == 'bert-base-cased' or model_path == 'bert-large-cased':
        tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=False, do_basic_tokenize=False)
    else:
        tokenizer = BertTokenizer.from_pretrained(model_path, do_basic_tokenize=False)
   
    examples = read_examples(input_sents)

    features = convert_examples_to_features(
        examples=examples, seq_length=max_seq_length, tokenizer=tokenizer)

    unique_id_to_feature = {}
    for feature in features:
        unique_id_to_feature[feature.unique_id] = feature

    model = BertModel.from_pretrained(model_path)

    model.to(device)

    if local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                          output_device=local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

    eval_data = TensorDataset(all_input_ids, all_input_mask, all_example_index)
    if local_rank == -1:
        eval_sampler = SequentialSampler(eval_data)
    else:
        eval_sampler = DistributedSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)

    model.eval()

    output_embeds = []
    for input_ids, input_mask, example_indices in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)

        all_encoder_layers, _ = model(input_ids, token_type_ids=None, attention_mask=input_mask)
        all_encoder_layers = all_encoder_layers

        for b, example_index in enumerate(example_indices):
            feature = features[example_index.item()]
            unique_id = int(feature.unique_id)
            # feature = unique_id_to_feature[unique_id]
            output_bert = collections.OrderedDict()
            output_bert["linex_index"] = unique_id
            all_out_features = []
            for (i, token) in enumerate(feature.tokens):
                all_layers = []
                for (j, layer_index) in enumerate(layer_indexes):
                    layer_output = all_encoder_layers[int(layer_index)].detach().cpu().numpy()
                    layer_output = layer_output[b]
                    layers = collections.OrderedDict()
                    layers["index"] = layer_index
                    layers["values"] = [
                        round(x.item(), 6) for x in layer_output[i]
                    ]
                    all_layers.append(layers)
                out_features = collections.OrderedDict()
                out_features["token"] = token
                out_features["layers"] = all_layers
                all_out_features.append(out_features)
            output_bert["features"] = all_out_features
            output_embeds.append(output_bert)
        

    return get_retokenized_embeds(input_sents,output_embeds,concat)

    
