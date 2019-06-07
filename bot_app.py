# coding=utf8
from flask import Flask, render_template, redirect, url_for,request,jsonify,current_app,make_response
from datetime import timedelta
from flask_cors import CORS,cross_origin
from flask import make_response
# import statements
import wikipedia
import random
import re
from summa import summarizer
import operator
from rake_nltk import Rake
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
from allennlp.predictors.predictor import Predictor
from functools import update_wrapper
import nltk
import spacy


import numpy as np
import pandas as pd
import argparse
from pytorch_pretrained_bert.tokenization import (BasicTokenizer,
                                                  BertTokenizer, whitespace_tokenize)
import collections
import torch
from torch.utils.data import TensorDataset
from pytorch_pretrained_bert.modeling import BertForQuestionAnswering, BertConfig
import math
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm import tqdm
from termcolor import colored, cprint


class SquadExample(object):
    """
    A single training/test example for the Squad dataset.
    For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
                 example_id,
                 para_text,
                 qas_id,
                 question_text,
                 doc_tokens,
                 unique_id):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.example_id = example_id
        self.para_text = para_text
        self.unique_id = unique_id

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (self.qas_id)
        s += ", question_text: %s" % (
            self.question_text)
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))

        return s


### Convert paragraph to tokens and returns question_text
def read_squad_examples(input_data):
    """Read a SQuAD json file into a list of SquadExample."""

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    i = 0
    examples = []
    for entry in input_data:
        example_id = entry['id']
        paragraph_text = entry['text']
        doc_tokens = []
        prev_is_whitespace = True
        for c in paragraph_text:
            if is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False

        for qa in entry['ques']:
            qas_id = i
            question_text = qa

            example = SquadExample(example_id=example_id,
                                   qas_id=qas_id,
                                   para_text=paragraph_text,
                                   question_text=question_text,
                                   doc_tokens=doc_tokens,
                                   unique_id=i)
            i += 1
            examples.append(example)

    return examples


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_is_max_context,
                 token_to_orig_map,
                 input_ids,
                 input_mask,
                 segment_ids):
        self.doc_span_index = doc_span_index
        self.unique_id = unique_id
        self.example_index = example_index
        self.tokens = tokens
        self.token_is_max_context = token_is_max_context
        self.token_to_orig_map = token_to_orig_map
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 doc_stride, max_query_length):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    unique_id = 1
    for (example_index, example) in enumerate(examples):
        query_tokens = tokenizer.tokenize(example.question_text)
        ### Truncate the query if query length > max_query_length..
        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        tok_start_position = None
        tok_end_position = None

        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []
            tokens.append("[CLS]")
            segment_ids.append(0)
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(0)
            tokens.append("[SEP]")
            segment_ids.append(0)

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                       split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            features.append(InputFeatures(unique_id=unique_id,
                                          example_index=example_index,
                                          doc_span_index=doc_span_index,
                                          tokens=tokens,
                                          token_is_max_context=token_is_max_context,
                                          token_to_orig_map=token_to_orig_map,
                                          input_ids=input_ids,
                                          input_mask=input_mask,
                                          segment_ids=segment_ids))
            unique_id += 1

    return features


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)
    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])

    return best_indexes


def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            logger.info(
                "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info("Length not equal after stripping spaces: '%s' vs '%s'",
                        orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            logger.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            logger.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


_PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
    "PrelimPrediction",
    ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


_NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
    "NbestPrediction", ["text", "start_logit", "end_logit"])


def predict(examples, all_features, all_results, max_answer_length):
    n_best_size = 10

    ### Adding index to feature ###
    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    all_predictions = collections.OrderedDict()

    for example in examples:
        index = 0
        features = example_index_to_features[example.unique_id]
        prelim_predictions = []

        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            for start_index in start_indexes:
                for end_index in end_indexes:
                    #### we remove the indexes which are invalid @
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue

                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index]))

        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break

            feature = features[pred.feature_index]
            if pred.start_index > 0:  # this is a non-null prediction
                tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
                tok_text = " ".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = get_final_text(tok_text, orig_text, True)
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit))

        if not nbest:
            nbest.append(
                _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        probs = _compute_softmax(total_scores)
        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1
        all_predictions[example] = nbest_json[0]["text"]
        index = +1
    return all_predictions


RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = 'BERT/bert_model.bin'
config_file = 'BERT/bert_config.json'
config = BertConfig(config_file)
model = BertForQuestionAnswering(config)
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.to(device)

def bert_predict(context, q):
    parser = argparse.ArgumentParser()
    parser.add_argument("--paragraph", default=None, type=str)
    parser.add_argument("--model", default=None, type=str)
    parser.add_argument("--max_seq_length", default=384, type=int)
    parser.add_argument("--doc_stride", default=128, type=int)
    parser.add_argument("--max_query_length", default=64, type=int)
    parser.add_argument("--config_file", default=None, type=str)
    parser.add_argument("--max_answer_length", default=30, type=int)

    # args = parser.parse_args()
    # para_file = args.paragraph
    # model_path = args.model
    model_path = 'BERT/bert_model.bin'
    config_file = 'BERT/bert_config.json'
    max_answer_length = 30
    max_query_length = 64
    doc_stride = 128
    max_seq_length = 384

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    '''
    ### Raeding paragraph
    f = open(para_file, 'r',encoding="utf8")
    para = f.read()
    f.close()

    ## Reading question
#     f = open(ques_file, 'r')
#     ques = f.read()
#     f.close()

    para_list = para.split('\n\n')

    input_data = []
    i = 1
    for para in para_list :
        paragraphs = {}
        splits = para.split('\nQuestions:')
        paragraphs['id'] = i
        paragraphs['text'] = splits[0].replace('Paragraph:', '').strip('\n')
        paragraphs['ques']=splits[1].lstrip('\n').split('\n')
        input_data.append(paragraphs)
        i+=1
    '''
    input_data = [{'id': 1, 'text': context, 'ques': [q]}]
    ## input_data is a list of dictionary which has a paragraph and questions

    examples = read_squad_examples(input_data)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    eval_features = convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        doc_stride=doc_stride,
        max_query_length=max_query_length)

    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

    ### Loading Pretrained model for QnA
    '''
    config = BertConfig(config_file)
    model = BertForQuestionAnswering(config)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.to(device)
    '''

    pred_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)
    # Run prediction for full data
    pred_sampler = SequentialSampler(pred_data)
    pred_dataloader = DataLoader(pred_data, sampler=pred_sampler, batch_size=9)

    predictions = []
    for input_ids, input_mask, segment_ids, example_indices in tqdm(pred_dataloader):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)

        with torch.no_grad():
            batch_start_logits, batch_end_logits = model(input_ids, segment_ids, input_mask)

        features = []
        example = []
        all_results = []

        for i, example_index in enumerate(example_indices):
            start_logits = batch_start_logits[i].detach().cpu().tolist()
            end_logits = batch_end_logits[i].detach().cpu().tolist()
            feature = eval_features[example_index.item()]
            unique_id = int(feature.unique_id)
            features.append(feature)
            all_results.append(RawResult(unique_id=unique_id,
                                         start_logits=start_logits,
                                         end_logits=end_logits))

        output = predict(examples, features, all_results, max_answer_length)
        predictions.append(output)

    return list(predictions[0].values())[0]

basestring = (str,bytes)


def preprocess(text):
    tokenized_text = nltk.word_tokenize(text)
    tagged_text = nltk.tag.pos_tag(tokenized_text)
    simple_tagged_text = [(word, nltk.tag.map_tag('en-ptb', 'universal', tag)) for word, tag in tagged_text]
    return simple_tagged_text

priorities = {
    "PERSON":1,
    "EVENT":2,
    "ORG":3,
    "PRODUCT":4,
    "LOC":5,
    "GPE":6,
    "NORP":7,
    "LANGUAGE":8,
    "DATE":9,
    "OTHER":10
    }

nlp = spacy.load("en_core_web_md")
def spacy_ner(TEXT):
    # Much worse but faster NER with "en_core_web_sm"
    doc = nlp(TEXT)

    # tagged_text = preprocess(TEXT)
    tagged_text = []
    for token in doc:
        tagged_text.append((token.text, token.tag_))
    prev = ""
    ents_label_list = []
    for X in doc.ents:
        if X.label_ not in priorities.keys():
            ents_label_list.append((X.text, "OTHER"))
        else:
            if prev == "DATE" and X.label_ == "EVENT":
                old_ent = ents_label_list.pop()
                new_ent = (old_ent[0] + " " + X.text, "EVENT")
                ents_label_list.append(new_ent)
            else:
                ents_label_list.append((X.text, X.label_))
                prev = X.label_

    ents_label_list = sorted(ents_label_list, key=lambda x: priorities[x[1]])
    print("Entities and their labels")
    return ents_label_list

def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    if request.method == 'OPTIONS':
        response.headers['Access-Control-Allow-Methods'] = 'DELETE, GET, POST, PUT'
        headers = request.headers.get('Access-Control-Request-Headers')
        if headers:
            response.headers['Access-Control-Allow-Headers'] = headers
    return response



def crossdomain(origin=None, methods=None, headers=None,
                max_age=21600, attach_to_all=True,
                automatic_options=True):
    if methods is not None:
        methods = ', '.join(sorted(x.upper() for x in methods))
    if headers is not None and not isinstance(headers, basestring):
        headers = ', '.join(x.upper() for x in headers)
    if not isinstance(origin, basestring):
        origin = ', '.join(origin)
    if isinstance(max_age, timedelta):
        max_age = max_age.total_seconds()

    def get_methods():
        if methods is not None:
            return methods

        options_resp = current_app.make_default_options_response()
        return options_resp.headers['allow']

    def decorator(f):
        def wrapped_function(*args, **kwargs):
            if automatic_options and request.method == 'OPTIONS':
                resp = current_app.make_default_options_response()
            else:
                resp = make_response(f(*args, **kwargs))
            if not attach_to_all and request.method != 'OPTIONS':
                return resp

            h = resp.headers

            h['Access-Control-Allow-Origin'] = origin
            h['Access-Control-Allow-Methods'] = get_methods()
            h['Access-Control-Max-Age'] = str(max_age)
            if headers is not None:
                h['Access-Control-Allow-Headers'] = headers
            return resp

        f.provide_automatic_options = False
        return update_wrapper(wrapped_function, f)
    return decorator

st = StanfordNERTagger('stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz',
                       'stanford-ner/stanford-ner.jar',
                       encoding='utf-8')


NLTKSTOPWORDS = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
                 "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
                 "they", "them", "their", "theirs",  "themselves", "what", "which", "who", "whom", "this", "that",
                 "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
                 "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because",
                 "as", "until", "while", "of", "at", "by", "for", "with", "about", "against",
                 "between", "into", "through", "during", "before", "after", "above", "below", "to", "from",
                 "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here",
                 "there", "when", "where", "why", "how", "all", "any",
                 "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own",
                 "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]


PUNCIGN = "()"

LOCALINFO = {"you":'Data/About_Self',"yourself":'Data/About_Self',"You":'Data/About_Self',"Yourself":'Data/About_Self',
             "PESU":'Data/About_PESU',"PES University":'Data/About_PESU'}

DATAKEYS = LOCALINFO.keys()

r = Rake(stopwords=NLTKSTOPWORDS, punctuations=PUNCIGN)

#predictor = Predictor.from_path("QAModels/allen_bidaf")


def replace_pronouns(text, noun):
    rep_pronouns = ["She", "she", "He", "he", "They", "they", "It", "it"]
    try:
        for rep in rep_pronouns:
            if text[0:len(rep)] == rep:
                text = noun + text[len(rep):]
                break
    except IndexError:
        pass
    return text





class Context:
    def __init__(self, topic, match):
        if match:
            try:
                self.page = wikipedia.page(topic)
            except wikipedia.exceptions.DisambiguationError as err:
                self.page = wikipedia.page(err.options[0])
        else:
            try:
                results = wikipedia.search(topic, results=100)
                self.page = wikipedia.page(results[random.randint(0, 100)])
            except wikipedia.exceptions.DisambiguationError as err:
                self.page = wikipedia.page(err.options[0])
        self.sections = self.page.sections
        self.summary = ""
        self.keywords = ""
        self.keywords_full = []
        self.sentences = []
        self.questions = []
        self.ktotal = []

    def get_section(self):
        print("Selected article:", self.page.title)
        print("Available sections:")
        print("0 : All")
        i = 1
        for section in self.sections:
            print(i, ":", section)
            i += 1
        print()
        choice = int(input("Selected section:"))
        if choice == 0:
            self.summary = summarizer.summarize(self.page.content, words=300)
            r.extract_keywords_from_text(self.page.content)
            self.ktotal = r.get_ranked_phrases_with_scores()
        else:
            self.summary = summarizer.summarize(self.page.section(self.sections[choice-1]), words=300)
            r.extract_keywords_from_text(self.page.section(self.sections[choice-1]))
            self.ktotal = r.get_ranked_phrases_with_scores()

    def gen_questions(self):

        r.extract_keywords_from_text(self.summary)
        self.keywords_full = r.get_ranked_phrases_with_scores()
        self.sentences = self.summary.split(".")
        # print("All keywords:",self.keywords_full)
        for s in self.sentences:
            r.extract_keywords_from_text(s)
            only_sentence_keywords = r.get_ranked_phrases_with_scores()
            # print("Only Sentence:", only_sentence_keywords)
            s.rstrip("\n")
            sentence_keywords = []
            for tup in self.keywords_full:
                i = list(tup)
                if ")" in i[1]:
                    split_string = i[1].split(")")
                    repstring = ""
                    for k in split_string:
                        repstring += k + "\)"
                    i[1] = repstring
                if "(" in i[1]:
                    split_string = i[1].split("(")
                    repstring = ""
                    for k in split_string:
                        repstring += "\(" + k
                    i[1] = repstring
                if re.search(i[1], s, flags=re.IGNORECASE):
                    sentence_keywords.append(i)
            for tup in only_sentence_keywords:
                sentence_keywords.append(list(tup))
            sentence_keywords.sort(key=operator.itemgetter(0), reverse=True)
            if len(sentence_keywords) != 0:
                if ")" in sentence_keywords[0][1]:
                    split_string = sentence_keywords[0][1].split(")")
                    repstring = ""
                    for k in split_string:
                        repstring += k+"\)"
                    sentence_keywords[0][1] = repstring
                if "(" in sentence_keywords[0][1]:
                    split_string = sentence_keywords[0][1].split("(")
                    repstring = ""
                    for k in split_string:
                        repstring += "\("+k
                    sentence_keywords[0][1] = repstring
                qtext = re.sub(sentence_keywords[0][1],
                               "_"*len(sentence_keywords[0][1]), s, flags=re.IGNORECASE)
                self.questions.append([qtext, sentence_keywords[0]])
        print("Generated Questions:")
        for q in self.questions:
            print(q[0])
            print(q[1])

    def gen_questions_df(self):
        r.extract_keywords_from_text(self.summary)
        self.keywords_full = r.get_ranked_phrases_with_scores()
        self.sentences = self.summary.split(".")
        # print("All keywords:",self.keywords_full)
        for s in self.sentences:
            r.extract_keywords_from_text(s)
            only_sentence_keywords = r.get_ranked_phrases_with_scores()
            # print("Only Sentence:", only_sentence_keywords)
            s.rstrip("\n")
            sentence_keywords=[]
            for tup in self.keywords_full:
                i = list(tup)
                if ")" in i[1]:
                    split_string = i[1].split(")")
                    repstring = ""
                    for k in split_string:
                        repstring += k+"\)"
                    i[1] = repstring
                if "(" in i[1]:
                    split_string = i[1].split("(")
                    repstring = ""
                    for k in split_string:
                        repstring += "\("+k
                    i[1] = repstring
                if re.search(i[1], s, flags=re.IGNORECASE):
                    sentence_keywords.append(i)
            for tup in only_sentence_keywords:
                sentence_keywords.append(list(tup))
            sentence_keywords.sort(key=operator.itemgetter(0), reverse=True)
            if len(sentence_keywords) != 0:
                qtext = re.sub(sentence_keywords[0][1],"_"*len(sentence_keywords[0][1]), s, flags=re.IGNORECASE)
                self.questions.append([qtext, sentence_keywords[0]])
        self.questions.append(["USING FKF", "****"])
        self.keywords_full.sort(key=operator.itemgetter(0), reverse=True)
        for s in self.sentences:
            for kf in self.keywords_full:
                if kf[1] in s:
                    qtext = re.sub(kf[1],"_"*len(kf[1]),s,flags=re.IGNORECASE)
                    self.questions.append([qtext, kf[1]])
                    self.keywords_full.remove(kf)

        print("Generated Questions:")
        for q in self.questions:
            print(q[0])
            print(q[1])

    def highvolume(self):
        print("500 word summary:")
        print(self.summary)
        print("Fulltext keywords:")
        for key in self.ktotal:
            if key[0]>4:
                print(key)


class QInput:
    def __init__(self,ip_txt):
        self.questext = ip_txt
        if self.questext[len(self.questext)-1]!="?":
            self.questext+="?"

        self.inbotdata = False
        self.dkey = ""
        for k in DATAKEYS:
            if k in self.questext:
                self.inbotdata = True
                self.dkey = k

        self.tokenized_text = word_tokenize(ip_txt)
        self.classified_text = st.tag(self.tokenized_text)
        self.people_names = []
        self.locations = []
        self.orgs = []
        self.others = []
        self.qkey = r.extract_keywords_from_text(ip_txt)
        prev_tag = ""
        current = ""
        for i in self.classified_text:
            if i[1] !=0:
                if i[1] == "PERSON":
                    if prev_tag != "PERSON":
                        prev_tag = "PERSON"
                        current = i[0]
                    else:
                        current += " "+i[0]
                else:
                    if prev_tag == "PERSON":
                        self.people_names.append(current)
                        current = ""
                    prev_tag = i[1]

        prev_tag = ""
        current = ""
        for i in self.classified_text:
            if i[1] != 0:
                if i[1] == "LOCATION":
                    if prev_tag != "LOCATION":
                        prev_tag = "LOCATION"
                        current = i[0]
                    else:
                        current += " "+i[0]
                else:
                    if prev_tag == "LOCATION":
                        self.locations.append(current)
                        current = ""
                    prev_tag = i[1]

        prev_tag = ""
        current = ""
        for i in self.classified_text:
            if i[1] != 0:
                if i[1] == "ORGANIZATION":
                    if prev_tag != "ORGANIZATION":
                        prev_tag = "ORGANIZATION"
                        current = i[0]
                    else:
                        current += " "+i[0]
                else:
                    if prev_tag == "ORGANIZATION":
                        self.orgs.append(current)
                        current = ""
                    prev_tag = i[1]
        self.pos_text = pos_tag(self.tokenized_text)
        self.backup_keys = []
        for i in self.pos_text:
            if i[1][:2] == "NN":
                self.backup_keys.append(i[0])


    def showner(self):
        print(self.pos_text)
        print("People found :", self.people_names)
        print("Locations found :", self.locations)
        print("Organizations found :", self.orgs)
        print("Backup keys :", self.backup_keys)

    def gen_searchstring(self):
        self.search = []
        self.search.extend(self.people_names)
        self.search.extend(self.orgs)
        for i in self.backup_keys:
            if i not in self.search and i not in self.locations:
                self.search.append(i)
        self.search.extend(self.locations)
        if self.inbotdata==True:
            f = open(LOCALINFO[self.dkey],"r")
            self.con_final = f.read()
            self.context = self.con_final
            self.spacy_res = []
        else:
            self.spacy_res = spacy_ner(self.questext)
            other_nn = []
            if len(self.spacy_res)==0:
                #print("Looking for NN")
                doc = nlp(self.questext)
                for token in doc:
                    #print(token.tag_,end=",")
                    if 'NN' in token.tag_:
                        other_nn.append(token.lemma_)
                self.spacy_res = [other_nn]
            self.context = Context(self.spacy_res[0][0],1)
            print("Page used:",self.context.page.title)
            self.con_final = self.context.page.summary
            self.con_summary = self.context.page.summary

    def reduced_context(self):
        if self.inbotdata==True:
            text = self.context
            self.con_final = text
        else:
            text = self.context.page.content
            q = self.questext
            doc = nlp(q)
            rdc = ""
            doc_roots = []
            for chunk in doc.noun_chunks:
                doc_roots.append(chunk.root.text)
            for nkey in self.spacy_res:
                if nkey[0] in doc_roots:
                    doc_roots.remove(nkey[0])
            text = text.split('\n')

            if "== See also ==" in text:
                text = text[:text.index("== See also ==")]
            if "== Notes ==" in text:
                text = text[:text.index("== Notes ==")]
            if "== References ==" in text:
                text = text[:text.index("== References ==")]

            for line in text:
                for root in doc_roots:
                    if root in line:
                        sen = line.split(".")
                        for s in sen:
                            if root in s:
                                rdc += s + "."

            self.con_final = self.con_summary + rdc

    def guessans(self):
        #return(predictor.predict(passage=self.con_final, question=self.questext)['best_span_str'])
        self.reduced_context()
        #print(lultxt)
        return bert_predict(self.con_final,self.questext)


def qanswer(q_text):
    qobj = QInput(q_text)
    qobj.gen_searchstring()
    return qobj.guessans()

app = Flask(__name__)
CORS(app)
app.after_request(add_cors_headers)

@app.route("/")
def home():
    return render_template("ChatUI/index.html")


@app.route('/chat', methods=['GET'])
@crossdomain(origin='*')
def chat():
   message = "This the chat app"
   if request.method == 'GET':
        q_text = request.args.get('question')
        try:
            ans_text = qanswer(q_text)
            print(ans_text)
        except IndexError:
            ans_text = "Sorry! I am unable to answer that question!"
        return ans_text


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug = True)
