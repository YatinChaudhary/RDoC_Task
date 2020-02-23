
#!/usr/bin/env python
# -*- coding: utf-8 -*-


import sys, random
import os
# sys.setdefaultencoding() does not exist, here!
#reload(sys)  # Reload does the trick!
#sys.setdefaultencoding('UTF8')

# TREC

# After running this code, we can confirm that the collection has a
# split of 5435/50/2189 documents for train/validation/test respectively,
# and a total of 6 different categories.

from nltk import word_tokenize
from nltk.corpus import reuters
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
import re, random, pickle
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from gensim import corpora, models, similarities
from os.path import expanduser
from collections import defaultdict
import codecs

import csv
from nltk.tokenize import RegexpTokenizer
import numpy as np
import tensorflow as tf
import argparse
from sklearn.utils import shuffle

import model.data

seed = 42
np.random.seed(seed)
tf.set_random_seed(seed)

csv.field_size_limit(2**28)
tokenizer = RegexpTokenizer(r'\w+')

cachedStopWords = []
with open("english_stopwords.txt", "r") as f:
	cachedStopWords.extend([line.strip() for line in f.readlines()])


def tokens(text):
	return [w.lower() for w in text.split()]

def counts_to_sequence(counts):
	seq = []
	for i in range(len(counts)):
		seq.extend([i] * int(counts[i]))
	return seq

def log_counts(ids, vocab_size):
	counts = np.bincount(ids, minlength=vocab_size)
	return np.floor(0.5 + np.log(counts + 1))

def preprocess(text, vocab_to_id, dataset_type):

	ids = [vocab_to_id.get(x) for x in tokens(text) if not (vocab_to_id.get(x) is None)]
	
	if dataset_type == "docnade":
		counts = log_counts(ids, len(vocab_to_id))
		sequence = counts_to_sequence(counts)
	else:
		sequence = ids
	
	if len(sequence) == 0:
		return None
	else:
		return ' '.join([str(x) for x in sequence])

def TF(docs, max_features=2000):
	cv = CountVectorizer(tokenizer=tokens, min_df=1, max_df=1.0, max_features=max_features, encoding='utf-8', decode_error='ignore')
	cv.fit(docs)
	return cv


def load_file(filename):
	"""
	Read the tab delimited file containing the labels and the docs.

	"""
	labels = []
	docs = []

	with open(filename) as f:
		for line in f:
			content = line.split('\t')

			if len(content) > 2:
				print('incorrect read')
				exit()

			if len(content[1]) == 0: continue

			docs.append(str(content[1]).strip('\r').strip('\n').strip('\r\n'))
			labels.append(content[0])

	return docs, labels


def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')


def main(args):
	args.split_train_val = str2bool(args.split_train_val)

	doc_train_filename = args.training_file
	doc_val_filename = args.validation_file
	doc_test_filename = args.test_file
	
	train_csv_filename = os.path.join(args.data_output, "training.csv")
	val_csv_filename = os.path.join(args.data_output, "validation.csv")
	test_csv_filename = os.path.join(args.data_output, "test.csv")

	if not os.path.exists(args.data_output):
		os.makedirs(args.data_output)

	docnade_vocabulary = args.vocab_size
	docnade_vocab_filename = os.path.join(args.data_output, "vocab_docnade.vocab")
	lstm_vocab_filename = os.path.join(args.data_output, "vocab_lstm.vocab")

	mapping_dict_filename = os.path.join(args.data_output, "mapping_dict.pkl")
	

	train_docs, train_docs_labels = load_file(doc_train_filename)
	test_docs, test_docs_labels = load_file(doc_test_filename)
	#if not args.split_train_val:
	val_docs, val_docs_labels = load_file(doc_val_filename)

	print(np.unique(train_docs_labels))

	train_docs, train_docs_labels = shuffle(train_docs, train_docs_labels, random_state=123)
	val_docs, val_docs_labels = shuffle(val_docs, val_docs_labels, random_state=123)
	#test_docs, test_docs_labels = shuffle(test_docs, test_docs_labels, random_state=123)

	###########################################################################
	# Prepare CSV file

	new_train_docs = []
	with open(train_csv_filename, 'w', newline='') as csvfile:
		filewriter = csv.writer(csvfile, delimiter=',')
		for doc, label in zip(train_docs, train_docs_labels):
			new_doc_tokens = tokens(str(doc).lower().strip())
			new_doc_tokens = [token for token in new_doc_tokens if not token in cachedStopWords]
			new_doc = ' '.join(new_doc_tokens)
			li = [str(label).lower().strip(), str(new_doc).lower().strip()]
			filewriter.writerow(li)
			new_train_docs.append(str(new_doc).lower().strip())

	new_val_docs = []
	with open(val_csv_filename, 'w', newline='') as csvfile:
		filewriter = csv.writer(csvfile, delimiter=',')
		for doc, label in zip(val_docs, val_docs_labels):
			new_doc_tokens = tokens(str(doc).lower().strip())
			new_doc_tokens = [token for token in new_doc_tokens if not token in cachedStopWords]
			new_doc = ' '.join(new_doc_tokens)
			li = [str(label).lower().strip(), str(new_doc).lower().strip()]
			filewriter.writerow(li)
			new_val_docs.append(str(new_doc).lower().strip())

	new_test_docs = []
	with open(test_csv_filename, 'w', newline='') as csvfile:
		filewriter = csv.writer(csvfile, delimiter=',')
		for doc, label in zip(test_docs, test_docs_labels):
			new_doc_tokens = tokens(str(doc).lower().strip())
			new_doc_tokens = [token for token in new_doc_tokens if not token in cachedStopWords]
			new_doc = ' '.join(new_doc_tokens)
			li = [str(label).lower().strip(), str(new_doc).lower().strip()]
			filewriter.writerow(li)
			new_test_docs.append(str(new_doc).lower().strip())

	total_docs = []
	total_docs.extend(new_train_docs)
	total_docs.extend(new_val_docs)
	#total_docs.extend(new_test_docs)
	
	# Saving docnade vocabulary
	#representer = TF(total_docs, max_features=docnade_vocabulary)
	representer = TF(total_docs, max_features=None)
	vocab_dict_docnade = representer.get_feature_names()

	with open(docnade_vocab_filename, "w") as f:
		f.write('\n'.join(vocab_dict_docnade))

	# Preparing CSV files for DocNADE Tensorflow
	data = model.data.Dataset(args.data_output)

	with open(docnade_vocab_filename, 'r') as f:
		vocab = [w.strip() for w in f.readlines()]
	vocab_to_id = dict(zip(vocab, range(len(vocab))))

	if not os.path.isdir(args.data_output):
		os.mkdir(args.data_output)

	labels = {}
	removed_indices = {"training":[], "test":[], "validation":[]}
	for collection in data.collections:
		output_path = os.path.join(args.data_output, '{}_docnade.csv'.format(collection))
		with open(output_path, 'w', newline='') as f:
		#with open(output_path, 'w') as f:
			w = csv.writer(f, delimiter=',')
			count = -1
			for y, x in data.rows(collection, num_epochs=1):
				count += 1
				try:
					pre = preprocess(x, vocab_to_id, "docnade")
					if pre is None:
						removed_indices[str(collection).lower()].append(count)
						continue
					if ':' in y:
						temp_labels = y.split(':')
						new_label = []
						for label in temp_labels:
							if label not in labels:
								labels[label] = len(labels)
							new_label.append(str(labels[label]))
						temp_label = ':'.join(new_label)
						w.writerow((temp_label, pre))
					else:
						if y not in labels:
							labels[y] = len(labels)
						w.writerow((labels[y], pre))
				except:
					import pdb; pdb.set_trace()

	with open(os.path.join(args.data_output, 'labels.txt'), 'w') as f:
		f.write('\n'.join([k for k in sorted(labels, key=labels.get)]))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training-file', type=str, required=True,
                        help='path to validation text file')
    parser.add_argument('--validation-file', type=str, required=True,
                        help='path to validation text file')
    parser.add_argument('--test-file', type=str, required=True,
                        help='path to validation text file')
    parser.add_argument('--data-output', type=str, required=True,
                        help='path to data output directory')
    parser.add_argument('--vocab-size', type=int, default=2000,
                        help='the vocab size')
    parser.add_argument('--split-train-val', type=str, default="False",
                        help='whether to do train-val split')
    parser.add_argument('--split-num', type=int, default=50,
                        help='number of documents in validation set')

    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())