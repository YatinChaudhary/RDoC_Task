import os, sys, csv, re
import numpy as np
import pandas as pd
import string
import nltk
from sklearn.model_selection import train_test_split
from nltk.tokenize import RegexpTokenizer
from sklearn.utils import shuffle

tokenizer = RegexpTokenizer(r'\w+')

cachedStopWords = []
with open("english_stopwords.txt", "r") as f:
	cachedStopWords.extend([line.strip() for line in f.readlines()])

data_dir_1 = "./RDoC_raw_data/RDoCTask/RDoCTask2TestData/Combined_batch/"
data_output_dir_1 = "./datasets/Task2_test_data_combined_batch_without_acronym/"

RDoC_contruct_fnames = [fname.split(".")[0] for fname in os.listdir(data_dir_1)]

FLOAT_REGEXP = re.compile(r'^[-+]?([0-9]+|[0-9]*\.[0-9]+)([eE][-+]?[0-9]+)?$')

CAPS_REMOVE_LIST = ["APPROACH", "BACKGROUND", "COMPARISON", "CONCLUSION", "CONCLUSIONS", "FINDINGS", "IMPLICATIONS", 
					"INTRODUCTION", "LIMITATIONS", "MEASURES", "METHODOLOGY", "METHOD", "METHODS", "OBJECTIVE", 
					"OBJECTIVES", "OUTCOME", "PURPOSE", "RESULTS", "RESEARCH", "SIGNIFICANCE", "STATEMENT", "STUDY", "SUMMARY"]

def is_float(str):
	return True if FLOAT_REGEXP.match(str) else False

def replace_num(tokens):
	new_tokens = []
	for token in tokens:
		if is_float(token):
			new_tokens.append("<num>")
		else:
			new_tokens.append(token)
	return new_tokens

def preprocess_token(token):
	return "".join([char for char in token if char.isalpha()])

def remove_punctuation_and_replace_num(doc):
	doc_tokens = [token.strip(string.punctuation) for token in tokenizer.tokenize(doc)]
	doc_tokens = [token for token in doc_tokens if not token.isupper()]
	doc_tokens = replace_num(doc_tokens)
	doc_tokens = [preprocess_token(token.lower()) for token in doc_tokens]
	doc_tokens = [token for token in doc_tokens if not token in cachedStopWords]
	#if len(doc_tokens) > 1:
	#	return " ".join(doc_tokens)
	#else:
	#	return ''
	if len(doc_tokens) == 1:
		print(doc_tokens)
	return " ".join(doc_tokens)


if not os.path.exists(data_output_dir_1):
	os.makedirs(data_output_dir_1)


total_docs = {RDoC_name:[] for RDoC_name in RDoC_contruct_fnames}
total_titles = {RDoC_name:[] for RDoC_name in RDoC_contruct_fnames}
total_labels = {RDoC_name:[] for RDoC_name in RDoC_contruct_fnames}
total_ids = {RDoC_name:[] for RDoC_name in RDoC_contruct_fnames}

for RDoC_name in RDoC_contruct_fnames:
	xl = pd.read_excel(open(data_dir_1 + RDoC_name + ".xlsx", 'rb'))
	total_ids[RDoC_name].extend(xl['pmid'].tolist())
	total_titles[RDoC_name].extend(xl['title'].tolist())
	total_docs[RDoC_name].extend(xl['abstract'].tolist())
	total_labels[RDoC_name].extend([RDoC_name] * len(total_docs[RDoC_name]))

for RDoC_name in RDoC_contruct_fnames:

	temp_docs = []
	for doc in total_docs[RDoC_name]:
		temp_doc_sents = nltk.sent_tokenize(doc.strip())
		temp_docs.append("\t".join(temp_doc_sents))
	total_docs[RDoC_name] = temp_docs

with open(data_dir_1 + "/test_docs_sents.txt", "w") as f:
	for RDoC_name in RDoC_contruct_fnames:
		f.write("======================================================================\n")
		f.write(RDoC_name + "\n")
		f.write("======================================================================\n")
		for doc, id in zip(total_docs[RDoC_name], total_ids[RDoC_name]):
			f.write(str(id) + "\n\n")
			f.write("\n".join(doc.split("\t")))
			f.write("\n\n")


test_docs = []
test_titles = []
test_labels = []
test_ids = []

for RDoC_name in RDoC_contruct_fnames:
	
	test_docs.extend(total_docs[RDoC_name])
	test_titles.extend(total_titles[RDoC_name])
	test_labels.extend(total_labels[RDoC_name])
	test_ids.extend(total_ids[RDoC_name])

with open(data_dir_1 + "/test_docs.txt", "w") as f:
	for doc, title, id, label in zip(test_docs, test_titles, test_ids, test_labels):
		doc_String = str(id) + "<<>>" + label + "<<>>" + title + "<<>>" + doc + "<<>>" + "dummy"
		f.write(doc_String + "\n")


with open(data_output_dir_1 + "test_docs.txt", "w") as f:
	for id, label, title, doc in zip(test_ids, test_labels, test_titles, test_docs):
		doc = "\t".join([remove_punctuation_and_replace_num(sent) for sent in doc.split("\t") if remove_punctuation_and_replace_num(sent)])
		title = remove_punctuation_and_replace_num(title)
		f.write(str(id) + "<<>>" + label + "<<>>" + title + "<<>>" + doc + "<<>>" + "dummy" + "\n")