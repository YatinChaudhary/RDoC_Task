import os, sys, csv, re
import numpy as np
import pandas as pd
import string
from sklearn.model_selection import train_test_split
from nltk.tokenize import RegexpTokenizer
from sklearn.utils import shuffle

tokenizer = RegexpTokenizer(r'\w+')

data_dir_1_train_dev = "./RDoC_raw_data/RDoCTask/RDoCTask1TrainData/Combined_Batch/"
data_dir_1_test = "./RDoC_raw_data/RDoCTask/RDoCTask1TestData/Combined_Batch/"

data_output_dir_1_train_dev = "./datasets/Task1_without_acronym/"
data_output_dir_1_test = "./datasets/Task1_test_data_without_acronym/"


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
	#doc_tokens = [token for token in doc_tokens if not token in CAPS_REMOVE_LIST]
	doc_tokens = replace_num(doc_tokens)
	doc_tokens = [preprocess_token(token.lower()) for token in doc_tokens]
	return " ".join(doc_tokens)

## Train/Dev data split and pre-processing

RDoC_contruct_fnames = [fname.split(".")[0] for fname in os.listdir(data_dir_1_train_dev) if fname.endswith(".xlsx")]

total_docs = {RDoC_name:[] for RDoC_name in RDoC_contruct_fnames}
total_titles = {RDoC_name:[] for RDoC_name in RDoC_contruct_fnames}
total_labels = {RDoC_name:[] for RDoC_name in RDoC_contruct_fnames}
total_ids = {RDoC_name:[] for RDoC_name in RDoC_contruct_fnames}

for RDoC_name in RDoC_contruct_fnames:
	xl = pd.read_excel(open(data_dir_1_train_dev + RDoC_name + ".xlsx", 'rb'))
	total_ids[RDoC_name].extend(xl['pmid'].tolist())
	total_titles[RDoC_name].extend(xl['title'].tolist())
	total_docs[RDoC_name].extend(xl['abstract'].tolist())
	total_labels[RDoC_name].extend([RDoC_name] * len(total_docs[RDoC_name]))

train_docs = []
train_titles = []
train_labels = []
train_ids = []

val_docs = []
val_titles = []
val_labels = []
val_ids = []

for RDoC_name in RDoC_contruct_fnames:
	
	docs_train, docs_val, labels_train, labels_val, titles_train, titles_val, ids_train, ids_val \
		= train_test_split(total_docs[RDoC_name], total_labels[RDoC_name], total_titles[RDoC_name], total_ids[RDoC_name], test_size=0.2, random_state=123)

	train_docs.extend(docs_train)
	train_titles.extend(titles_train)
	train_labels.extend(labels_train)
	train_ids.extend(ids_train)

	val_docs.extend(docs_val)
	val_titles.extend(titles_val)
	val_labels.extend(labels_val)
	val_ids.extend(ids_val)

with open(data_dir_1_train_dev + "train_docs.txt", "w") as f:
	for id, label, title, doc in zip(train_ids, train_labels, train_titles, train_docs):
		f.write(str(id) + "<<>>" + label + "<<>>" + title + "<<>>" + doc + "\n")

with open(data_dir_1_train_dev + "val_docs.txt", "w") as f:
	for id, label, title, doc in zip(val_ids, val_labels, val_titles, val_docs):
		f.write(str(id) + "<<>>" + label + "<<>>" + title + "<<>>" + doc + "\n")

train_docs = [remove_punctuation_and_replace_num(doc) for doc in train_docs]
val_docs = [remove_punctuation_and_replace_num(doc) for doc in val_docs]

train_titles = [remove_punctuation_and_replace_num(doc) for doc in train_titles]
val_titles = [remove_punctuation_and_replace_num(doc) for doc in val_titles]

train_docs, train_titles, train_labels = shuffle(train_docs, train_titles, train_labels, random_state=123)
val_docs, val_titles, val_labels = shuffle(val_docs, val_titles, val_labels, random_state=123)

if not os.path.exists(data_output_dir_1_train_dev):
	os.makedirs(data_output_dir_1_train_dev)

with open(data_output_dir_1_train_dev + "train_docs.txt", "w") as f:
	f.write("\n".join([label + "\t" + title + " " + doc for doc, title, label in zip(train_docs, train_titles, train_labels)]))

with open(data_output_dir_1_train_dev + "train_ids.txt", "w") as f:
	f.write("\n".join([str(id) for id in train_ids]))

with open(data_output_dir_1_train_dev + "val_docs.txt", "w") as f:
	f.write("\n".join([label + "\t" + title + " " + doc for doc, title, label in zip(val_docs, val_titles, val_labels)]))

with open(data_output_dir_1_train_dev + "val_ids.txt", "w") as f:
	f.write("\n".join([str(id) for id in val_ids]))


## Test data pre-processing

RDoC_contruct_fnames_test = [fname.split(".")[0] for fname in os.listdir(data_dir_1_test) if fname.endswith(".xlsx")]

total_docs = {RDoC_name:[] for RDoC_name in RDoC_contruct_fnames_test}
total_titles = {RDoC_name:[] for RDoC_name in RDoC_contruct_fnames_test}
total_labels = {RDoC_name:[] for RDoC_name in RDoC_contruct_fnames_test}
total_ids = {RDoC_name:[] for RDoC_name in RDoC_contruct_fnames_test}

with open(data_dir_1_test + "test_docs.txt", "w") as f:
	pass

for RDoC_name in RDoC_contruct_fnames_test:
	xl = pd.read_excel(open(data_dir_1_test + RDoC_name + ".xlsx", 'rb'))
	total_ids[RDoC_name].extend(xl['pmid'].tolist())
	total_titles[RDoC_name].extend(xl['title'].tolist())
	total_docs[RDoC_name].extend(xl['abstract'].tolist())
	total_labels[RDoC_name].extend([RDoC_name] * len(total_docs[RDoC_name]))
	
	with open(data_dir_1_test + "test_docs.txt", "a") as f:
		for id, label, title, doc in zip(total_ids[RDoC_name], total_labels[RDoC_name], total_titles[RDoC_name], total_docs[RDoC_name]):
			f.write(str(id) + "<<>>" + label + "<<>>" + title + "<<>>" + doc + "\n")

test_docs = []
test_titles = []
test_labels = []
test_ids = []

with open(data_dir_1_test + "test_docs.txt", "r") as f:
	for line in f.readlines():
		id, label, title, doc = line.strip().split("<<>>")
		test_docs.append(doc)
		test_titles.append(title)
		test_labels.append(label)
		test_ids.append(id)

test_docs = [remove_punctuation_and_replace_num(doc) for doc in test_docs]
test_titles = [remove_punctuation_and_replace_num(doc) for doc in test_titles]

new_test_labels = []
for label in test_labels:
	if label == "Potential_Threat_Anxiety":
		new_test_labels.append("Potential_Threat_Anxiety_")
	else:
		new_test_labels.append(label)

if not os.path.exists(data_output_dir_1_test):
	os.makedirs(data_output_dir_1_test)

with open(data_output_dir_1_test + "test_docs.txt", "w") as f:
	#f.write("\n".join([label + "\t" + title + " " + doc for doc, title, label in zip(test_docs, test_titles, test_labels)]))
	f.write("\n".join([label + "\t" + title + " " + doc for doc, title, label in zip(test_docs, test_titles, new_test_labels)]))

with open(data_output_dir_1_test + "test_ids.txt", "w") as f:
	f.write("\n".join([str(id) for id in test_ids]))

with open(data_output_dir_1_train_dev + "test_docs.txt", "w") as f:
	#f.write("\n".join([label + "\t" + title + " " + doc for doc, title, label in zip(test_docs, test_titles, test_labels)]))
	f.write("\n".join([label + "\t" + title + " " + doc for doc, title, label in zip(test_docs, test_titles, new_test_labels)]))

with open(data_output_dir_1_train_dev + "test_ids.txt", "w") as f:
	f.write("\n".join([str(id) for id in test_ids]))