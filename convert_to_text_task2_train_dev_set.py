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

data_dir_1 = "./RDoC_raw_data/RDoCTask/RDoCTask2TrainData/Combined_Batch/"
data_output_dir_1 = "./datasets/Task2_without_acronym/"

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
	return " ".join(doc_tokens)


if not os.path.exists(data_output_dir_1):
	os.makedirs(data_output_dir_1)


total_docs = {RDoC_name:[] for RDoC_name in RDoC_contruct_fnames}
total_titles = {RDoC_name:[] for RDoC_name in RDoC_contruct_fnames}
total_labels = {RDoC_name:[] for RDoC_name in RDoC_contruct_fnames}
total_ids = {RDoC_name:[] for RDoC_name in RDoC_contruct_fnames}
total_relevant_context = {RDoC_name:[] for RDoC_name in RDoC_contruct_fnames}

for RDoC_name in RDoC_contruct_fnames:
	xl = pd.read_excel(open(data_dir_1 + RDoC_name + ".xlsx", 'rb'))
	total_ids[RDoC_name].extend(xl['pmid'].tolist())
	total_titles[RDoC_name].extend(xl['title'].tolist())
	total_docs[RDoC_name].extend(xl['abstract'].tolist())
	total_relevant_context[RDoC_name].extend(xl['Relevant Context'].tolist())
	total_labels[RDoC_name].extend([RDoC_name] * len(total_docs[RDoC_name]))

for RDoC_name in RDoC_contruct_fnames:
	
	temp_relevance_context = []
	for rel_con in total_relevant_context[RDoC_name]:
		temp_rel_con = [sent.strip("'") for sent in rel_con[1:-1].split("', '")]
		temp_relevance_context.append("\t".join(temp_rel_con))
	total_relevant_context[RDoC_name] = temp_relevance_context

	temp_docs = []
	for doc in total_docs[RDoC_name]:
		temp_doc_sents = nltk.sent_tokenize(doc.strip())
		temp_docs.append("\t".join(temp_doc_sents))
	total_docs[RDoC_name] = temp_docs

with open(data_dir_1 + "/total_docs_sents.txt", "w") as f:
	for RDoC_name in RDoC_contruct_fnames:
		f.write("======================================================================\n")
		f.write(RDoC_name + "\n")
		f.write("======================================================================\n")
		for doc, id in zip(total_docs[RDoC_name], total_ids[RDoC_name]):
			f.write(str(id) + "\n\n")
			f.write("\n".join(doc.split("\t")))
			f.write("\n\n")

with open(data_dir_1 + "/total_docs_relevant_contexts.txt", "w") as f:
	for RDoC_name in RDoC_contruct_fnames:
		f.write("======================================================================\n")
		f.write(RDoC_name + "\n")
		f.write("======================================================================\n")
		for rel_con, id in zip(total_relevant_context[RDoC_name], total_ids[RDoC_name]):
			f.write(str(id) + "\n\n")
			f.write("\n".join(rel_con.split("\t")))
			f.write("\n\n")


train_docs = []
train_titles = []
train_labels = []
train_ids = []
train_relevant_context = []

val_docs = []
val_titles = []
val_labels = []
val_ids = []
val_relevant_context = []


for RDoC_name in RDoC_contruct_fnames:

	docs_train, docs_val, labels_train, labels_val, titles_train, titles_val, ids_train, ids_val, rc_train, rc_val \
		= train_test_split(total_docs[RDoC_name], total_labels[RDoC_name], total_titles[RDoC_name], total_ids[RDoC_name], total_relevant_context[RDoC_name], test_size=0.2, random_state=123)
	
	train_docs.extend(docs_train)
	train_titles.extend(titles_train)
	train_labels.extend(labels_train)
	train_ids.extend(ids_train)
	train_relevant_context.extend(rc_train)

	val_docs.extend(docs_val)
	val_titles.extend(titles_val)
	val_labels.extend(labels_val)
	val_ids.extend(ids_val)
	val_relevant_context.extend(rc_val)

with open(data_dir_1 + "/train_docs.txt", "w") as f:
	for doc, title, id, label, rel_con in zip(train_docs, train_titles, train_ids, train_labels, train_relevant_context):
		doc_String = str(id) + "<<>>" + label + "<<>>" + title + "<<>>" + doc + "<<>>" + rel_con
		f.write(doc_String + "\n")

with open(data_dir_1 + "/val_docs.txt", "w") as f:
	for doc, title, id, label, rel_con in zip(val_docs, val_titles, val_ids, val_labels, val_relevant_context):
		doc_String = str(id) + "<<>>" + label + "<<>>" + title + "<<>>" + doc + "<<>>" + rel_con
		f.write(doc_String + "\n")


with open(data_output_dir_1 + "train_docs.txt", "w") as f:
	for id, label, title, doc, rel_con in zip(train_ids, train_labels, train_titles, train_docs, train_relevant_context):
		doc = "\t".join([remove_punctuation_and_replace_num(sent) for sent in doc.split("\t") if remove_punctuation_and_replace_num(sent)])
		rel_con = "\t".join([remove_punctuation_and_replace_num(sent) for sent in rel_con.split("\t") if remove_punctuation_and_replace_num(sent)])
		title = remove_punctuation_and_replace_num(title)
		f.write(str(id) + "<<>>" + label + "<<>>" + title + "<<>>" + doc + "<<>>" + rel_con + "\n")

with open(data_output_dir_1 + "val_docs.txt", "w") as f:
	for id, label, title, doc, rel_con in zip(val_ids, val_labels, val_titles, val_docs, val_relevant_context):
		doc = "\t".join([remove_punctuation_and_replace_num(sent) for sent in doc.split("\t") if remove_punctuation_and_replace_num(sent)])
		rel_con = "\t".join([remove_punctuation_and_replace_num(sent) for sent in rel_con.split("\t") if remove_punctuation_and_replace_num(sent)])
		title = remove_punctuation_and_replace_num(title)
		f.write(str(id) + "<<>>" + label + "<<>>" + title + "<<>>" + doc + "<<>>" + rel_con + "\n")

with open(data_output_dir_1 + "test_docs.txt", "w") as f:
	for id, label, title, doc, rel_con in zip(val_ids, val_labels, val_titles, val_docs, val_relevant_context):
		doc = "\t".join([remove_punctuation_and_replace_num(sent) for sent in doc.split("\t") if remove_punctuation_and_replace_num(sent)])
		rel_con = "\t".join([remove_punctuation_and_replace_num(sent) for sent in rel_con.split("\t") if remove_punctuation_and_replace_num(sent)])
		title = remove_punctuation_and_replace_num(title)
		f.write(str(id) + "<<>>" + label + "<<>>" + title + "<<>>" + doc + "<<>>" + rel_con + "\n")

"""
train_docs = [remove_punctuation_and_replace_num(doc) for doc in train_docs]
val_docs = [remove_punctuation_and_replace_num(doc) for doc in val_docs]

train_titles = [remove_punctuation_and_replace_num(doc) for doc in train_titles]
val_titles = [remove_punctuation_and_replace_num(doc) for doc in val_titles]

#train_docs, train_titles, train_labels = shuffle(train_docs, train_titles, train_labels, random_state=123)
#val_docs, val_titles, val_labels = shuffle(val_docs, val_titles, val_labels, random_state=123)

with open(data_output_dir_1 + "train_docs.txt", "w") as f:
	f.write("\n".join([label + "\t" + title + " " + doc for doc, title, label in zip(train_docs, train_titles, train_labels)]))

with open(data_output_dir_1 + "train_ids.txt", "w") as f:
	f.write("\n".join([str(id) for id in train_ids]))

with open(data_output_dir_1 + "val_docs.txt", "w") as f:
	f.write("\n".join([label + "\t" + title + " " + doc for doc, title, label in zip(val_docs, val_titles, val_labels)]))

with open(data_output_dir_1 + "val_ids.txt", "w") as f:
	f.write("\n".join([str(id) for id in val_ids]))

with open(data_output_dir_1 + "test_docs.txt", "w") as f:
	f.write("\n".join([label + "\t" + title + " " + doc for doc, title, label in zip(val_docs, val_titles, val_labels)]))

with open(data_output_dir_1 + "test_ids.txt", "w") as f:
	f.write("\n".join([str(id) for id in val_ids]))
"""