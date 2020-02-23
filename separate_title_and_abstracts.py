import os, sys, csv, re
import numpy as np
import pandas as pd
import string
from sklearn.model_selection import train_test_split
from nltk.tokenize import RegexpTokenizer
from sklearn.utils import shuffle

seed = 42
np.random.seed(seed)

tokenizer = RegexpTokenizer(r'\w+')

data_dir_1 = "./RDoC_raw_data/RDoCTask/RDoCTask1TrainData/Combined_Batch/"
#data_dir_1 = "./RDoC_raw_data/RDoCTask_Augmented_Def_SampleData/RDoCTask1TrainData/Combined_Batch/"
#data_dir_1 = "./RDoC_raw_data/RDoCTask_Augmented_Def/RDoCTask1TrainData/Combined_Batch/"
#data_dir_1 = "./RDoC_raw_data/RDoCTask_Augmented_Def_only_3/RDoCTask1TrainData/Combined_Batch/"
data_output_dir_1 = "./datasets/Task1_without_acronym/"
#data_output_dir_1 = "./datasets/Task1_Augmented_Def_SampleData/"
#data_output_dir_1 = "./datasets/Task1_Augmented_Def/"
#data_output_dir_1 = "./datasets/Task1_Augmented_Def_only_3_new/"

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
	#doc_tokens = [token.strip(string.punctuation) for token in doc.split()]
	doc_tokens = [token.strip(string.punctuation) for token in tokenizer.tokenize(doc)]
	#all_caps_tokens = [token for token in doc_tokens if token.isupper()]
	doc_tokens = [token for token in doc_tokens if not token.isupper()]
	#doc_tokens = [token for token in doc_tokens if not token in CAPS_REMOVE_LIST]
	#doc_tokens = [token for token in replace_num(doc_tokens)]
	doc_tokens = replace_num(doc_tokens)
	doc_tokens = [preprocess_token(token.lower()) for token in doc_tokens]
	#return " ".join(doc_tokens), all_caps_tokens
	return " ".join(doc_tokens)

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


"""
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
"""

if not os.path.exists(data_output_dir_1):
	os.makedirs(data_output_dir_1)

train_docs = []
train_titles = []
train_labels = []
train_ids = []

val_docs = []
val_titles = []
val_labels = []
val_ids = []

"""
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
"""

with open(data_dir_1 + "train_docs.txt", "r") as f:
	for line in f.readlines():
		id, label, title, doc = line.strip().split("<<>>")
		train_docs.append(doc)
		train_titles.append(title)
		train_labels.append(label)
		train_ids.append(id)

with open(data_dir_1 + "val_docs.txt", "r") as f:
	for line in f.readlines():
		id, label, title, doc = line.strip().split("<<>>")
		val_docs.append(doc)
		val_titles.append(title)
		val_labels.append(label)
		val_ids.append(id)

train_docs = [remove_punctuation_and_replace_num(doc) for doc in train_docs]
val_docs = [remove_punctuation_and_replace_num(doc) for doc in val_docs]

train_titles = [remove_punctuation_and_replace_num(doc) for doc in train_titles]
val_titles = [remove_punctuation_and_replace_num(doc) for doc in val_titles]

train_docs, train_titles, train_labels = shuffle(train_docs, train_titles, train_labels, random_state=123)
val_docs, val_titles, val_labels = shuffle(val_docs, val_titles, val_labels, random_state=123)
#test_docs, test_docs_labels = shuffle(test_docs, test_docs_labels, random_state=123)

#import pdb; pdb.set_trace()

with open(data_output_dir_1 + "/vocab_docnade.vocab", "r") as f:
	docnade_vocab = [line.strip() for line in f.readlines()]
vocab_to_id = dict(zip(docnade_vocab, range(len(docnade_vocab))))

with open(data_output_dir_1 + "/labels.txt", "r") as f:
	labels = [line.strip() for line in f.readlines()]
labels = dict(zip(labels, range(len(labels))))


with open(data_output_dir_1 + "training_docnade_abstracts.csv", "w", newline='') as f:
	csv_writer = csv.writer(f, delimiter=',')
	for doc, label in zip(train_docs, train_labels):
		new_doc = preprocess(doc, vocab_to_id, "docnade")
		new_label = labels[label.lower().strip()]
		csv_writer.writerow([new_label, new_doc])

with open(data_output_dir_1 + "training_docnade_titles.csv", "w", newline='') as f:
	csv_writer = csv.writer(f, delimiter=',')
	for title, label in zip(train_titles, train_labels):
		new_title = preprocess(title, vocab_to_id, "docnade")
		new_label = labels[label.lower().strip()]
		csv_writer.writerow([new_label, new_title])

with open(data_output_dir_1 + "validation_docnade_abstracts.csv", "w", newline='') as f:
	csv_writer = csv.writer(f, delimiter=',')
	for doc, label in zip(val_docs, val_labels):
		new_doc = preprocess(doc, vocab_to_id, "docnade")
		new_label = labels[label.lower().strip()]
		csv_writer.writerow([new_label, new_doc])

with open(data_output_dir_1 + "validation_docnade_titles.csv", "w", newline='') as f:
	csv_writer = csv.writer(f, delimiter=',')
	for title, label in zip(val_titles, val_labels):
		new_title = preprocess(title, vocab_to_id, "docnade")
		new_label = labels[label.lower().strip()]
		csv_writer.writerow([new_label, new_title])

with open(data_output_dir_1 + "test_docnade_abstracts.csv", "w", newline='') as f:
	csv_writer = csv.writer(f, delimiter=',')
	for doc, label in zip(val_docs, val_labels):
		new_doc = preprocess(doc, vocab_to_id, "docnade")
		new_label = labels[label.lower().strip()]
		csv_writer.writerow([new_label, new_doc])

with open(data_output_dir_1 + "test_docnade_titles.csv", "w", newline='') as f:
	csv_writer = csv.writer(f, delimiter=',')
	for title, label in zip(val_titles, val_labels):
		new_title = preprocess(title, vocab_to_id, "docnade")
		new_label = labels[label.lower().strip()]
		csv_writer.writerow([new_label, new_title])

"""
with open(data_output_dir_1 + "test_docs.txt", "w") as f:
	f.write("\n".join([label + "\t" + title + " " + doc for doc, title, label in zip(test_docs, test_titles, test_labels)]))

with open(data_output_dir_1 + "test_ids.txt", "w") as f:
	f.write("\n".join([str(id) for id in test_ids]))
"""