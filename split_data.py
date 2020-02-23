import os, sys, csv, re
import numpy as np
import pandas as pd
import string
from sklearn.model_selection import train_test_split
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')

data_dir_1 = "./RDoC_raw_data/RDoCTask/RDoCTask1TrainData/Combined_Batch/"
#data_dir_1 = "./RDoC_raw_data/RDoCTask_Augmented_Def_SampleData/RDoCTask1TrainData/Combined_Batch/"
#data_dir_1 = "./RDoC_raw_data/RDoCTask_Augmented_Def/RDoCTask1TrainData/Combined_Batch/"

RDoC_contruct_fnames = [fname.split(".")[0] for fname in os.listdir(data_dir_1)]


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

#import pdb; pdb.set_trace()

with open(data_dir_1 + "train_docs.txt", "w") as f:
	f.write("\n".join([str(id) + "<<>>" + label + "<<>>" + title + "<<>>" + doc for id, doc, title, label in zip(train_ids, train_docs, train_titles, train_labels)]))

with open(data_dir_1 + "val_docs.txt", "w") as f:
	f.write("\n".join([str(id) + "<<>>" + label + "<<>>" + title + "<<>>" + doc for id, doc, title, label in zip(val_ids, val_docs, val_titles, val_labels)]))