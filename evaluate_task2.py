import tensorflow as tf
import numpy as np
import os, sys
import pickle
import csv
import operator
import model.evaluate as eval
import model.data as data
import codecs
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.svm import SVC, LinearSVC
import sklearn.metrics.pairwise as pw
import BM25_task2 as BM25

np.random.seed(1234)

home_dir = os.getenv("HOME")

maxInt = sys.maxsize
decrement = True

while decrement:
	# decrease the maxInt value by factor 10 
	# as long as the OverflowError occurs.

	decrement = False
	try:
		csv.field_size_limit(maxInt)
	except OverflowError:
		maxInt = int(maxInt/10)
		decrement = True


params = {}
params['dataset'] = "./datasets/Task2_without_acronym"
params['results_path'] = "./Results_Task2_SiameseLSTM/val_sim_scores_label_title.txt"

log_dir = os.path.join(params['dataset'], 'logs')

if not os.path.exists(log_dir):
	os.makedirs(log_dir)

val_docs = []
val_relevant_context = []
val_labels = []
val_titles = []
val_ids = []

with open(params['dataset'] + "/val_docs.txt", "r") as f:
	for line in f.readlines():
		id, label, title, doc, rel_con = line.strip().split("<<>>")
		val_docs.append(doc.split("\t"))
		val_relevant_context.append(rel_con.split("\t"))
		val_labels.append(" ".join(label.lower().strip().split("_")))
		val_titles.append(title)
		val_ids.append(id)

labels = []
with open("./datasets/Task1_Augmented_Def_new/labels.txt", "r") as f:
	for line in f.readlines():
		label_tokens = line.lower().strip().split("_")
		label_tokens = [token for token in label_tokens if token]
		query = " ".join(label_tokens)
		labels.append(query)

validation_ids = []
validation_labels = []
for i, doc in enumerate(val_docs):
	for sent in doc:
		temp_label = 0.0
		for rc_sent in val_relevant_context[i]:
			if rc_sent in sent:
				temp_label = 1.0
		validation_labels.append(temp_label)
		validation_ids.append(val_ids[i])

with open(params['results_path'], "r") as f:
	pred_probs = [float(line.strip()) for line in f.readlines()]

temp_pred_labels = np.array(validation_labels, dtype=np.int8)
temp_pred_probs = np.array(pred_probs, dtype=np.float32)

unique_val_ids = np.unique(validation_ids)
val_docs_pred_labels = {id:[] for id in unique_val_ids}
val_docs_pred_probs = {id:[] for id in unique_val_ids}
for i, id in enumerate(validation_ids):
	val_docs_pred_labels[id].append(temp_pred_labels[i])
	val_docs_pred_probs[id].append(temp_pred_probs[i])

#import pdb; pdb.set_trace()

val_pred_list = []
for id in unique_val_ids:
	max_index = np.argmax(val_docs_pred_probs[id])
	val_pred_list.append(val_docs_pred_labels[id][max_index])

#import pdb; pdb.set_trace()

#acc_score = accuracy_score(np.ones(len(val_pred_list), dtype=np.int8), np.array(val_pred_list))
acc_score = np.mean(val_pred_list)

#precision, recall, f1, support = precision_recall_fscore_support(np.ones(len(val_pred_list), dtype=np.int8), np.array(val_pred_list), average='macro')

#print("Test acc_score: ", acc_score)
print("Validation acc_scores: ", acc_score)
#print("Test precision: ", precision)
#print("Test recall: ", recall)
#print("Test f1: ", f1)
#print("Test support: ", support)

with open(os.path.join(log_dir, "info.txt"), "w") as f:
	#f.write("\n\nBest accuracy score: %s" % (acc_score))
	f.write("\n\nValidation accuracy scores: %s" % (acc_score))
	#f.write("\n\nBest precision score: %s" % (precision))
	#f.write("\n\nBest recall score: %s" % (recall))
	#f.write("\n\nBest f1 score: %s" % (f1))
	#f.write("\n\nBest support score: %s" % (support))

#import pdb; pdb.set_trace()

# logging information
with open(os.path.join(log_dir, "reload_info_clusters_with_siameseLSTM.txt"), "w") as f:
	for doc_counter, query, id in zip(range(len(val_pred_list)), val_labels, val_ids):
		sorted_indices = np.argsort(val_docs_pred_probs[id])[::-1]
		relevance_labels = np.array(val_docs_pred_labels[id])[sorted_indices]
		relevance_scores = np.array(val_docs_pred_probs[id])[sorted_indices]
		f.write("Doc " + str(doc_counter) + ": " + str(query) + ": " + str(id) + "\n\n")
		f.write("Predicted_labels:  " + " ".join([str(int(l)) for l in relevance_labels]) + "\n")
		f.write("Predicted_probs:   " + " ".join([str(l) for l in relevance_scores]) + "\n")
		f.write("Sentence indices:  " + " ".join([str(l) for l in sorted_indices]) + "\n")
		f.write("\n=================================================================================\n\n")

print("Done.")