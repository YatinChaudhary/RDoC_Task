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

#reload(sys)  # Reload does the trick!
#sys.setdefaultencoding('UTF8')

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


def perform_classification_test(train_data, test_data, c_list, classification_model="logistic", norm_before_classification=False):
	docVectors_train, train_labels = train_data
	docVectors_test, test_labels = test_data

	if norm_before_classification:
		"""
		temp_mat = np.vstack((docVectors_train, docVectors_test))

		mean = np.mean(temp_mat, axis=0)
		std = np.std(temp_mat, axis=0)
		temp_mat_normed = (temp_mat - mean) / std

		docVectors_train = temp_mat_normed[:len(docVectors_train), :]
		docVectors_test = temp_mat_normed[-len(docVectors_test):, :]
		"""

		mean = np.mean(np.vstack((docVectors_train, docVectors_test)), axis=0)
		std = np.std(np.vstack((docVectors_train, docVectors_test)), axis=0)

		docVectors_train = (docVectors_train - mean) / std
		docVectors_test = (docVectors_test - mean) / std

	#import pdb; pdb.set_trace()

	## Classification Accuracy
	test_acc = []
	test_f1  = []

	test_pred_labels = []
	test_pred_probs = []
	
	for c in c_list:
		if classification_model == "logistic":
			clf = LogisticRegression(C=c)
		elif classification_model == "svm":
			#clf = SVC(C=c, kernel='precomputed')
			#clf = SVC(C=c)
			clf = SVC(C=c, probability=True)
		
		clf.fit(docVectors_train, train_labels)
		pred_test_labels = clf.predict(docVectors_test)
		pred_test_probs = clf.predict_proba(docVectors_test)

		acc_test = accuracy_score(test_labels, pred_test_labels)
		#f1_test = precision_recall_fscore_support(test_labels, pred_test_labels, pos_label=None, average='macro')[2]

		test_acc.append(acc_test)
		#test_f1.append(f1_test)

		test_pred_labels.append(pred_test_labels)
		test_pred_probs.append(pred_test_probs)

	if classification_model == "logistic":
		return test_acc, test_f1
	elif classification_model == "svm":
		#return test_acc, test_f1, pred_test_probs
		return test_acc, test_f1, test_pred_probs, test_pred_labels


def reload_evaluation_f1(params, training_vectors, validation_vectors, training_labels, 
						validation_labels, training_ids, validation_ids, val_labels, val_ids, suffix=""):

	### Classification - F1

	dataset = data.Dataset(params['dataset'])
	log_dir = os.path.join(params['model'], 'logs')

	if not os.path.exists(log_dir):
		os.makedirs(log_dir)

	#c_list = [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 3.0, 5.0, 10.0, 100.0, 500.0, 1000.0, 10000.0]
	#c_list = [1.0, 3.0, 5.0, 10.0, 100.0, 500.0, 1000.0, 10000.0]
	c_list = [1000.0]

	test_acc = []
	test_f1 = []
	val_acc = []
	val_f1 = []

	test_acc_W = []
	test_f1_W = []
	val_acc_W = []
	val_f1_W = []

	#y_train = training_labels
	#y_test = validation_labels
	
	train_data = (np.array(training_vectors, dtype=np.float32), np.array(training_labels, dtype=np.int32))
	test_data = (np.array(validation_vectors, dtype=np.float32), np.array(validation_labels, dtype=np.int32))

	test_acc, test_f1, pred_probs, pred_labels = perform_classification_test(train_data, test_data, c_list, classification_model="svm", norm_before_classification=False)
	
	with open(os.path.join(log_dir, "info.txt"), "w") as f:
		f.write("\n\nTest accuracy with h vector IR: %s" % (test_acc))
	
	with open(os.path.join(log_dir, "info.txt"), "a") as f:
		f.write("\n\nTest F1 score with h vector IR: %s" % (test_f1))

	print("Test acc: ", test_acc)
	print("Test f1: ", test_f1)

	#import pdb; pdb.set_trace()

	acc_scores = []
	best_acc_score = -1.0
	best_val_docs_pred_labels = {}
	best_val_docs_pred_probs = {}
	for j in range(len(c_list)):
		temp_pred_labels = np.array(validation_labels, dtype=np.int32)
		temp_pred_probs = pred_probs[j][:, 1]

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

		acc_scores.append(acc_score)

		if acc_score > best_acc_score:
			best_acc_score = acc_score
			best_val_docs_pred_labels = val_docs_pred_labels
			best_val_docs_pred_probs = val_docs_pred_probs

		#precision, recall, f1, support = precision_recall_fscore_support(np.ones(len(val_pred_list), dtype=np.int8), np.array(val_pred_list), average='macro')

	#print("Test acc_score: ", acc_score)
	print("Validation acc_scores: ", acc_scores)
	#print("Test precision: ", precision)
	#print("Test recall: ", recall)
	#print("Test f1: ", f1)
	#print("Test support: ", support)

	with open(os.path.join(log_dir, "info.txt"), "a") as f:
		#f.write("\n\nBest accuracy score: %s" % (acc_score))
		f.write("\n\nValidation accuracy scores: %s" % (acc_scores))
		#f.write("\n\nBest precision score: %s" % (precision))
		#f.write("\n\nBest recall score: %s" % (recall))
		#f.write("\n\nBest f1 score: %s" % (f1))
		#f.write("\n\nBest support score: %s" % (support))

	#import pdb; pdb.set_trace()
	val_scores_dict = {}
	val_ids_dict = {}
	for label, id in zip(val_labels, val_ids):
		if not label in val_scores_dict:
			val_scores_dict[label] = []
			val_ids_dict[label] = []
		max_index = np.argmax(best_val_docs_pred_probs[id])
		pred_label = best_val_docs_pred_labels[id][max_index]
		val_scores_dict[label].append(pred_label)
		val_ids_dict[label].append(id)

	classwise_scores = []
	for label in val_scores_dict.keys():
		class_val_prec = np.mean(val_scores_dict[label])
		classwise_scores.append(class_val_prec)

	with open(log_dir + "/classwise_scores.txt", "w") as f:
		for label, scores in val_scores_dict.items():
			f.write("Label:  " + label + "\n")
			f.write("ids:    " + str(val_ids_dict[label]) + "\n")
			f.write("scores:  " + str(scores) + "\n")
			f.write("prec:  " + str(np.mean(scores)) + "\n")
			f.write("\n\n")

	# logging information
	with open(os.path.join(log_dir, "reload_info_clusters_with_SVM.txt"), "w") as f:
		for doc_counter, query, id in zip(range(len(val_pred_list)), val_labels, val_ids):
			sorted_indices = np.argsort(best_val_docs_pred_probs[id])[::-1]
			relevance_labels = np.array(best_val_docs_pred_labels[id])[sorted_indices]
			relevance_scores = np.array(best_val_docs_pred_probs[id])[sorted_indices]
			f.write("Doc " + str(doc_counter) + ": " + str(query) + ": " + str(id) + "\n\n")
			f.write("Predicted_labels:  " + " ".join([str(int(l)) for l in relevance_labels]) + "\n")
			f.write("Predicted_probs:   " + " ".join([str(l) for l in relevance_scores]) + "\n")
			f.write("Sentence indices:  " + " ".join([str(l) for l in sorted_indices]) + "\n")
			f.write("\n=================================================================================\n\n")

	


def BOW_representation(tokens, vocab_size):
	vec = np.zeros(vocab_size, dtype=np.float32)
	for token in tokens:
		vec[int(token)] += 1.0
	return vec

def get_bm25_ids(bm25, tokens):
	ids = []
	for token in tokens:
		try:
			ids.append(bm25.dictionary.token2id[token])
		except KeyError:
			pass
	return ids



params = {}
params['dataset'] = "./datasets/Task2_without_acronym"
params['model'] = "./model/Task2_without_acronym_supervised_SVM_classwise"
params['use_bio_prior'] = False
params['use_fasttext_prior'] = False
params['use_BOW_repesentation'] = False
params['use_DocNADE_W'] = False
params['split_abstract_title'] = False
params['attention_EmbSum_type'] = "sum"
params['use_label_as_query'] = True
params['use_title_as_query'] = False
params['label_title_weights'] = [0.5, 0.5]
params['label_title_max'] = False
params['use_bm25_extra'] = False
params['use_pos_vector'] = False

hidden_size = 200

params['use_bio_prior_for_training'] = False
params['use_fasttext_prior_for_training'] = False
params['use_BOW_repesentation_for_training'] = True

if params['use_label_as_query']:
	params['model'] += "_label"

if params['use_title_as_query']:
	params['model'] += "_title"

if params['use_bio_prior']:
	params['model'] += "_bioNLP"

if params['use_fasttext_prior']:
	params['model'] += "_fastText"

if params['use_BOW_repesentation']:
	params['model'] += "_BOW"

if params['use_BOW_repesentation_for_training']:
	params['model'] += "_BOW"

if params['use_bm25_extra']:
	params['model'] += "_bm25_extra"

if params['use_pos_vector']:
	params['model'] += "_position_vec"

#if params['label_title_max']:
#	params['model'] += "_max"
#elif params['use_label_as_query'] and params['use_title_as_query']:
#	params['model'] += str(params['label_title_weights'])

params['model'] += "_" + params['attention_EmbSum_type']

log_dir = os.path.join(params['model'], 'logs')

if not os.path.exists(log_dir):
	os.makedirs(log_dir)

train_docs = []
train_relevant_context = []
train_labels = []
train_titles = []
train_ids = []
train_bm25_scores = []

with open(params['dataset'] + "/train_docs.txt", "r") as f:
	for line in f.readlines():
		id, label, title, doc, rel_con = line.strip().split("<<>>")
		train_docs.append(doc.split("\t"))
		train_relevant_context.append(rel_con.split("\t"))
		train_labels.append(" ".join(label.lower().strip().split("_")))
		train_titles.append(title)
		train_ids.append(id)

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

"""
training_docs = []
for i, doc in enumerate(train_docs):
	for sent in doc:
		label = train_labels[i]
		title = train_titles[i]
		#query = label
		query = title
		#query = label + " " + title
		temp_relevance = 0.0
		for rc_sent in train_relevant_context[i]:
			if rc_sent in sent:
				temp_relevance = 1.0
		training_docs.append((sent, query, temp_relevance))

validation_docs = []
for i, doc in enumerate(val_docs):
	for sent in doc:
		label = val_labels[i]
		title = val_titles[i]
		#query = label
		query = title
		#query = label + " " + title
		temp_relevance = 0.0
		for rc_sent in val_relevant_context[i]:
			if rc_sent in sent:
				temp_relevance = 1.0
		validation_docs.append((sent, query, temp_relevance))

#with open('Task2_training_data_label_as_query.pkl', "wb") as f:
with open('Task2_training_data_title_as_query.pkl', "wb") as f:
#with open('Task2_training_data_label_title_as_query.pkl', "wb") as f:
	pickle.dump(training_docs, f)

#with open('Task2_validation_data_label_as_query.pkl', "wb") as f:
with open('Task2_validation_data_title_as_query.pkl', "wb") as f:
#with open('Task2_validation_data_label_title_as_query.pkl', "wb") as f:
	pickle.dump(validation_docs, f)

sys.exit()
"""

labels = []
with open("./datasets/Task1_Augmented_Def_new/labels.txt", "r") as f:
	for line in f.readlines():
		label_tokens = line.lower().strip().split("_")
		label_tokens = [token for token in label_tokens if token]
		query = " ".join(label_tokens)
		labels.append(query)

queries = []
if params['use_label_as_query']:
	queries.extend(labels)

if params['use_title_as_query']:
	queries.extend(train_titles)
	queries.extend(val_titles)

query_words = []
for query in queries:
	query_words.extend(query.strip().split())
query_words = np.unique(query_words)
query_words_list = query_words.tolist()

## Building vocab
total_words = []

for doc in train_docs:
	for sent in doc:
		total_words.extend(sent.strip().split())

for doc in val_docs:
	for sent in doc:
		total_words.extend(sent.strip().split())

if params['use_label_as_query'] or params['use_title_as_query']:
	total_words.extend(query_words_list)

total_vocab = list(set(total_words))
print("Vocab size: ", len(total_vocab))

with open(log_dir + "/vocab.txt", "w") as f:
	f.write("\n".join(total_vocab))
#sys.exit()

#with codecs.open("./datasets/Task1_Augmented_Def_new/vocab_docnade.vocab", "r", encoding='utf-8', errors='ignore') as f:
#	docnade_vocab_large = [line.strip() for line in f.readlines()]

prior_embedding_matrices = []

if params['use_BOW_repesentation_for_training']:
	#BOW_representations = np.eye(len(docnade_vocab_large), dtype=np.float32)
	BOW_representations = np.eye(len(total_vocab), dtype=np.float32)
	prior_embedding_matrices.append(BOW_representations)

docnade_embedding_matrix = np.concatenate(prior_embedding_matrices, axis=1)

training_vecs = []
training_labels = []
training_ids = []
for i, doc in enumerate(train_docs):
	for sent in doc:
		#tokens = [docnade_vocab_large.index(word) for word in sent.strip().split()]
		tokens = [total_vocab.index(word) for word in sent.strip().split()]
		Embs = docnade_embedding_matrix[np.array(tokens), :]
		EmbSum = np.sum(Embs, axis=0)
		training_vecs.append(EmbSum)
		temp_label = 0.0
		for rc_sent in train_relevant_context[i]:
			if rc_sent in sent:
				temp_label = 1.0
		training_labels.append(temp_label)
		training_ids.append(train_ids[i])

validation_vecs = []
validation_labels = []
validation_ids = []
for i, doc in enumerate(val_docs):
	for sent in doc:
		#tokens = [docnade_vocab_large.index(word) for word in sent.strip().split()]
		tokens = [total_vocab.index(word) for word in sent.strip().split()]
		Embs = docnade_embedding_matrix[np.array(tokens), :]
		EmbSum = np.sum(Embs, axis=0)
		validation_vecs.append(EmbSum)
		temp_label = 0.0
		for rc_sent in val_relevant_context[i]:
			if rc_sent in sent:
				temp_label = 1.0
		validation_labels.append(temp_label)
		validation_ids.append(val_ids[i])

if params['use_label_as_query'] or params['use_title_as_query']:
	training_query_vecs = []
	for i, doc in enumerate(train_docs):
		for sent in doc:
			query = []
			if params['use_label_as_query']:
				query.append(" ".join(train_labels[i].lower().strip().split("_")))
			if params['use_title_as_query']:
				query.append(train_titles[i])
			query = " ".join(query)
			#tokens = [docnade_vocab_large.index(word) for word in query.strip().split()]
			tokens = [total_vocab.index(word) for word in query.strip().split()]
			Embs = docnade_embedding_matrix[np.array(tokens), :]
			EmbSum = np.sum(Embs, axis=0)
			training_query_vecs.append(EmbSum)

	validation_query_vecs = []
	for i, doc in enumerate(val_docs):
		for sent in doc:
			query = []
			if params['use_label_as_query']:
				query.append(" ".join(val_labels[i].lower().strip().split("_")))
			if params['use_title_as_query']:
				query.append(val_titles[i])
			query = " ".join(query)
			#tokens = [docnade_vocab_large.index(word) for word in query.strip().split()]
			tokens = [total_vocab.index(word) for word in query.strip().split()]
			Embs = docnade_embedding_matrix[np.array(tokens), :]
			EmbSum = np.sum(Embs, axis=0)
			validation_query_vecs.append(EmbSum)
	
	training_vecs = np.concatenate([np.array(training_vecs), np.array(training_query_vecs)], axis=1)
	validation_vecs = np.concatenate([np.array(validation_vecs), np.array(validation_query_vecs)], axis=1)
	#training_vecs = np.subtract(np.array(training_vecs), np.array(training_query_vecs))
	#validation_vecs = np.subtract(np.array(validation_vecs), np.array(validation_query_vecs))
else:
	training_vecs = np.array(training_vecs)
	validation_vecs = np.array(validation_vecs)

if params['use_bm25_extra']:
	training_bm25_extra_vecs = []
	for i, doc in enumerate(train_docs):
		query = []
		query.append(" ".join(train_labels[i].lower().strip().split("_")))
		query.append(train_titles[i])
		query = " ".join(query)

		bm25 = BM25.BM25(doc, delimiter=' ')
		relevance_score_bm25 = bm25.BM25Score(query.split())
		relevance_score_bm25 = np.array(relevance_score_bm25, dtype=np.float32)

		extra_features = []
		for sent in doc:
			sent = sent.split()
			sent_ids = get_bm25_ids(bm25, sent)
			query_ids = get_bm25_ids(bm25, query.split())
			feats = bm25.query_doc_overlap(query_ids, sent_ids)
			extra_features.append(feats)
		
		relevance_score_bm25 = np.concatenate([np.expand_dims(relevance_score_bm25, axis=1), np.array(extra_features, dtype=np.float32)], axis=1)
		training_bm25_extra_vecs.append(relevance_score_bm25)
	training_bm25_extra_vecs = np.vstack(training_bm25_extra_vecs)

	validation_bm25_extra_vecs = []
	for i, doc in enumerate(val_docs):
		query = []
		query.append(" ".join(val_labels[i].lower().strip().split("_")))
		query.append(val_titles[i])
		query = " ".join(query)

		bm25 = BM25.BM25(doc, delimiter=' ')
		relevance_score_bm25 = bm25.BM25Score(query.split())
		relevance_score_bm25 = np.array(relevance_score_bm25, dtype=np.float32)

		extra_features = []
		for sent in doc:
			sent = sent.split()
			sent_ids = get_bm25_ids(bm25, sent)
			query_ids = get_bm25_ids(bm25, query.split())
			feats = bm25.query_doc_overlap(query_ids, sent_ids)
			extra_features.append(feats)
		
		relevance_score_bm25 = np.concatenate([np.expand_dims(relevance_score_bm25, axis=1), np.array(extra_features, dtype=np.float32)], axis=1)
		validation_bm25_extra_vecs.append(relevance_score_bm25)
	validation_bm25_extra_vecs = np.vstack(validation_bm25_extra_vecs)

	training_vecs = np.concatenate([training_vecs, training_bm25_extra_vecs], axis=1)
	validation_vecs = np.concatenate([validation_vecs, validation_bm25_extra_vecs], axis=1)

if params['use_pos_vector']:
	training_position_vecs = []
	for doc in train_docs:
		for i, sent in enumerate(doc):
			position = float(i) / float(len(doc))
			if (position >= 0.0) and (position <= 3.3):
				training_position_vecs.append(np.array([1.0, 0.0, 0.0], dtype=np.float32))
			elif (position > 3.3) and (position <= 6.6):
				training_position_vecs.append(np.array([0.0, 1.0, 0.0], dtype=np.float32))
			elif (position > 6.6) and (position <= 1.0):
				training_position_vecs.append(np.array([0.0, 0.0, 1.0], dtype=np.float32))
	training_position_vecs = np.array(training_position_vecs, dtype=np.float32)

	validation_position_vecs = []
	for doc in val_docs:
		for i, sent in enumerate(doc):
			position = float(i) / float(len(doc))
			if (position >= 0.0) and (position <= 3.3):
				validation_position_vecs.append(np.array([1.0, 0.0, 0.0], dtype=np.float32))
			elif (position > 3.3) and (position <= 6.6):
				validation_position_vecs.append(np.array([0.0, 1.0, 0.0], dtype=np.float32))
			elif (position > 6.6) and (position <= 1.0):
				validation_position_vecs.append(np.array([0.0, 0.0, 1.0], dtype=np.float32))
	validation_position_vecs = np.array(validation_position_vecs, dtype=np.float32)

	#import pdb; pdb.set_trace()

	training_vecs = np.concatenate([training_vecs, training_position_vecs], axis=1)
	validation_vecs = np.concatenate([validation_vecs, validation_position_vecs], axis=1)

test_vecs = validation_vecs
test_labels = validation_labels

reload_evaluation_f1(params, training_vecs, validation_vecs, training_labels, validation_labels, training_ids, validation_ids, val_labels, val_ids)

print("Done.")