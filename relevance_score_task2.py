import os
import argparse
import json
import numpy as np
import tensorflow as tf
import model.data as data
import model.model_DocNADE as m
import model.evaluate as eval
import datetime
import json
import sys
import pickle
import codecs
import csv

from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

from gensim.models.keyedvectors import KeyedVectors
import sklearn.metrics.pairwise as pw
from sklearn.metrics import accuracy_score
import BM25_task2 as BM25
from sklearn.utils.extmath import softmax


params = {}

# select only one of the below two options
evaluate_train_dev = False
evaluate_test = True

params['dataset'] = "./datasets/Task2_without_acronym"
params['dataset_original'] = "./RDoC_raw_data/RDoCTask/RDoCTask2TrainData/Combined_batch/"

params['dataset_test'] = "./datasets/Task2_test_data_combined_batch_without_acronym"
params['dataset_test_original'] = "./RDoC_raw_data/RDoCTask/RDoCTask2TestData/Combined_batch/"

params['model'] = "./model/Task2_without_acronym_unsupervised_classwise"
params['use_bio_prior'] = False
params['use_fasttext_prior'] = True
params['use_BOW_repesentation'] = False
params['use_DocNADE_W'] = False
params['split_abstract_title'] = False
params['attention_EmbSum_type'] = "sum"
params['use_label_as_query'] = True
params['use_title_as_query'] = True
params['label_title_weights'] = [0.5, 0.5]
params['label_title_max'] = False
params['use_one_hot'] = False

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

if params['label_title_max']:
	params['model'] += "_max"
elif params['use_label_as_query'] and params['use_title_as_query']:
	params['model'] += str(params['label_title_weights'])

if params['use_one_hot']:
	params['model'] += "_one_hot"

params['model'] += "_" + params['attention_EmbSum_type']

now = datetime.datetime.now()

params['model'] += "_" + str(now.day) + "_" + str(now.month) + "_" + str(now.year)

log_dir = log_dir = os.path.join(params['model'], 'logs')

if not os.path.exists(log_dir):
	os.makedirs(log_dir)

total_docs = []
total_relevant_context = []
total_labels = []
total_titles = []
total_ids = []

if evaluate_train_dev:
	with open(params['dataset'] + "/train_docs.txt", "r") as f:
		for line in f.readlines():
			id, label, title, doc, rel_con = line.strip().split("<<>>")
			total_docs.append(doc.split("\t"))
			total_relevant_context.append(rel_con.split("\t"))
			total_labels.append(" ".join(label.lower().strip().split("_")))
			total_titles.append(title)
			total_ids.append(id)

	with open(params['dataset'] + "/val_docs.txt", "r") as f:
		for line in f.readlines():
			id, label, title, doc, rel_con = line.strip().split("<<>>")
			total_docs.append(doc.split("\t"))
			total_relevant_context.append(rel_con.split("\t"))
			total_labels.append(" ".join(label.lower().strip().split("_")))
			total_titles.append(title)
			total_ids.append(id)

if evaluate_test:
	with open(params['dataset_test'] + "/test_docs.txt", "r") as f:
		for line in f.readlines():
			id, label, title, doc, rel_con = line.strip().split("<<>>")
			total_docs.append(doc.split("\t"))
			total_relevant_context.append(rel_con.split("\t"))
			total_labels.append(" ".join(label.lower().strip().split("_")))
			total_titles.append(title)
			total_ids.append(id)


total_original_docs = []

if evaluate_train_dev:
	with open(params['dataset_original'] + "/train_docs.txt", "r") as f:
		for line in f.readlines():
			id, label, title, doc, rel_con = line.strip().split("<<>>")
			total_original_docs.append(doc.split("\t"))

	with open(params['dataset_original'] + "/val_docs.txt", "r") as f:
		for line in f.readlines():
			id, label, title, doc, rel_con = line.strip().split("<<>>")
			total_original_docs.append(doc.split("\t"))

if evaluate_test:
	with open(params['dataset_test_original'] + "/test_docs.txt", "r") as f:
		for line in f.readlines():
			id, label, title, doc, rel_con = line.strip().split("<<>>")
			total_original_docs.append(doc.split("\t"))

labels = []
with open("./datasets/Task1_without_acronym/labels.txt", "r") as f:
	for line in f.readlines():
		label_tokens = line.lower().strip().split("_")
		label_tokens = [token for token in label_tokens if token]
		query = " ".join(label_tokens)
		labels.append(query)

queries = []
if params['use_label_as_query']:
	queries.extend(labels)
	#queries.extend(["cortisol corticosterone dopamine endogenous cannabinoids glutamate neuropeptide neurosteroids orexin oxytocin serotonin vasopressin glia neurons pyramidal nervous nucleus hippocampus dorsal hypothalamus cortex startle heart rate respiration analgesia facial freezing inhibition"])
	#queries.extend(["startle"])
	#queries.extend(["threatening"])
	queries.extend(["reward heart rate aggression affective states"])

if params['use_title_as_query']:
	queries.extend(total_titles)

query_words = []
for query in queries:
	query_words.extend(query.strip().split())
query_words = np.unique(query_words)
query_words_list = query_words.tolist()

## Building vocab
total_words = []

for doc in total_docs:
	for sent in doc:
		total_words.extend(sent.strip().split())

if params['use_label_as_query'] or params['use_title_as_query']:
	total_words.extend(query_words_list)

total_vocab = list(set(total_words))
print("Vocab size unigrams: ", len(total_vocab))

total_sents = []
for doc in total_docs:
	for sent in doc:
		total_sents.append(sent)

def tokens(text):
	return [w.lower() for w in text.split()]

cv = CountVectorizer(tokenizer=tokens, min_df=1, max_df=1.0, ngram_range=(2,2), max_features=None, encoding='utf-8', decode_error='ignore')
cv.fit(total_sents)
total_vocab_bigrams = cv.get_feature_names()
print("Vocab size bigrams: ", len(total_vocab_bigrams))

def get_one_hot(query, doc):

	query_vecs = np.zeros((1, len(total_vocab)), dtype=np.float32)
	indices = [total_vocab.index(word.strip()) for word in query.strip().split()]
	for index in indices:
		query_vecs[0, index] = 1.0

	doc_sents_vecs = np.zeros((len(doc), len(total_vocab)), dtype=np.float32)
	for i, sent in enumerate(doc):
		indices = [total_vocab.index(word.strip()) for word in sent.strip().split()]
		for index in indices:
			doc_sents_vecs[i, index] = 1.0
	
	return query_vecs, doc_sents_vecs

def get_bigram(query, doc):

	#import pdb; pdb.set_trace()

	query_vecs = np.array(cv.transform([query]).todense(), dtype=np.float32)
	#query_vecs = np.tile(query_vecs, [len(doc), 1])

	doc_sents_vecs = []
	for i, sent in enumerate(doc):
		doc_sents_vecs.append(np.squeeze(np.array(cv.transform([sent]).todense()), axis=0))
	doc_sents_vecs = np.array(doc_sents_vecs, dtype=np.float32)
	
	return query_vecs, doc_sents_vecs


prior_embedding_matrices = []
query_embedding_matrices = []

with codecs.open("./pretrained_embeddings/biggest_vocab.vocab", "r", encoding='utf-8', errors='ignore') as f:
	docnade_vocab_large = [line.strip() for line in f.readlines()]

if params['use_bio_prior']:
	bio_embeddings_large = np.load('./pretrained_embeddings/bionlp_embeddings_biggest_vocab.npy')
	prior_embedding_matrices.append(bio_embeddings)

	query_bio_embeddings = np.zeros((len(query_words), bio_embeddings_large.shape[1]), dtype=np.float32)
	for i, word in enumerate(query_words):
		query_bio_embeddings[i, :] = bio_embeddings_large[int(docnade_vocab_large.index(word.strip())), :]
	query_embedding_matrices.append(query_bio_embeddings)

if params['use_fasttext_prior']:
	fasttext_embeddings_large = np.load('./pretrained_embeddings/fasttext_embeddings_biggest_vocab.npy')
	prior_embedding_matrices.append(fasttext_embeddings_large)

	query_fasttext_embeddings = np.zeros((len(query_words), fasttext_embeddings_large.shape[1]), dtype=np.float32)
	for i, word in enumerate(query_words):
		query_fasttext_embeddings[i, :] = fasttext_embeddings_large[int(docnade_vocab_large.index(word.strip())), :]
	query_embedding_matrices.append(query_fasttext_embeddings)

if params['use_BOW_repesentation']:
	BOW_representations = np.eye(len(docnade_vocab_large), dtype=np.float32)
	BOW_representations_queries = BOW_representations[np.array([int(docnade_vocab_large.index(word)) for word in query_words]), :]
	prior_embedding_matrices.append(BOW_representations)
	query_embedding_matrices.append(BOW_representations_queries)

if params['use_DocNADE_W']:
	DocNADE_W = session_acc.run("embedding:0")
	prior_embedding_matrices.append(DocNADE_W)

	query_W_embeddings = np.zeros((len(query_words), DocNADE_W.shape[1]), dtype=np.float32)
	for i, word in enumerate(query_words):
		if word in docnade_vocab_large:
			query_W_embeddings[i, :] = DocNADE_W[int(docnade_vocab_large.index(word.strip())), :]
	query_embedding_matrices.append(query_W_embeddings)

docnade_embedding_matrix = np.concatenate(prior_embedding_matrices, axis=1)
query_embedding_matrix = np.concatenate(query_embedding_matrices, axis=1)

bm25_AP_list = []
bm25_extra_AP_list = []
EmbSum_AP_list = []
bm25_Extra_EmbSum_AP_list = []
attention_based_EmbSum_AP_list = []
bm25_Extra_attention_based_EmbSum_AP_list = []
bm25_Extra_with_embeddings_AP_list = []
bm25_Extra_with_embeddings_EmbSum_AP_list = []
bm25_Extra_with_embeddings_attention_based_EmbSum_AP_list = []

bm25_acc_list = []
bm25_extra_acc_list = []
EmbSum_acc_list = []
bm25_Extra_EmbSum_acc_list = []
attention_based_EmbSum_acc_list = []
bm25_Extra_attention_based_EmbSum_acc_list = []
bm25_Extra_with_embeddings_acc_list = []
bm25_Extra_with_embeddings_EmbSum_acc_list = []
bm25_Extra_with_embeddings_attention_based_EmbSum_acc_list = []

doc_counter = -1
#import pdb; pdb.set_trace()
val_scores_dict = {}
val_ids_dict = {}
val_sents_dict = {}
for id, label, doc, title, rel_con in zip(total_ids, total_labels, total_docs, total_titles, total_relevant_context):
	#if label == "frustrative nonreward":
	#	label = "frustrative reward physical relational aggression"
	#	label = "reward physical relational aggression"

	#if label == "acute threat fear":
		#label += " cortisol corticosterone dopamine endogenous cannabinoids glutamate neuropeptide neurosteroids orexin oxytocin serotonin vasopressin glia neurons pyramidal nervous nucleus hippocampus dorsal hypothalamus cortex startle heart rate respiration analgesia facial freezing inhibition"
		#label += " threatening"
		#label = "threat fear"
	#	pass

	#if label == "loss":
	#	label += " worry"

	#import pdb; pdb.set_trace()

	doc_counter += 1

	query = []
	
	if params['use_label_as_query']:
		query.append(label)
	
	if params['use_title_as_query']:
		query.append(title)

	query = " ".join(query)

	temp_rel_con = []
	#for rc_sent in set(rel_con):
	for rc_sent in rel_con:
		for sent in doc:
			if rc_sent in sent:
				temp_rel_con.append(sent)
				break

	#if len(temp_rel_con) != len(set(rel_con)):
	#	print("Mismatch RC sentences.")
	#	import pdb; pdb.set_trace()
	
	doc_true_relevance_labels = []
	for sent in doc:
		if sent in temp_rel_con:
			doc_true_relevance_labels.append(float(1))
		else:
			doc_true_relevance_labels.append(float(0))

	#import pdb; pdb.set_trace()

	def get_bm25_ids(tokens):
		ids = []
		for token in tokens:
			try:
				ids.append(bm25.dictionary.token2id[token])
			except KeyError:
				pass
		return ids

	## Using BM25 score for relevance ranking and mAP calculation

	bm25 = BM25.BM25(doc, delimiter=' ')
	"""
	relevance_score_bm25 = bm25.BM25Score(query.split())
	relevance_score_bm25 = np.array(relevance_score_bm25, dtype=np.float32)
	"""
	#temp_score_bm25 = []

	label_score_bm25 = []
	if params['use_label_as_query']:
		if label == "frustrative nonreward":
			label = "physical relational aggression"
		#if label == "sustained threat":
		#	label += " amygdala attention bias"
		#if label == "arousal":
		#	label += " affective states heart rate"
		relevance_score_bm25_label = bm25.BM25Score(label.split())
		if label == "physical relational aggression":
			label = "frustrative nonreward"
		#if label == "sustained threat amygdala attention bias":
		#	label = "sustained threat"
		#if label == "arousal affective states heart rate":
		#	label = "arousal"
		#if label == "acute threat fear":
		#relevance_score_bm25_label = bm25.BM25Score((label + " cortisol corticosterone dopamine endogenous cannabinoids glutamate neuropeptide neurosteroids orexin oxytocin serotonin vasopressin glia neurons pyramidal nervous nucleus hippocampus dorsal hypothalamus cortex startle heart rate respiration analgesia facial freezing inhibition").split())
		relevance_score_bm25_label = np.array(relevance_score_bm25_label, dtype=np.float32)
		#temp_score_bm25.append(relevance_score_bm25_label)
		label_score_bm25.append(relevance_score_bm25_label)

	title_score_bm25 = []
	if params['use_title_as_query']:
		relevance_score_bm25_title = bm25.BM25Score(title.split())
		relevance_score_bm25_title = np.array(relevance_score_bm25_title, dtype=np.float32)
		#temp_score_bm25.append(relevance_score_bm25_title)
		title_score_bm25.append(relevance_score_bm25_title)
	#import pdb; pdb.set_trace()
	relevance_score_bm25 = (np.array(label_score_bm25) * np.array(label_score_bm25)) + (np.array(title_score_bm25) * np.array(title_score_bm25))
	relevance_score_bm25 = relevance_score_bm25.reshape(-1)
	
	"""
	if len(temp_score_bm25) == 1:
		relevance_score_bm25 = temp_score_bm25[0]
	else:
		relevance_score_bm25 = (params['label_title_weights'][0] * temp_score_bm25[0]) + (params['label_title_weights'][1] * temp_score_bm25[1])
	"""

	AP, relevance_labels, relevance_scores, sorted_indices = eval.most_relevant(doc_true_relevance_labels, relevance_score_bm25)

	bm25_AP_list.append(AP)
	bm25_acc_list.append(relevance_labels[0])

	# logging information
	with open(os.path.join(log_dir, "reload_info_clusters_with_BM25.txt"), "a") as f:
		f.write("Doc " + str(doc_counter) + ": " + str(query) + ": " + str(id) + "\n\n")
		f.write("Average precision: " + str(AP) + "\n")
		f.write("Predicted_labels:  " + " ".join([str(int(l)) for l in relevance_labels]) + "\n")
		f.write("Predicted_probs:   " + " ".join([str(l) for l in relevance_scores]) + "\n")
		f.write("Sentence indices:  " + " ".join([str(l) for l in sorted_indices]) + "\n")
		f.write("\n=================================================================================\n\n")

	## Using BM25 Extra for relevance ranking and mAP calculation
	"""
	extra_features = []
	for sent in doc:
		sent = sent.split()
		sent_ids = get_bm25_ids(sent)
		query_ids = get_bm25_ids(query.split())
		feats = bm25.query_doc_overlap(query_ids, sent_ids)
		extra_features.append(np.sum(feats))
	"""
	label_extra_features = []
	for sent in doc:
		if label == "frustrative nonreward":
			label = "physical relational aggression"
		#if label == "sustained threat":
		#	label += " amygdala attention bias anxiety"
		#if label == "arousal":
		#	label += " affective states heart rate"
		sent = sent.split()
		sent_ids = get_bm25_ids(sent)
		query_ids = get_bm25_ids(label.split())
		if label == "physical relational aggression":
			label = "frustrative nonreward"
		#if label == "sustained threat amygdala attention bias":
		#	label = "sustained threat"
		#if label == "arousal affective states heart rate":
		#	label = "arousal"
		#if label == "acute threat fear":
		#query_ids = get_bm25_ids((label + " cortisol corticosterone dopamine endogenous cannabinoids glutamate neuropeptide neurosteroids orexin oxytocin serotonin vasopressin glia neurons pyramidal nervous nucleus hippocampus dorsal hypothalamus cortex startle heart rate respiration analgesia facial freezing inhibition").split())
		feats = bm25.query_doc_overlap(query_ids, sent_ids)
		label_extra_features.append(np.sum(feats))

	title_extra_features = []
	for sent in doc:
		sent = sent.split()
		sent_ids = get_bm25_ids(sent)
		query_ids = get_bm25_ids(title.split())
		feats = bm25.query_doc_overlap(query_ids, sent_ids)
		title_extra_features.append(np.sum(feats))

	extra_features = np.array(label_extra_features) + np.array(title_extra_features)
	#import pdb; pdb.set_trace()
	label_relevance_score_bm25_extra = np.add(np.array(label_score_bm25).reshape(-1), np.array(label_extra_features, dtype=np.float32))
	title_relevance_score_bm25_extra = np.add(np.array(title_score_bm25).reshape(-1), np.array(title_extra_features, dtype=np.float32))
	#title_relevance_score_bm25_extra = 0.0
	#relevance_score_bm25_extra = (label_relevance_score_bm25_extra * label_relevance_score_bm25_extra) + (title_relevance_score_bm25_extra * title_relevance_score_bm25_extra)
	relevance_score_bm25_extra = label_relevance_score_bm25_extra
	
	#relevance_score_bm25_extra = np.add(relevance_score_bm25, np.array(extra_features, dtype=np.float32))

	AP, relevance_labels, relevance_scores, sorted_indices = eval.most_relevant(doc_true_relevance_labels, relevance_score_bm25_extra)

	bm25_extra_AP_list.append(AP)
	bm25_extra_acc_list.append(relevance_labels[0])
	"""
	if not label in val_scores_dict.keys():
		val_scores_dict[label] = []
		val_ids_dict[label] = []
		val_sents_dict[label] = []
	val_scores_dict[label].append(relevance_labels[0])
	val_ids_dict[label].append(id)
	val_sents_dict[label].append(total_original_docs[doc_counter][sorted_indices[0]])
	if len(total_original_docs[doc_counter]) != len(doc):
		print("Mismatch lengths of original and preprocessed docs")
		import pdb; pdb.set_trace()
	"""
	# logging information
	with open(os.path.join(log_dir, "reload_info_clusters_with_BM25_Extra.txt"), "a") as f:
		f.write("Doc " + str(doc_counter) + ": " + str(query) + ": " + str(id) + "\n\n")
		f.write("Average precision: " + str(AP) + "\n")
		f.write("Predicted_labels:  " + " ".join([str(int(l)) for l in relevance_labels]) + "\n")
		f.write("Predicted_probs:   " + " ".join([str(l) for l in relevance_scores]) + "\n")
		f.write("Sentence indices:  " + " ".join([str(l) for l in sorted_indices]) + "\n")
		f.write("\n=================================================================================\n\n")

	## Using EmbSum for relevance ranking and mAP calculation
	
	temp_similarity_scores = []

	if params['use_label_as_query']:
		tokens = [int(query_words_list.index(word)) for word in label.strip().split()]
		Embs = query_embedding_matrix[np.array(tokens), :]
		EmbSum = np.sum(Embs, axis=0)
		query_vec = EmbSum
	
		doc_sents_vecs = []
		for sent in doc:
			tokens = [docnade_vocab_large.index(word.strip()) for word in sent.strip().split()]
			Embs = docnade_embedding_matrix[np.array(tokens), :]
			EmbSum = np.sum(Embs, axis=0)
			doc_sents_vecs.append(EmbSum)

		## Adding bigrams representation
		label_vecs_bigram, label_doc_vecs_bigram = get_bigram(label, doc)
		doc_sents_vecs = np.concatenate([np.array(doc_sents_vecs), label_doc_vecs_bigram], axis=1)
		query_vec = np.concatenate([np.expand_dims(query_vec, axis=0), label_vecs_bigram], axis=1)
		
		#similarity_scores_label = pw.cosine_similarity(np.array(doc_sents_vecs), np.expand_dims(query_vec, axis=0))
		similarity_scores_label = pw.cosine_similarity(doc_sents_vecs, query_vec)
		similarity_scores_label = np.squeeze(similarity_scores_label, axis=1)
		temp_similarity_scores.append(similarity_scores_label)

	if params['use_title_as_query']:
		tokens = [int(query_words_list.index(word)) for word in title.strip().split()]
		Embs = query_embedding_matrix[np.array(tokens), :]
		EmbSum = np.sum(Embs, axis=0)
		query_vec = EmbSum
	
		doc_sents_vecs = []
		for sent in doc:
			tokens = [docnade_vocab_large.index(word.strip()) for word in sent.strip().split()]
			Embs = docnade_embedding_matrix[np.array(tokens), :]
			EmbSum = np.sum(Embs, axis=0)
			doc_sents_vecs.append(EmbSum)

		## Adding bigrams representation
		title_vecs_bigram, title_doc_vecs_bigram = get_bigram(title, doc)
		doc_sents_vecs = np.concatenate([np.array(doc_sents_vecs), title_doc_vecs_bigram], axis=1)
		query_vec = np.concatenate([np.expand_dims(query_vec, axis=0), title_vecs_bigram], axis=1)
		
		#similarity_scores_title = pw.cosine_similarity(np.array(doc_sents_vecs), np.expand_dims(query_vec, axis=0))
		similarity_scores_title = pw.cosine_similarity(doc_sents_vecs, query_vec)
		similarity_scores_title = np.squeeze(similarity_scores_title, axis=1)
		temp_similarity_scores.append(similarity_scores_title)
	
	"""
	if params['use_label_as_query'] and params['use_title_as_query']:

		tokens = [int(query_words_list.index(word)) for word in label.strip().split()]
		Embs = query_embedding_matrix[np.array(tokens), :]
		EmbSum = np.sum(Embs, axis=0)
		label_vec = EmbSum
	
		label_doc_sents_vecs = []
		for sent in doc:
			tokens = [docnade_vocab_large.index(word.strip()) for word in sent.strip().split()]
			Embs = docnade_embedding_matrix[np.array(tokens), :]
			EmbSum = np.sum(Embs, axis=0)
			label_doc_sents_vecs.append(EmbSum)

		tokens = [int(query_words_list.index(word)) for word in title.strip().split()]
		Embs = query_embedding_matrix[np.array(tokens), :]
		EmbSum = np.sum(Embs, axis=0)
		title_vec = EmbSum
	
		title_doc_sents_vecs = []
		for sent in doc:
			tokens = [docnade_vocab_large.index(word.strip()) for word in sent.strip().split()]
			Embs = docnade_embedding_matrix[np.array(tokens), :]
			EmbSum = np.sum(Embs, axis=0)
			title_doc_sents_vecs.append(EmbSum)
		#import pdb; pdb.set_trace()
		#if len(doc) == 1:
		#	import pdb; pdb.set_trace()

		query_one_hot, doc_one_hot = get_one_hot(label + " " + title, doc)
		query_vec = np.concatenate([np.expand_dims(label_vec, axis=0), np.expand_dims(title_vec, axis=0), query_one_hot], axis=1)
		doc_sents_vecs = np.concatenate([label_doc_sents_vecs, title_doc_sents_vecs, doc_one_hot], axis=1)
		similarity_scores = np.squeeze(pw.cosine_similarity(query_vec, doc_sents_vecs), axis=0)
	"""
	
	if len(temp_similarity_scores) == 1:
		similarity_scores = temp_similarity_scores[0]
	elif len(temp_similarity_scores) > 1:
		if params['label_title_max']:
			similarity_scores = np.maximum(temp_similarity_scores[0], temp_similarity_scores[1])
		else:
			similarity_scores = (params['label_title_weights'][0] * temp_similarity_scores[0]) + (params['label_title_weights'][1] * temp_similarity_scores[1])
	
	AP, relevance_labels, relevance_scores, sorted_indices = eval.most_relevant(doc_true_relevance_labels, similarity_scores)

	EmbSum_AP_list.append(AP)
	EmbSum_acc_list.append(relevance_labels[0])

	# logging information
	with open(os.path.join(log_dir, "reload_info_clusters_with_EmbSum.txt"), "a") as f:
		f.write("Doc " + str(doc_counter) + ": " + str(query) + ": " + str(id) + "\n\n")
		f.write("Average precision: " + str(AP) + "\n")
		f.write("Predicted_labels:  " + " ".join([str(int(l)) for l in relevance_labels]) + "\n")
		f.write("Predicted_probs:   " + " ".join([str(l) for l in relevance_scores]) + "\n")
		f.write("Sentence indices:  " + " ".join([str(l) for l in sorted_indices]) + "\n")
		f.write("\n=================================================================================\n\n")
				
	## Using BM25 Extra + EmbSum for relevance ranking and mAP calculation
	
	similarity_scores = np.array(similarity_scores)
	similarity_scores[(similarity_scores<0.0)] = 0.0
	relevance_score_bm25_extra = np.array(relevance_score_bm25_extra)
	relevance_score_bm25_extra[(relevance_score_bm25_extra<0.0)] = 0.0

	similarity_scores_combined = np.squeeze(np.add(softmax(np.expand_dims(similarity_scores, axis=0)), softmax(np.expand_dims(relevance_score_bm25_extra, axis=0))), axis=0)

	AP, relevance_labels, relevance_scores, sorted_indices = eval.most_relevant(doc_true_relevance_labels, similarity_scores_combined)

	bm25_Extra_EmbSum_AP_list.append(AP)
	bm25_Extra_EmbSum_acc_list.append(relevance_labels[0])

	# logging information
	with open(os.path.join(log_dir, "reload_info_clusters_with_BM25_Extra_and_EmbSum.txt"), "a") as f:
		f.write("Doc " + str(doc_counter) + ": " + str(query) + ": " + str(id) + "\n\n")
		f.write("Average precision: " + str(AP) + "\n")
		f.write("Predicted_labels:  " + " ".join([str(int(l)) for l in relevance_labels]) + "\n")
		f.write("Predicted_probs:   " + " ".join([str(l) for l in relevance_scores]) + "\n")
		f.write("Sentence indices:  " + " ".join([str(l) for l in sorted_indices]) + "\n")
		f.write("\n=================================================================================\n\n")

	## Using Attention based EmbSum for relevance ranking and mAP calculation

	temp_similarity_scores_Attention_Based_EmbSum = []

	similarity_scores_Attention_Based_EmbSum_label = []
	#label_query_vecs = []
	#label_doc_vecs = []
	if params['use_label_as_query']:
		for j, sent in enumerate(doc):
			tokens = [docnade_vocab_large.index(word.strip()) for word in sent.strip().split()]
			Embs = docnade_embedding_matrix[np.array(tokens), :]
			if label == "frustrative nonreward":
				label = "physical relational aggression"
			if label == "arousal":
				label += " affective states heart rate"
			if label == "sustained threat":
				label += " amygdala attention bias"
			query_tokens = label.strip().split()
			if label == "physical relational aggression":
				label = "frustrative nonreward"
			if label == "arousal affective states heart rate":
				label = "arousal"
			if label == "sustained threat amygdala attention bias":
				label = "sustained threat"
			#if label == "sustained threat amygdala attention bias anxiety":
			#	label = "sustained threat"
			"""
			extra_query_tokens = []
			for qword in query_tokens:
				sim = pw.cosine_similarity(np.expand_dims(query_embedding_matrix[query_words_list.index(qword), :], axis=0), Embs)
				indices = [index for index, val in enumerate(np.squeeze(sim)) if val >= 0.8]
				extra_query_tokens.extend([qword for index in indices])
			query_tokens.extend(extra_query_tokens)
			"""
			#query_tokens.extend(label.strip().split())
			#query_tokens.extend(title.strip().split())
			
			EmbSum_attns = []
			query_vecs_attns = []
			for qword in query_tokens:
				query_vector = query_embedding_matrix[query_words_list.index(qword), :]
				query_vector = np.expand_dims(query_vector, axis=0)
				query_attentions = pw.cosine_similarity(query_vector, Embs)
				#import pdb; pdb.set_trace()
				query_attentions[(query_attentions < 0.8)] = 0.0
				#query_attentions = softmax(query_attentions)
				EmbSum_attentions = np.dot(query_attentions, Embs)
				EmbSum_attns.append(EmbSum_attentions)
				#if qword == "acute":
				#	query_vecs_attns.append(query_vector * 0.5)
				#else:
				#	query_vecs_attns.append(query_vector)
				query_vecs_attns.append(query_vector)
				#if label == "frustrative reward physical relational aggression":
				#	import pdb; pdb.set_trace()
				#if len(np.nonzero(query_attentions)[1]) == 0:
				#	query_vecs_attns.append(query_vector * np.squeeze(np.sum(query_attentions)))
				#else:
				#	query_vecs_attns.append(query_vector * (np.squeeze(np.sum(query_attentions)) / len(np.nonzero(query_attentions)[1])))
				#if qword in tokens:
				#	query_vecs_attns.append(query_vector)
				#else:
				#	query_vecs_attns.append(query_vector * 0.0)
			if params['attention_EmbSum_type'] == "sum":
				EmbSum = np.sum(EmbSum_attns, axis=0)
				query_EmbSum_vector = np.sum(query_vecs_attns, axis=0)
				#label_query_vecs.append(np.squeeze(query_EmbSum_vector))
				#label_doc_vecs.append(np.squeeze(EmbSum))
				## Adding bigrams representation
				#label_vecs_bigram, label_doc_vecs_bigram = get_bigram(label, [sent])
				#EmbSum = np.concatenate([EmbSum, label_doc_vecs_bigram], axis=1)
				#query_EmbSum_vector = np.concatenate([query_EmbSum_vector, label_vecs_bigram], axis=1)
				similarity_score = pw.cosine_similarity(query_EmbSum_vector, EmbSum)
				similarity_scores_Attention_Based_EmbSum_label.append(similarity_score[0][0])
			elif params['attention_EmbSum_type'] == "concat":
				EmbSum = np.concatenate(EmbSum_attns, axis=1)
				query_EmbSum_vector = np.concatenate(query_vecs_attns, axis=1)
				similarity_score = pw.cosine_similarity(query_EmbSum_vector, EmbSum)
				similarity_scores_Attention_Based_EmbSum_label.append(similarity_score[0][0])
			elif params['attention_EmbSum_type'] == "max":
				max_similarity = -1.1
				for q_vec, d_vec in zip(query_vecs_attns, EmbSum_attns):
					similarity_score = pw.cosine_similarity(q_vec, d_vec)
					if similarity_score[0][0] > max_similarity:
						max_similarity = similarity_score[0][0]
				similarity_scores_Attention_Based_EmbSum_label.append(max_similarity)
		temp_similarity_scores_Attention_Based_EmbSum.append(similarity_scores_Attention_Based_EmbSum_label)
	
	similarity_scores_Attention_Based_EmbSum_title = []
	#title_query_vecs = []
	#title_doc_vecs = []
	if params['use_title_as_query']:
		for j, sent in enumerate(doc):
			tokens = [docnade_vocab_large.index(word.strip()) for word in sent.strip().split()]
			Embs = docnade_embedding_matrix[np.array(tokens), :]
			query_tokens = title.strip().split()
			EmbSum_attns = []
			query_vecs_attns = []
			for qword in query_tokens:
				query_vector = query_embedding_matrix[query_words_list.index(qword), :]
				query_vector = np.expand_dims(query_vector, axis=0)
				query_attentions = pw.cosine_similarity(query_vector, Embs)
				#import pdb; pdb.set_trace()
				query_attentions[(query_attentions < 0.8)] = 0.0
				#query_attentions = softmax(query_attentions)
				EmbSum_attentions = np.dot(query_attentions, Embs)
				EmbSum_attns.append(EmbSum_attentions)
				query_vecs_attns.append(query_vector)
				#if len(np.nonzero(query_attentions)[1]) == 0:
				#	query_vecs_attns.append(query_vector * np.squeeze(np.sum(query_attentions)))
				#else:
				#	query_vecs_attns.append(query_vector * (np.squeeze(np.sum(query_attentions)) / len(np.nonzero(query_attentions)[1])))
				#if qword in tokens:
				#	query_vecs_attns.append(query_vector)
				#else:
				#	query_vecs_attns.append(query_vector * 0.0)
			if params['attention_EmbSum_type'] == "sum":
				EmbSum = np.sum(EmbSum_attns, axis=0)
				query_EmbSum_vector = np.sum(query_vecs_attns, axis=0)
				#title_query_vecs.append(np.squeeze(query_EmbSum_vector))
				#title_doc_vecs.append(np.squeeze(EmbSum))
				## Adding bigrams representation
				#title_vecs_bigram, title_doc_vecs_bigram = get_bigram(title, [sent])
				#EmbSum = np.concatenate([EmbSum, title_doc_vecs_bigram], axis=1)
				#query_EmbSum_vector = np.concatenate([query_EmbSum_vector, title_vecs_bigram], axis=1)
				similarity_score = pw.cosine_similarity(query_EmbSum_vector, EmbSum)
				similarity_scores_Attention_Based_EmbSum_title.append(similarity_score[0][0])
			elif params['attention_EmbSum_type'] == "concat":
				EmbSum = np.concatenate(EmbSum_attns, axis=1)
				query_EmbSum_vector = np.concatenate(query_vecs_attns, axis=1)
				similarity_score = pw.cosine_similarity(query_EmbSum_vector, EmbSum)
				similarity_scores_Attention_Based_EmbSum_title.append(similarity_score[0][0])
			elif params['attention_EmbSum_type'] == "max":
				max_similarity = -1.1
				for q_vec, d_vec in zip(query_vecs_attns, EmbSum_attns):
					similarity_score = pw.cosine_similarity(q_vec, d_vec)
					if similarity_score[0][0] > max_similarity:
						max_similarity = similarity_score[0][0]
				similarity_scores_Attention_Based_EmbSum_title.append(max_similarity)
		temp_similarity_scores_Attention_Based_EmbSum.append(similarity_scores_Attention_Based_EmbSum_title)
		#import pdb; pdb.set_trace()
		#query_vecs = np.concatenate([np.expand_dims(label_query_vecs[0], axis=0), np.expand_dims(title_query_vecs[0], axis=0), query_one_hot], axis=1)
		#query_vecs = np.concatenate([np.expand_dims(label_query_vecs[0], axis=0), np.expand_dims(title_query_vecs[0], axis=0)], axis=1)
		#doc_sents_vecs = np.concatenate([label_doc_vecs, title_doc_vecs, doc_one_hot], axis=1)
		#doc_sents_vecs = np.concatenate([label_doc_vecs, title_doc_vecs], axis=1)
		#similarity_scores_Attention_Based_EmbSum = np.squeeze(pw.cosine_similarity(query_vecs, doc_sents_vecs), axis=0)
	
	if len(temp_similarity_scores_Attention_Based_EmbSum) == 1:
		similarity_scores_Attention_Based_EmbSum = np.array(temp_similarity_scores_Attention_Based_EmbSum[0], dtype=np.float32)
	elif len(temp_similarity_scores_Attention_Based_EmbSum) > 1:
		if params['label_title_max']:
			similarity_scores_Attention_Based_EmbSum = np.maximum(temp_similarity_scores_Attention_Based_EmbSum[0], temp_similarity_scores_Attention_Based_EmbSum[1])
		else:
			#similarity_scores_Attention_Based_EmbSum = (params['label_title_weights'][0] * np.array(temp_similarity_scores_Attention_Based_EmbSum[0])) + (params['label_title_weights'][1] * np.array(temp_similarity_scores_Attention_Based_EmbSum[1]))
			#similarity_scores_Attention_Based_EmbSum = (np.array(temp_similarity_scores_Attention_Based_EmbSum[0]) * np.array(temp_similarity_scores_Attention_Based_EmbSum[0])) + (np.array(temp_similarity_scores_Attention_Based_EmbSum[1]) * np.array(temp_similarity_scores_Attention_Based_EmbSum[1]))
			#similarity_scores_Attention_Based_EmbSum = np.array(temp_similarity_scores_Attention_Based_EmbSum[0])
			#similarity_scores_Attention_Based_EmbSum = (np.array(temp_similarity_scores_Attention_Based_EmbSum[0]) * np.array(temp_similarity_scores_Attention_Based_EmbSum[0])) + (relevance_score_bm25_extra * relevance_score_bm25_extra)
			#similarity_scores_Attention_Based_EmbSum = (0.3 * np.array(temp_similarity_scores_Attention_Based_EmbSum[0])) + (0.7 * relevance_score_bm25_extra)
			#import pdb; pdb.set_trace()
			#attentions = softmax(np.stack([np.array(temp_similarity_scores_Attention_Based_EmbSum[0]), relevance_score_bm25_extra], axis=1))
			#attentions = softmax(np.stack([np.array(temp_similarity_scores_Attention_Based_EmbSum[0]), np.array(temp_similarity_scores_Attention_Based_EmbSum[1]), relevance_score_bm25_extra], axis=1))
			attentions = softmax(np.stack([np.array(temp_similarity_scores_Attention_Based_EmbSum[0]), np.array(temp_similarity_scores_Attention_Based_EmbSum[1])], axis=1))
			#similarity_scores_Attention_Based_EmbSum = (attentions[:,0] * np.array(temp_similarity_scores_Attention_Based_EmbSum[0])) + (attentions[:,1] * relevance_score_bm25_extra)
			#similarity_scores_Attention_Based_EmbSum = (attentions[:,0] * np.array(temp_similarity_scores_Attention_Based_EmbSum[0])) + (attentions[:,1] * np.array(temp_similarity_scores_Attention_Based_EmbSum[1])) + (attentions[:,2] * relevance_score_bm25_extra)
			temp = np.array((attentions[:,0] < attentions[:,1]), dtype=np.float32)
			temp_2 = temp * np.abs(attentions[:,0] - attentions[:,1])
			temp_3 = temp_2 * attentions[:,1]
			#similarity_scores_Attention_Based_EmbSum = (attentions[:,0] * np.array(temp_similarity_scores_Attention_Based_EmbSum[0])) + (attentions[:,1] * np.array(temp_similarity_scores_Attention_Based_EmbSum[1]))
			similarity_scores_Attention_Based_EmbSum = attentions[:,0] * np.array(temp_similarity_scores_Attention_Based_EmbSum[0]) + temp_3 * np.array(temp_similarity_scores_Attention_Based_EmbSum[1])
			#similarity_scores_Attention_Based_EmbSum = np.array(temp_similarity_scores_Attention_Based_EmbSum[0])
			#similarity_scores_Attention_Based_EmbSum = attentions[:,0] * np.array(temp_similarity_scores_Attention_Based_EmbSum[0]) + temp_2 * np.array(temp_similarity_scores_Attention_Based_EmbSum[1])
			#import pdb; pdb.set_trace()
			#similarity_scores_Attention_Based_EmbSum = np.squeeze((0.5 * softmax(np.array(temp_similarity_scores_Attention_Based_EmbSum[0]).reshape((1,-1)))) + (0.5 * softmax(relevance_score_bm25_extra.reshape((1,-1)))))
			#similarity_scores_Attention_Based_EmbSum = (np.array(temp_similarity_scores_Attention_Based_EmbSum[0]) * np.array(temp_similarity_scores_Attention_Based_EmbSum[0])) + ((1.0 - np.array(temp_similarity_scores_Attention_Based_EmbSum[0])) * np.array(temp_similarity_scores_Attention_Based_EmbSum[1]))
			#if label == "frustrative reward physical relational aggression":
			#if label == "frustrative nonreward":
			#	print(temp_similarity_scores_Attention_Based_EmbSum)
	#import pdb; pdb.set_trace()
	
	AP, relevance_labels, relevance_scores, sorted_indices = eval.most_relevant(doc_true_relevance_labels, similarity_scores_Attention_Based_EmbSum)

	attention_based_EmbSum_AP_list.append(AP)
	attention_based_EmbSum_acc_list.append(relevance_labels[0])

	
	if not label in val_scores_dict.keys():
		val_scores_dict[label] = []
		val_ids_dict[label] = []
		val_sents_dict[label] = []
	val_scores_dict[label].append(relevance_labels[0])
	val_ids_dict[label].append(id)
	val_sents_dict[label].append(total_original_docs[doc_counter][sorted_indices[0]])
	if len(total_original_docs[doc_counter]) != len(doc):
		print("Mismatch lengths of original and preprocessed docs")
		import pdb; pdb.set_trace()
	


	# logging information
	with open(os.path.join(log_dir, "reload_info_clusters_with_attention_based_EmbSum.txt"), "a") as f:
		f.write("Doc " + str(doc_counter) + ": " + str(query) + ": " + str(id) + "\n\n")
		f.write("Average precision: " + str(AP) + "\n")
		f.write("Predicted_labels:  " + " ".join([str(int(l)) for l in relevance_labels]) + "\n")
		f.write("Predicted_probs:   " + " ".join([str(l) for l in relevance_scores]) + "\n")
		f.write("Sentence indices:  " + " ".join([str(l) for l in sorted_indices]) + "\n")
		f.write("\n=================================================================================\n\n")

	## Using BM25 Extra + Attention based EmbSum for relevance ranking and mAP calculation

	similarity_scores_Attention_Based_EmbSum = np.array(similarity_scores_Attention_Based_EmbSum)
	similarity_scores_Attention_Based_EmbSum[(similarity_scores_Attention_Based_EmbSum<0.0)] = 0.0
	relevance_score_bm25_extra = np.array(relevance_score_bm25_extra)
	relevance_score_bm25_extra[(relevance_score_bm25_extra<0.0)] = 0.0

	#similarity_scores_Attention_Based_EmbSum_combined = np.squeeze(np.add(softmax(np.expand_dims(similarity_scores_Attention_Based_EmbSum, axis=0)), softmax(np.expand_dims(relevance_score_bm25_extra, axis=0))), axis=0)
	similarity_scores_Attention_Based_EmbSum_combined = np.squeeze(np.add(0.0 * np.expand_dims(similarity_scores_Attention_Based_EmbSum, axis=0), np.expand_dims(relevance_score_bm25_extra, axis=0)), axis=0)

	AP, relevance_labels, relevance_scores, sorted_indices = eval.most_relevant(doc_true_relevance_labels, similarity_scores_Attention_Based_EmbSum_combined)

	bm25_Extra_attention_based_EmbSum_AP_list.append(AP)
	bm25_Extra_attention_based_EmbSum_acc_list.append(relevance_labels[0])
	"""
	if not label in val_scores_dict.keys():
		val_scores_dict[label] = []
		val_ids_dict[label] = []
		val_sents_dict[label] = []
	val_scores_dict[label].append(relevance_labels[0])
	val_ids_dict[label].append(id)
	val_sents_dict[label].append(total_original_docs[doc_counter][sorted_indices[0]])
	if len(total_original_docs[doc_counter]) != len(doc):
		print("Mismatch lengths of original and preprocessed docs")
		import pdb; pdb.set_trace()
	"""
	# logging information
	with open(os.path.join(log_dir, "reload_info_clusters_with_BM25_Extra_and_attention_based_EmbSum.txt"), "a") as f:
		f.write("Doc " + str(doc_counter) + ": " + str(query) + ": " + str(id) + "\n\n")
		f.write("Average precision: " + str(AP) + "\n")
		f.write("Predicted_labels:  " + " ".join([str(int(l)) for l in relevance_labels]) + "\n")
		f.write("Predicted_probs:   " + " ".join([str(l) for l in relevance_scores]) + "\n")
		f.write("Sentence indices:  " + " ".join([str(l) for l in sorted_indices]) + "\n")
		f.write("\n=================================================================================\n\n")

	## Using BM25 Extra with embeddings for relevance ranking and mAP calculation

	relevance_score_bm25_with_emb = bm25.BM25Score(Query=query.split(), 
													embedding_matrix=docnade_embedding_matrix, 
													embedding_vocab=docnade_vocab_large, 
													query_matrix=query_embedding_matrix, 
													query_vocab=query_words_list, 
													sim_threshold=0.50)
	
	relevance_score_bm25_with_emb = np.array(relevance_score_bm25_with_emb, dtype=np.float32)
	relevance_score_bm25_extra_with_emb = np.add(relevance_score_bm25_with_emb, np.array(extra_features, dtype=np.float32))
	
	AP, relevance_labels, relevance_scores, sorted_indices = eval.most_relevant(doc_true_relevance_labels, relevance_score_bm25_extra_with_emb)

	bm25_Extra_with_embeddings_AP_list.append(AP)
	bm25_Extra_with_embeddings_acc_list.append(relevance_labels[0])

	# logging information
	with open(os.path.join(log_dir, "reload_info_clusters_with_BM25_Extra_using_embeddings.txt"), "a") as f:
		f.write("Doc " + str(doc_counter) + ": " + str(query) + ": " + str(id) + "\n\n")
		f.write("Average precision: " + str(AP) + "\n")
		f.write("Predicted_labels:  " + " ".join([str(int(l)) for l in relevance_labels]) + "\n")
		f.write("Predicted_probs:   " + " ".join([str(l) for l in relevance_scores]) + "\n")
		f.write("Sentence indices:  " + " ".join([str(l) for l in sorted_indices]) + "\n")
		f.write("\n=================================================================================\n\n")

	## Using BM25 Extra using embeddings + EmbSum for relevance ranking and mAP calculation

	similarity_scores = np.array(similarity_scores)
	similarity_scores[(similarity_scores<0.0)] = 0.0
	relevance_score_bm25_extra_with_emb = np.array(relevance_score_bm25_extra_with_emb)
	relevance_score_bm25_extra_with_emb[(relevance_score_bm25_extra_with_emb<0.0)] = 0.0

	similarity_scores_combined = np.squeeze(np.add(softmax(np.expand_dims(similarity_scores, axis=0)), softmax(np.expand_dims(relevance_score_bm25_extra_with_emb, axis=0))), axis=0)

	AP, relevance_labels, relevance_scores, sorted_indices = eval.most_relevant(doc_true_relevance_labels, similarity_scores_combined)

	bm25_Extra_with_embeddings_EmbSum_AP_list.append(AP)
	bm25_Extra_with_embeddings_EmbSum_acc_list.append(relevance_labels[0])

	# logging information
	with open(os.path.join(log_dir, "reload_info_clusters_with_BM25_Extra_using_embeddings_and EmbSum.txt"), "a") as f:
		f.write("Doc " + str(doc_counter) + ": " + str(query) + ": " + str(id) + "\n\n")
		f.write("Average precision: " + str(AP) + "\n")
		f.write("Predicted_labels:  " + " ".join([str(int(l)) for l in relevance_labels]) + "\n")
		f.write("Predicted_probs:   " + " ".join([str(l) for l in relevance_scores]) + "\n")
		f.write("Sentence indices:  " + " ".join([str(l) for l in sorted_indices]) + "\n")
		f.write("\n=================================================================================\n\n")
		
	## Using BM25 Extra using embeddings + Attention based EmbSum for relevance ranking and mAP calculation

	similarity_scores_Attention_Based_EmbSum = np.array(similarity_scores_Attention_Based_EmbSum)
	similarity_scores_Attention_Based_EmbSum[(similarity_scores_Attention_Based_EmbSum<0.0)] = 0.0
	relevance_score_bm25_extra_with_emb = np.array(relevance_score_bm25_extra_with_emb)
	relevance_score_bm25_extra_with_emb[(relevance_score_bm25_extra_with_emb<0.0)] = 0.0

	similarity_scores_Attention_Based_EmbSum_combined = np.squeeze(np.add(softmax(np.expand_dims(similarity_scores_Attention_Based_EmbSum, axis=0)), softmax(np.expand_dims(relevance_score_bm25_extra_with_emb, axis=0))), axis=0)

	AP, relevance_labels, relevance_scores, sorted_indices = eval.most_relevant(doc_true_relevance_labels, similarity_scores_Attention_Based_EmbSum_combined)

	bm25_Extra_with_embeddings_attention_based_EmbSum_AP_list.append(AP)
	bm25_Extra_with_embeddings_attention_based_EmbSum_acc_list.append(relevance_labels[0])

	# logging information
	with open(os.path.join(log_dir, "reload_info_clusters_with_BM25_Extra_using_embeddings_and_attention_based_EmbSum.txt"), "a") as f:
		f.write("Doc " + str(doc_counter) + ": " + str(query) + ": " + str(id) + "\n\n")
		f.write("Average precision: " + str(AP) + "\n")
		f.write("Predicted_labels:  " + " ".join([str(int(l)) for l in relevance_labels]) + "\n")
		f.write("Predicted_probs:   " + " ".join([str(l) for l in relevance_scores]) + "\n")
		f.write("Sentence indices:  " + " ".join([str(l) for l in sorted_indices]) + "\n")
		f.write("\n=================================================================================\n\n")


classwise_scores = []
for label in val_scores_dict.keys():
	class_val_prec = np.mean(val_scores_dict[label])
	classwise_scores.append(class_val_prec)

with open(os.path.join(log_dir, "classwise_scores.txt"), "w") as f:
	for label, scores in val_scores_dict.items():
		f.write("Label:  " + label + "\n")
		f.write("ids:    " + str(val_ids_dict[label]) + "\n")
		f.write("scores:  " + str(scores) + "\n")
		f.write("prec:  " + str(np.mean(scores)) + "\n")
		f.write("\n\n")

with open(os.path.join(log_dir, "results_file.txt"), "w") as f:
	for label, scores in val_scores_dict.items():
		pmids = val_ids_dict[label]
		sents = val_sents_dict[label]
		if label == "acute threat fear":
			f.write("Acute_Threat_Fear" + "\n")
		if label == "loss":
			f.write("Loss" + "\n")
		if label == "arousal":
			f.write("Arousal" + "\n")
		if label == "circadian rhythms":
			f.write("Circadian_Rhythms" + "\n")
		if label == "frustrative nonreward":
			f.write("Frustrative_Nonreward" + "\n")
		if label == "potential threat anxiety":
			f.write("Potential_Threat_Anxiety" + "\n")
		if label == "sleep wakefulness":
			f.write("Sleep_Wakefulness" + "\n")
		if label == "sustained threat":
			f.write("Sustained_Threat" + "\n")
		#f.write(label + "\n")
		for pmid, sent in zip(pmids, sents):
			f.write(pmid + "\t" + sent + "\n")



with open(os.path.join(log_dir, "summary.txt"), "w") as f:
	f.write("bm25_acc_average: " + str(np.mean(bm25_acc_list)) + "\n")
	f.write("bm25_extra_acc_average: " + str(np.mean(bm25_extra_acc_list)) + "\n")
	f.write("EmbSum_acc_average: " + str(np.mean(EmbSum_acc_list)) + "\n")
	f.write("bm25_Extra_EmbSum_acc_average: " + str(np.mean(bm25_Extra_EmbSum_acc_list)) + "\n")
	f.write("attention_based_EmbSum_acc_average: " + str(np.mean(attention_based_EmbSum_acc_list)) + "\n")
	f.write("bm25_Extra_attention_based_EmbSum_acc_average: " + str(np.mean(bm25_Extra_attention_based_EmbSum_acc_list)) + "\n")
	f.write("bm25_Extra_with_embeddings_acc_average: " + str(np.mean(bm25_Extra_with_embeddings_acc_list)) + "\n")
	f.write("bm25_Extra_with_embeddings_EmbSum_acc_average: " + str(np.mean(bm25_Extra_with_embeddings_EmbSum_acc_list)) + "\n")
	f.write("bm25_Extra_with_embeddings_attention_based_EmbSum_acc_average: " + str(np.mean(bm25_Extra_with_embeddings_attention_based_EmbSum_acc_list)) + "\n")
	f.write("\n\n=====================================================================================\n\n")
	f.write("bm25_AP_average: " + str(np.mean(bm25_AP_list)) + "\n")
	f.write("bm25_extra_AP_average: " + str(np.mean(bm25_extra_AP_list)) + "\n")
	f.write("EmbSum_AP_average: " + str(np.mean(EmbSum_AP_list)) + "\n")
	f.write("bm25_Extra_EmbSum_AP_average: " + str(np.mean(bm25_Extra_EmbSum_AP_list)) + "\n")
	f.write("attention_based_EmbSum_AP_average: " + str(np.mean(attention_based_EmbSum_AP_list)) + "\n")
	f.write("bm25_Extra_attention_based_EmbSum_AP_average: " + str(np.mean(bm25_Extra_attention_based_EmbSum_AP_list)) + "\n")
	f.write("bm25_Extra_with_embeddings_AP_average: " + str(np.mean(bm25_Extra_with_embeddings_AP_list)) + "\n")
	f.write("bm25_Extra_with_embeddings_EmbSum_AP_average: " + str(np.mean(bm25_Extra_with_embeddings_EmbSum_AP_list)) + "\n")
	f.write("bm25_Extra_with_embeddings_attention_based_EmbSum_AP_average: " + str(np.mean(bm25_Extra_with_embeddings_attention_based_EmbSum_AP_list)) + "\n")

print("Complete.")