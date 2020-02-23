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

seed = 42
tf_op_seed = 1234

np.random.seed(seed)
tf.set_random_seed(seed)


params = {}
params['dataset'] = "./datasets/Task2_without_acronym"
params['dataset_test'] = "./datasets/Task2_test_data_combined_batch_without_acronym"
params['model'] = "./model/Task2_without_acronym_supervised_exp_classwise"
params['use_bio_prior'] = False
params['use_fasttext_prior'] = True
params['use_BOW_repesentation'] = False
params['use_DocNADE_W'] = False
params['split_abstract_title'] = False
params['attention_EmbSum_type'] = "sum"
params['use_label_as_query'] = True
params['use_title_as_query'] = True
params['label_title_weights'] = [0.7, 0.3]

#params['dataset_test_original'] = "./RDoC_raw_data/RDoCTask/RDoCTask2TestData/Batch_1/"
params['dataset_test_original'] = "./RDoC_raw_data/RDoCTask/RDoCTask2TestData/Combined_batch/"

## Training hyperparameters

params['num_cores'] = 2
params['relevance_hidden_size'] = 100
params['relevance_num_epochs'] = 100
params['relevance_learning_rate'] = 0.001
params['include_bm25_extra_feats'] = True

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

if params['include_bm25_extra_feats']:
	params['model'] += "_with_bm25_extra"

params['model'] += "_" + params['attention_EmbSum_type']

params['model'] += "_" + str(params['relevance_hidden_size'])

now = datetime.datetime.now()

params['model'] += "_" + str(now.day) + "_" + str(now.month) + "_" + str(now.year)

#params['model'] += "_only_validation_with_thresh_0.8_without_softmax"
#params['model'] += "_only_test_with_thresh_0.8_without_softmax_label_title_bm25_simlabel_simtitle_concat_l1_with_sigmoid"
#params['model'] += "_only_test_with_thresh_0.8_without_softmax_label_title_bm25_simlabel_simtitle_concat_l1_without_sigmoid"
#params['model'] += "_only_test_with_thresh_0.8_without_softmax_label_bm25_simlabel_concat_l1_with_sigmoid"
#params['model'] += "_only_test_with_thresh_0.8_without_softmax_label_bm25_simlabel_concat_l1_without_sigmoid"
#params['model'] += "_only_test_with_thresh_0.8_without_softmax_label_bm25_simlabel_concat_l1_without_sigmoid_xx_y_crafted_label_title_W_different"
#params['model'] += "_only_test_with_thresh_0.8_without_softmax_label_bm25_simlabel_concat_l1_without_sigmoid_xx_y_crafted_label_title_with_unsup_score_W_same"
#params['model'] += "_only_test_with_thresh_0.8_without_softmax_label_bm25_simlabel_concat_l1_without_sigmoid_xx_y_crafted_label_title_with_unsup_score_W_different"
#params['model'] += "_test_combined_batch_with_thresh_0.8_without_softmax_label_23_with_sigmoid_W_shared"
#params['model'] += "_test_combined_batch_with_thresh_0.8_without_softmax_label_18_with_expansion"
#params['model'] += "_test_combined_batch_with_thresh_0.8_without_softmax_label_23_W_same"
params['model'] += "_test_combined_batch_with_thresh_0.8_without_softmax_label_24_W_different"
#params['model'] += "_only_test_with_thresh_0.8_without_softmax_label_title_l1_x1_yy_new"
#params['model'] += "_only_test_with_thresh_0.8_without_softmax_label_title_l1_xx_y_crafted_label_title_W_different"
#params['model'] += "_only_test_with_thresh_0.8_without_softmax_label_title_l1_xx_y_crafted_label_title_W_same"

log_dir = os.path.join(params['model'], 'logs')

if not os.path.exists(log_dir):
	os.makedirs(log_dir)

train_docs = []
train_relevant_context = []
train_labels = []
train_titles = []
train_ids = []

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

test_docs = []
test_relevant_context = []
test_labels = []
test_titles = []
test_ids = []

with open(params['dataset_test'] + "/test_docs.txt", "r") as f:
	for line in f.readlines():
		id, label, title, doc, rel_con = line.strip().split("<<>>")
		test_docs.append(doc.split("\t"))
		test_relevant_context.append(rel_con.split("\t"))
		test_labels.append(" ".join(label.lower().strip().split("_")))
		test_titles.append(title)
		test_ids.append(id)

test_original_docs = []
with open(params['dataset_test_original'] + "/test_docs.txt", "r") as f:
	for line in f.readlines():
		id, label, title, doc, rel_con = line.strip().split("<<>>")
		test_original_docs.append(doc.split("\t"))

#import pdb; pdb.set_trace()

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

if params['use_title_as_query']:
	queries.extend(train_titles)
	queries.extend(val_titles)
	queries.extend(test_titles)

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
print("Vocab size unigrams: ", len(total_vocab))

total_docs = []
for doc in train_docs:
	for sent in doc:
		total_docs.append(sent)

for doc in val_docs:
	for sent in doc:
		total_docs.append(sent)

def tokens(text):
	return [w.lower() for w in text.split()]

cv = CountVectorizer(tokenizer=tokens, min_df=1, max_df=1.0, ngram_range=(2,2), max_features=None, encoding='utf-8', decode_error='ignore')
cv.fit(total_docs)
total_vocab_bigrams = cv.get_feature_names()
print("Vocab size bigrams: ", len(total_vocab_bigrams))

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


def get_AB_EmbSum(query, doc):

	query_vecs = []
	tokens = [int(query_words_list.index(word)) for word in query.strip().split()]
	for token in tokens:
		query_emb = query_embedding_matrix[np.array(token), :]
		query_vecs.append(query_emb)
	
	doc_sents_vecs = []
	for j, sent in enumerate(doc):
		tokens = [docnade_vocab_large.index(word.strip()) for word in sent.strip().split()]
		Embs = docnade_embedding_matrix[np.array(tokens), :]
		query_tokens = query.split()
		EmbSum_attns = []
		for qword in query_tokens:
			query_vector = query_embedding_matrix[query_words_list.index(qword), :]
			query_vector = np.expand_dims(query_vector, axis=0)
			query_attentions = pw.cosine_similarity(query_vector, Embs)
			#query_attentions[(query_attentions < 0.8)] = 0.0
			query_attentions = softmax(query_attentions)
			EmbSum_attentions = np.dot(query_attentions, Embs)
			EmbSum_attns.append(EmbSum_attentions)

		EmbSum_attns = np.squeeze(EmbSum_attns)
		if len(EmbSum_attns.shape) == 1:
			EmbSum_attns = np.expand_dims(EmbSum_attns, axis=0)
		doc_sents_vecs.append(EmbSum_attns)
	#import pdb; pdb.set_trace()
	sim = pw.cosine_similarity(np.array(doc_sents_vecs).sum(axis=1), np.sum(query_vecs, axis=0, keepdims=True))

	return query_vecs, doc_sents_vecs
	#return query_vecs, doc_sents_vecs, sim

def get_AB_EmbSum_sim(query, doc):

	query_vecs = []
	tokens = [int(query_words_list.index(word)) for word in query.strip().split()]
	for token in tokens:
		query_emb = query_embedding_matrix[np.array(token), :]
		query_vecs.append(query_emb)
	
	doc_sents_vecs = []
	for j, sent in enumerate(doc):
		tokens = [docnade_vocab_large.index(word.strip()) for word in sent.strip().split()]
		Embs = docnade_embedding_matrix[np.array(tokens), :]
		query_tokens = query.split()
		EmbSum_attns = []
		for qword in query_tokens:
			query_vector = query_embedding_matrix[query_words_list.index(qword), :]
			query_vector = np.expand_dims(query_vector, axis=0)
			query_attentions = pw.cosine_similarity(query_vector, Embs)
			query_attentions[(query_attentions < 0.8)] = 0.0
			#query_attentions = softmax(query_attentions)
			EmbSum_attentions = np.dot(query_attentions, Embs)
			EmbSum_attns.append(EmbSum_attentions)

		EmbSum_attns = np.squeeze(EmbSum_attns)
		if len(EmbSum_attns.shape) == 1:
			EmbSum_attns = np.expand_dims(EmbSum_attns, axis=0)
		doc_sents_vecs.append(EmbSum_attns)
	#import pdb; pdb.set_trace()
	sim = pw.cosine_similarity(np.array(doc_sents_vecs).sum(axis=1), np.sum(query_vecs, axis=0, keepdims=True))

	#return query_vecs, doc_sents_vecs
	#return query_vecs, doc_sents_vecs, sim
	return sim


def get_EmbSum(query, doc):

	query_vecs = []
	tokens = [int(query_words_list.index(word)) for word in query.strip().split()]
	for token in tokens:
		query_emb = query_embedding_matrix[np.array(token), :]
		query_vecs.append(query_emb)

	doc_sents_vecs = []
	for sent in doc:
		tokens = [docnade_vocab_large.index(word.strip()) for word in sent.strip().split()]
		Embs = docnade_embedding_matrix[np.array(tokens), :]
		EmbSum = np.sum(Embs, axis=0)
		doc_sents_vecs.append(EmbSum)
	
	return query_vecs, doc_sents_vecs

def get_one_hot(query, doc):

	query_vecs = np.zeros((len(doc), len(total_vocab)), dtype=np.float32)
	indices = [total_vocab.index(word.strip()) for word in query.strip().split()]
	for index in indices:
		query_vecs[:, index] = 1.0

	doc_sents_vecs = np.zeros((len(doc), len(total_vocab)), dtype=np.float32)
	for i, sent in enumerate(doc):
		indices = [total_vocab.index(word.strip()) for word in sent.strip().split()]
		for index in indices:
			doc_sents_vecs[i, index] = 1.0
	
	return query_vecs, doc_sents_vecs

def get_bigram(query, doc):

	#import pdb; pdb.set_trace()

	query_vecs = np.array(cv.transform([query]).todense(), dtype=np.float32)
	query_vecs = np.tile(query_vecs, [len(doc), 1])

	doc_sents_vecs = []
	for i, sent in enumerate(doc):
		doc_sents_vecs.append(np.squeeze(np.array(cv.transform([sent]).todense()), axis=0))
	doc_sents_vecs = np.array(doc_sents_vecs, dtype=np.float32)
	
	return query_vecs, doc_sents_vecs


## Relevance network architecture

input_x_doc = tf.placeholder(tf.float32, shape=(None, None), name='input_x_doc')
input_x_query = tf.placeholder(tf.float32, shape=(None, None), name='input_x_query')
input_x_doc_2 = tf.placeholder(tf.float32, shape=(None, None), name='input_x_doc_2')
input_x_query_2 = tf.placeholder(tf.float32, shape=(None, None), name='input_x_query_2')
#input_x_doc_one_hot = tf.placeholder(tf.float32, shape=(None, None), name='input_x_doc_one_hot')
#input_x_query_one_hot = tf.placeholder(tf.float32, shape=(None, None), name='input_x_query_one_hot')
input_sim_1 = tf.placeholder(tf.float32, shape=(None), name='input_sim_1')
input_sim_2 = tf.placeholder(tf.float32, shape=(None), name='input_sim_2')
unsupervised_input = tf.placeholder(tf.float32, shape=(None), name='unsupervised_input')
if params['include_bm25_extra_feats']:
	#input_x_extra_feats = tf.placeholder(tf.float32, shape=(None, None), name='input_x_extra_feats')
	input_x_extra_feats = tf.placeholder(tf.float32, shape=(None), name='input_x_extra_feats')
input_y = tf.placeholder(tf.float32, shape=(None), name='input_y')

with tf.variable_scope("relevance_variables", reuse=tf.AUTO_REUSE):
	#max_embed_init = 1.0 / ((docnade_embedding_matrix.shape[1] * 2 + len(total_vocab) + len(total_vocab_bigrams)) * params['relevance_hidden_size'])
	#max_embed_init = 1.0 / (docnade_embedding_matrix.shape[1] * 2 * params['relevance_hidden_size'])
	max_embed_init = 1.0 / (docnade_embedding_matrix.shape[1] * params['relevance_hidden_size'])
	W_relevance = tf.get_variable(
		'W_relevance',
		#[docnade_embedding_matrix.shape[1] * 2 + len(total_vocab) + len(total_vocab_bigrams), params['relevance_hidden_size']],
		#[docnade_embedding_matrix.shape[1] * 2 + len(total_vocab) + len(total_vocab_bigrams), params['relevance_hidden_size']],
		#[docnade_embedding_matrix.shape[1] * 2, params['relevance_hidden_size']],
		[docnade_embedding_matrix.shape[1], params['relevance_hidden_size']],
		initializer=tf.random_uniform_initializer(
			maxval=max_embed_init,
			seed=tf_op_seed
		)
	)

	bias_relevance = tf.get_variable(
		'bias_relevance',
		[params['relevance_hidden_size']],
		initializer=tf.constant_initializer(0)
	)
	
	W_title_relevance = tf.get_variable(
		'W_title_relevance',
		#[docnade_embedding_matrix.shape[1] * 2 + len(total_vocab) + len(total_vocab_bigrams), params['relevance_hidden_size']],
		#[docnade_embedding_matrix.shape[1] * 2 + len(total_vocab) + len(total_vocab_bigrams), params['relevance_hidden_size']],
		#[docnade_embedding_matrix.shape[1] * 2, params['relevance_hidden_size']],
		[docnade_embedding_matrix.shape[1], params['relevance_hidden_size']],
		initializer=tf.random_uniform_initializer(
			maxval=max_embed_init,
			seed=tf_op_seed
		)
	)

	bias_title_relevance = tf.get_variable(
		'bias_title_relevance',
		[params['relevance_hidden_size']],
		initializer=tf.constant_initializer(0)
	)
	

relevance_output_doc = tf.tanh(tf.nn.xw_plus_b(input_x_doc, W_relevance, bias_relevance), name='relevance_output_doc')
relevance_output_query = tf.tanh(tf.nn.xw_plus_b(input_x_query, W_relevance, bias_relevance), name='relevance_output_query')

#relevance_output_doc_2 = tf.tanh(tf.nn.xw_plus_b(input_x_doc_2, W_relevance, bias_relevance), name='relevance_output_doc_2')
#relevance_output_query_2 = tf.tanh(tf.nn.xw_plus_b(input_x_query_2, W_relevance, bias_relevance), name='relevance_output_query_2')

relevance_output_doc_2 = tf.tanh(tf.nn.xw_plus_b(input_x_doc_2, W_title_relevance, bias_title_relevance), name='relevance_output_doc_2')
relevance_output_query_2 = tf.tanh(tf.nn.xw_plus_b(input_x_query_2, W_title_relevance, bias_title_relevance), name='relevance_output_query_2')

#relevance_output_doc = tf.concat([relevance_output_doc, input_x_doc], axis=1)
#relevance_output_query = tf.concat([relevance_output_query, input_x_query], axis=1)

#relevance_output_doc = tf.concat([relevance_output_doc, input_x_doc_one_hot], axis=1)
#relevance_output_query = tf.concat([relevance_output_query, input_x_query_one_hot], axis=1)

if params['include_bm25_extra_feats']:
	#relevance_output_temp = tf.exp(-1.0 * tf.norm(tf.subtract(relevance_output_doc, relevance_output_query), ord='euclidean', axis=1, name='relevance_score'))
	relevance_output_temp = tf.exp(-1.0 * tf.norm(tf.subtract(relevance_output_doc, relevance_output_query), ord=1, axis=1, name='relevance_score'))
	#relevance_output_temp_2 = tf.exp(-1.0 * tf.norm(tf.subtract(relevance_output_doc_2, relevance_output_query_2), ord='euclidean', axis=1, name='relevance_score_2'))
	relevance_output_temp_2 = tf.exp(-1.0 * tf.norm(tf.subtract(relevance_output_doc_2, relevance_output_query_2), ord=1, axis=1, name='relevance_score_2'))

	#relevance_output_temp = relevance_output_temp + input_sim_2 * relevance_output_temp_2
	relevance_output_temp = input_sim_1 * relevance_output_temp + input_sim_2 * relevance_output_temp_2
	
	#relevance_output_concat = tf.stack([relevance_output_temp, relevance_output_temp_2, input_x_extra_feats, input_sim_1, input_sim_2], axis=1)
	relevance_output_concat = tf.stack([relevance_output_temp, input_x_extra_feats, input_sim_1], axis=1)
	#relevance_output_concat = tf.stack([relevance_output_temp, input_x_extra_feats, input_sim_1, unsupervised_input], axis=1)

	with tf.variable_scope("relevance_variables", reuse=tf.AUTO_REUSE):
		#max_embed_init = 1.0 / (5 * 1)
		max_embed_init = 1.0 / (3 * 1)
		#max_embed_init = 1.0 / (4 * 1)
		"""
		W2_relevance = tf.get_variable(
			'W2_relevance',
			[5, 1],
			initializer=tf.random_uniform_initializer(
				maxval=max_embed_init,
				seed=tf_op_seed
			)
		)

		bias2_relevance = tf.get_variable(
			'bias2_relevance',
			[1],
			initializer=tf.constant_initializer(0)
		)
		"""
		W2_relevance = tf.get_variable(
			'W2_relevance',
			#[5],
			[3],
			#[4],
			initializer=tf.random_uniform_initializer(
				maxval=max_embed_init,
				seed=tf_op_seed
			)
		)

	#relevance_output = tf.sigmoid(tf.nn.xw_plus_b(relevance_output_concat, W2_relevance, bias2_relevance), name='relevance_score')
	#relevance_output = tf.nn.xw_plus_b(relevance_output_concat, W2_relevance, bias2_relevance)
	#relevance_output = tf.sigmoid(tf.tensordot(relevance_output_concat, W2_relevance, 1), name='relevance_score')
	relevance_output = tf.tensordot(relevance_output_concat, W2_relevance, 1)
else:
	#relevance_output = tf.exp(-1.0 * tf.norm(tf.subtract(relevance_output_doc, relevance_output_query), ord='euclidean', axis=1, name='relevance_score'))
	relevance_output_1 = tf.exp(-1.0 * tf.norm(tf.subtract(relevance_output_doc, relevance_output_query), ord=1, axis=1, name='relevance_score'))
	#relevance_output_2 = tf.exp(-1.0 * tf.norm(tf.subtract(relevance_output_doc_2, relevance_output_query_2), ord='euclidean', axis=1, name='relevance_score_2'))
	relevance_output_2 = tf.exp(-1.0 * tf.norm(tf.subtract(relevance_output_doc_2, relevance_output_query_2), ord=1, axis=1, name='relevance_score_2'))
	#relevance_output = (params['label_title_weights'][0] * relevance_output) + (params['label_title_weights'][1] * relevance_output_2)
	#relevance_output = (input_sim_1 * relevance_output_1) + (input_sim_2 * relevance_output_2)
	relevance_output = relevance_output_1
	#relevance_output = (relevance_output_1) + (input_sim_2 * relevance_output_2)

"""
if params['include_bm25_extra_feats']:
	input_x_extra_feats_clipped = tf.clip_by_value(input_x_extra_feats, 0.0000001, 1.0-0.0000001, name='clip')
	wt_before_softmax = tf.stack([relevance_output, input_x_extra_feats_clipped], axis=1)
	wt_after_softmax = tf.nn.softmax(wt_before_softmax, axis=1)
	relevance_output = tf.reduce_sum(tf.multiply(wt_after_softmax, wt_before_softmax), axis=1)
	#relevance_output = relevance_output + input_x_extra_feats
else:
	#wt_before_softmax = tf.stack([relevance_output, relevance_output_2], axis=1)
	#wt_after_softmax = tf.nn.softmax(wt_before_softmax, axis=1)
	#relevance_output = tf.reduce_sum(tf.multiply(wt_after_softmax, wt_before_softmax), axis=1)
	#relevance_output = (input_sim_1 * relevance_output) + (input_sim_2 * relevance_output_2)
	pass
"""

relevance_loss = tf.losses.mean_squared_error(input_y, relevance_output)

# Optimiser
step = tf.Variable(0, trainable=False)
relevance_trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='relevance_variables')

relevance_opt = m.gradients(
	opt=tf.train.AdamOptimizer(learning_rate=params['relevance_learning_rate']),
	loss=relevance_loss,
	vars=relevance_trainable_variables,
	step=step
)

## Training block

def get_bm25_ids(tokens):
	ids = []
	for token in tokens:
		try:
			ids.append(bm25.dictionary.token2id[token])
		except KeyError:
			pass
	return ids


with tf.Session(config=tf.ConfigProto(
	inter_op_parallelism_threads=params['num_cores'],
	intra_op_parallelism_threads=params['num_cores'],
	gpu_options=tf.GPUOptions(allow_growth=True)
)) as sess_relevance:
	model_dir_ppl = os.path.join(log_dir, 'model_ppl')

	saver = tf.train.Saver(tf.global_variables())

	tf.local_variables_initializer().run()
	tf.global_variables_initializer().run()

	best_val_prec = -1.0
	
	for epoch in range(params['relevance_num_epochs']):
		epoch_loss = 0.0

		doc_counter = -1
		for id, label, doc, title, rel_con in zip(train_ids, train_labels, train_docs, train_titles, train_relevant_context):
			doc_counter += 1

			#query = []
			#if params['use_label_as_query']:
			#	query.append(label)
			#if params['use_title_as_query']:
			#	query.append(title)

			#query = " ".join(query)

			temp_rel_con = []
			for rc_sent in rel_con:
				for sent in doc:
					if rc_sent in sent:
						temp_rel_con.append(sent)
						break

			if len(temp_rel_con) != len(set(rel_con)):
				print("Mismatch RC sentences.")
				import pdb; pdb.set_trace()
			
			doc_true_relevance_labels = []
			for sent in doc:
				if sent in temp_rel_con:
					doc_true_relevance_labels.append(float(1))
				else:
					doc_true_relevance_labels.append(float(0))

			#query_vecs, doc_vecs = get_EmbSum(query, doc)
			#query_vecs = np.tile(np.array(query_vecs).sum(axis=0, keepdims=True), [len(doc), 1])
			#doc_vecs = np.array(doc_vecs)

			#query_vecs, doc_vecs = get_AB_EmbSum(query, doc)
			#query_vecs = np.tile(np.expand_dims(query_vecs, axis=0), [len(doc), 1, 1])
			#if params['attention_EmbSum_type'] == "sum":
			#	query_vecs = query_vecs.sum(axis=1)
			#	doc_vecs = np.array(doc_vecs).sum(axis=1)

			train_feed_dict = {}
			
			if params['use_label_as_query'] and params['use_title_as_query']:
				if label == "frustrative nonreward":
					label = "physical relational aggression"
				if label == "arousal":
					label += " affective states heart rate"
				if label == "sustained threat":
					label += " amygdala attention bias"
				label_vecs, label_doc_vecs = get_AB_EmbSum(label, doc)
				#label_vecs, label_doc_vecs, label_sim = get_AB_EmbSum(label, doc)
				label_sim = get_AB_EmbSum_sim(label, doc)
				if label == "physical relational aggression":
					label = "frustrative nonreward"
				if label == "arousal affective states heart rate":
					label = "arousal"
				if label == "sustained threat amygdala attention bias":
					label = "sustained threat"
				label_vecs = np.tile(np.expand_dims(label_vecs, axis=0), [len(doc), 1, 1])
				if params['attention_EmbSum_type'] == "sum":
					label_vecs = label_vecs.sum(axis=1)
					label_doc_vecs = np.array(label_doc_vecs).sum(axis=1)
				#label_sim = pw.cosine_similarity(np.expand_dims(label_vecs[0], axis=1), label_doc_vecs)

				title_vecs, title_doc_vecs = get_AB_EmbSum(title, doc)
				#title_vecs, title_doc_vecs, title_sim = get_AB_EmbSum(title, doc)
				title_sim = get_AB_EmbSum_sim(title, doc)
				title_vecs = np.tile(np.expand_dims(title_vecs, axis=0), [len(doc), 1, 1])
				if params['attention_EmbSum_type'] == "sum":
					title_vecs = title_vecs.sum(axis=1)
					title_doc_vecs = np.array(title_doc_vecs).sum(axis=1)
				#title_sim = pw.cosine_similarity(np.expand_dims(title_vecs[0], axis=1), title_doc_vecs)

				#import pdb; pdb.set_trace()
				#doc_vecs = (params['label_title_weights'][0] * label_doc_vecs) + (params['label_title_weights'][1] * title_doc_vecs)
				#query_vecs = (params['label_title_weights'][0] * label_vecs) + (params['label_title_weights'][1] * title_vecs)
				#doc_vecs = np.multiply(label_sim, label_doc_vecs) + np.multiply(title_sim, title_doc_vecs)
				#query_vecs = np.multiply(label_sim, label_vecs) + np.multiply(title_sim, title_vecs)
				#import pdb; pdb.set_trace()
				label_title_sim = softmax(np.stack([label_sim.reshape(-1), title_sim.reshape(-1)], axis=1))
				#train_feed_dict[input_sim_1] = label_sim.reshape(-1)
				#train_feed_dict[input_sim_2] = title_sim.reshape(-1)
				train_feed_dict[input_sim_1] = label_title_sim[:,0]
				#train_feed_dict[input_sim_1] = label_sim.reshape(-1)
				#train_feed_dict[input_sim_2] = label_title_sim[:,1]
				temp = np.array((label_title_sim[:,0] < label_title_sim[:,1]), dtype=np.float32)
				temp_2 = temp * np.abs(label_title_sim[:,0] - label_title_sim[:,1])
				temp_3 = temp_2 * label_title_sim[:,1]
				train_feed_dict[input_sim_2] = temp_3
				#train_feed_dict[input_sim_2] = title_sim.reshape(-1)
				train_feed_dict[unsupervised_input] = label_title_sim[:,0] * label_sim.reshape(-1) + temp_3 * title_sim.reshape(-1)
				"""
				query_vecs = np.concatenate([label_vecs, title_vecs], axis=1)
				doc_vecs = np.concatenate([label_doc_vecs, title_doc_vecs], axis=1)
				
				#import pdb; pdb.set_trace()
				
				## Adding one-hot encoding
				query_vecs_one_hot, doc_vecs_one_hot = get_one_hot(label + " " + title, doc)

				## Adding bigrams representation
				query_vecs_bigram, doc_vecs_bigram = get_bigram(label + " " + title, doc)

				query_vecs_one_hot = np.concatenate([query_vecs_one_hot, query_vecs_bigram], axis=1)
				doc_vecs_one_hot = np.concatenate([doc_vecs_one_hot, doc_vecs_bigram], axis=1)

				query_vecs = np.concatenate([query_vecs, query_vecs_one_hot], axis=1)
				doc_vecs = np.concatenate([doc_vecs, doc_vecs_one_hot], axis=1)
				
				#train_feed_dict[input_x_doc_one_hot] = doc_vecs_one_hot
				#train_feed_dict[input_x_query_one_hot] = query_vecs_one_hot
				"""
			#train_feed_dict[input_x_doc] = doc_vecs
			#train_feed_dict[input_x_query] = query_vecs
			train_feed_dict[input_y] = doc_true_relevance_labels
			
			train_feed_dict[input_x_doc] = label_doc_vecs
			train_feed_dict[input_x_query] = label_vecs
			train_feed_dict[input_x_doc_2] = title_doc_vecs
			train_feed_dict[input_x_query_2] = title_vecs

			if params['include_bm25_extra_feats']:
				bm25 = BM25.BM25(doc, delimiter=' ')

				if label == "frustrative nonreward":
					label = "aggression"

				#relevance_score_bm25 = bm25.BM25Score(query.split())
				relevance_score_bm25 = bm25.BM25Score(label.split())
				relevance_score_bm25 = np.array(relevance_score_bm25, dtype=np.float32)

				if label == "aggression":
					label = "frustrative nonreward"

				extra_features = []
				for sent in doc:
					sent_ids = get_bm25_ids(sent.split())
					#query_ids = get_bm25_ids(query.split())
					query_ids = get_bm25_ids(label.split())
					feats = bm25.query_doc_overlap(query_ids, sent_ids)
					extra_features.append(feats)
				#import pdb; pdb.set_trace()
				relevance_score_bm25_extra = np.concatenate([np.expand_dims(relevance_score_bm25, axis=1), np.array(extra_features, dtype=np.float32)], axis=1)

				#train_feed_dict[input_x_extra_feats] = relevance_score_bm25_extra
				train_feed_dict[input_x_extra_feats] = relevance_score_bm25_extra.sum(axis=1)

			#import pdb; pdb.set_trace()
			#try:
			#_, loss, score, before, after = sess_relevance.run([relevance_opt, relevance_loss, relevance_output, wt_before_softmax, wt_after_softmax], feed_dict=train_feed_dict)
			_, loss, score = sess_relevance.run([relevance_opt, relevance_loss, relevance_output], feed_dict=train_feed_dict)
			#except ValueError:
			#	import pdb; pdb.set_trace()

			#import pdb; pdb.set_trace()

			epoch_loss += loss

		#print("BM25_before_clip: ", relevance_score_bm25_extra.sum(axis=1))
		#print("Before softmax: ", before)
		#print("After softmax: ", after)
		#print("Sum: ", score)
		#print("Relevance loss: ", epoch_loss)
		
		val_scores = []
		val_scores_dict = {}
		val_ids_dict = {}

		val_doc_counter = -1
		for id, label, doc, title, rel_con in zip(val_ids, val_labels, val_docs, val_titles, val_relevant_context):
			val_doc_counter += 1

			#query = []
			#if params['use_label_as_query']:
			#	query.append(label)
			#if params['use_title_as_query']:
			#	query.append(title)

			#query = " ".join(query)

			temp_rel_con = []
			for rc_sent in rel_con:
				for sent in doc:
					if rc_sent in sent:
						temp_rel_con.append(sent)
						break

			if len(temp_rel_con) != len(set(rel_con)):
				print("Mismatch RC sentences.")
				import pdb; pdb.set_trace()
			
			val_doc_true_relevance_labels = []
			for sent in doc:
				if sent in temp_rel_con:
					val_doc_true_relevance_labels.append(float(1))
				else:
					val_doc_true_relevance_labels.append(float(0))

			#val_query_vecs, val_doc_vecs = get_EmbSum(query, doc)
			#val_query_vecs = np.tile(np.array(val_query_vecs).sum(axis=0, keepdims=True), [len(doc), 1])
			#val_doc_vecs = np.array(val_doc_vecs)

			#val_query_vecs, val_doc_vecs = get_AB_EmbSum(query, doc)
			#val_query_vecs = np.tile(np.expand_dims(val_query_vecs, axis=0), [len(doc), 1, 1])
			#if params['attention_EmbSum_type'] == "sum":
			#	val_query_vecs = val_query_vecs.sum(axis=1)
			#	val_doc_vecs = np.array(val_doc_vecs).sum(axis=1)

			val_feed_dict = {}
			
			if params['use_label_as_query'] and params['use_title_as_query']:
				if label == "frustrative nonreward":
					label = "physical relational aggression"
				if label == "arousal":
					label += " affective states heart rate"
				if label == "sustained threat":
					label += " amygdala attention bias"
				label_vecs, label_doc_vecs = get_AB_EmbSum(label, doc)
				#label_vecs, label_doc_vecs, label_sim = get_AB_EmbSum(label, doc)
				label_sim = get_AB_EmbSum_sim(label, doc)
				if label == "physical relational aggression":
					label = "frustrative nonreward"
				if label == "arousal affective states heart rate":
					label = "arousal"
				if label == "sustained threat amygdala attention bias":
					label = "sustained threat"
				label_vecs = np.tile(np.expand_dims(label_vecs, axis=0), [len(doc), 1, 1])
				if params['attention_EmbSum_type'] == "sum":
					label_vecs = label_vecs.sum(axis=1)
					label_doc_vecs = np.array(label_doc_vecs).sum(axis=1)
				#label_sim = pw.cosine_similarity(np.expand_dims(label_vecs[0], axis=1), label_doc_vecs)

				title_vecs, title_doc_vecs = get_AB_EmbSum(title, doc)
				#title_vecs, title_doc_vecs, title_sim = get_AB_EmbSum(title, doc)
				title_sim = get_AB_EmbSum_sim(title, doc)
				title_vecs = np.tile(np.expand_dims(title_vecs, axis=0), [len(doc), 1, 1])
				if params['attention_EmbSum_type'] == "sum":
					title_vecs = title_vecs.sum(axis=1)
					title_doc_vecs = np.array(title_doc_vecs).sum(axis=1)
				#title_sim = pw.cosine_similarity(np.expand_dims(title_vecs[0], axis=1), title_doc_vecs)

				#val_doc_vecs = (params['label_title_weights'][0] * label_doc_vecs) + (params['label_title_weights'][1] * title_doc_vecs)
				#val_query_vecs = (params['label_title_weights'][0] * label_vecs) + (params['label_title_weights'][1] * title_vecs)
				#val_doc_vecs = np.multiply(label_sim, label_doc_vecs) + np.multiply(title_sim, title_doc_vecs)
				#val_query_vecs = np.multiply(label_sim, label_vecs) + np.multiply(title_sim, title_vecs)
				label_title_sim = softmax(np.stack([label_sim.reshape(-1), title_sim.reshape(-1)], axis=1))
				#val_feed_dict[input_sim_1] = label_sim.reshape(-1)
				#val_feed_dict[input_sim_2] = title_sim.reshape(-1)
				val_feed_dict[input_sim_1] = label_title_sim[:,0]
				#val_feed_dict[input_sim_1] = label_sim.reshape(-1)
				#val_feed_dict[input_sim_2] = label_title_sim[:,1]
				temp = np.array((label_title_sim[:,0] < label_title_sim[:,1]), dtype=np.float32)
				temp_2 = temp * np.abs(label_title_sim[:,0] - label_title_sim[:,1])
				temp_3 = temp_2 * label_title_sim[:,1]
				val_feed_dict[input_sim_2] = temp_3
				#val_feed_dict[input_sim_2] = title_sim.reshape(-1)
				val_feed_dict[unsupervised_input] = label_title_sim[:,0] * label_sim.reshape(-1) + temp_3 * title_sim.reshape(-1)
				"""
				val_query_vecs = np.concatenate([label_vecs, title_vecs], axis=1)
				val_doc_vecs = np.concatenate([label_doc_vecs, title_doc_vecs], axis=1)
				
				## Adding one-hot encoding
				val_query_vecs_one_hot, val_doc_vecs_one_hot = get_one_hot(label + " " + title, doc)

				## Adding bigrams representation
				val_query_vecs_bigram, val_doc_vecs_bigram = get_bigram(label + " " + title, doc)

				val_query_vecs_one_hot = np.concatenate([val_query_vecs_one_hot, val_query_vecs_bigram], axis=1)
				val_doc_vecs_one_hot = np.concatenate([val_doc_vecs_one_hot, val_doc_vecs_bigram], axis=1)

				val_query_vecs = np.concatenate([val_query_vecs, val_query_vecs_one_hot], axis=1)
				val_doc_vecs = np.concatenate([val_doc_vecs, val_doc_vecs_one_hot], axis=1)

				#val_feed_dict[input_x_doc_one_hot] = val_doc_vecs_one_hot
				#val_feed_dict[input_x_query_one_hot] = val_query_vecs_one_hot
				"""
			#val_feed_dict[input_x_doc] = val_doc_vecs
			#val_feed_dict[input_x_query] = val_query_vecs
			val_feed_dict[input_y] = val_doc_true_relevance_labels

			val_feed_dict[input_x_doc] = label_doc_vecs
			val_feed_dict[input_x_query] = label_vecs
			val_feed_dict[input_x_doc_2] = title_doc_vecs
			val_feed_dict[input_x_query_2] = title_vecs

			if params['include_bm25_extra_feats']:
				bm25 = BM25.BM25(doc, delimiter=' ')

				if label == "frustrative nonreward":
					label = "aggression"

				#val_relevance_score_bm25 = bm25.BM25Score(query.split())
				val_relevance_score_bm25 = bm25.BM25Score(label.split())
				val_relevance_score_bm25 = np.array(val_relevance_score_bm25, dtype=np.float32)

				if label == "aggression":
					label = "frustrative nonreward"

				val_extra_features = []
				for sent in doc:
					sent_ids = get_bm25_ids(sent.split())
					#query_ids = get_bm25_ids(query.split())
					query_ids = get_bm25_ids(label.split())
					feats = bm25.query_doc_overlap(query_ids, sent_ids)
					val_extra_features.append(feats)
				#import pdb; pdb.set_trace()
				val_relevance_score_bm25_extra = np.concatenate([np.expand_dims(val_relevance_score_bm25, axis=1), np.array(val_extra_features, dtype=np.float32)], axis=1)

				#val_feed_dict[input_x_extra_feats] = val_relevance_score_bm25_extra
				val_feed_dict[input_x_extra_feats] = val_relevance_score_bm25_extra.sum(axis=1)

			loss, score = sess_relevance.run([relevance_loss, relevance_output], feed_dict=val_feed_dict)

			AP, relevance_labels, relevance_scores, sorted_indices = eval.most_relevant(val_doc_true_relevance_labels, score)

			max_index = np.argmax(score)
			pred_label = val_doc_true_relevance_labels[max_index]

			val_scores.append(pred_label)

			if not label in val_scores_dict:
				val_scores_dict[label] = []
				val_ids_dict[label] = []
			val_scores_dict[label].append(pred_label)
			val_ids_dict[label].append(id)

			# logging information
			with open(log_dir + "/reload_info_clusters.txt", "a") as f:
				f.write("Doc " + str(val_doc_counter) + ": " + str(label + " <<>> " + title) + ": " + str(id) + "\n\n")
				f.write("Average precision: " + str(AP) + "\n")
				f.write("Predicted_labels:  " + " ".join([str(int(l)) for l in relevance_labels]) + "\n")
				f.write("Predicted_probs:   " + " ".join([str(l) for l in relevance_scores]) + "\n")
				f.write("Sentence indices:  " + " ".join([str(l) for l in sorted_indices]) + "\n")
				f.write("\n=================================================================================\n\n")
		
		val_prec = np.mean(val_scores)

		if val_prec > best_val_prec:
			best_val_prec = val_prec
			saver.save(sess_relevance, model_dir_ppl + '/model_ppl', global_step=1)

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

			test_scores = []
			test_scores_dict = {}
			test_ids_dict = {}
			test_sents_dict = {}

			test_doc_counter = -1
			for id, label, doc, title, rel_con in zip(test_ids, test_labels, test_docs, test_titles, test_relevant_context):
				test_doc_counter += 1

				#query = []
				#if params['use_label_as_query']:
				#	query.append(label)
				#if params['use_title_as_query']:
				#	query.append(title)

				#query = " ".join(query)

				temp_rel_con = []
				for rc_sent in rel_con:
					for sent in doc:
						if rc_sent in sent:
							temp_rel_con.append(sent)
							break

				#if len(temp_rel_con) != len(set(rel_con)):
				#	print("Mismatch RC sentences.")
				#	import pdb; pdb.set_trace()
				
				test_doc_true_relevance_labels = []
				for sent in doc:
					if sent in temp_rel_con:
						test_doc_true_relevance_labels.append(float(1))
					else:
						test_doc_true_relevance_labels.append(float(0))

				#val_query_vecs, val_doc_vecs = get_EmbSum(query, doc)
				#val_query_vecs = np.tile(np.array(val_query_vecs).sum(axis=0, keepdims=True), [len(doc), 1])
				#val_doc_vecs = np.array(val_doc_vecs)

				#val_query_vecs, val_doc_vecs = get_AB_EmbSum(query, doc)
				#val_query_vecs = np.tile(np.expand_dims(val_query_vecs, axis=0), [len(doc), 1, 1])
				#if params['attention_EmbSum_type'] == "sum":
				#	val_query_vecs = val_query_vecs.sum(axis=1)
				#	val_doc_vecs = np.array(val_doc_vecs).sum(axis=1)

				test_feed_dict = {}
				
				if params['use_label_as_query'] and params['use_title_as_query']:
					if label == "frustrative nonreward":
						label = "physical relational aggression"
					if label == "arousal":
						label += " affective states heart rate"
					if label == "sustained threat":
						label += " amygdala attention bias"
					label_vecs, label_doc_vecs = get_AB_EmbSum(label, doc)
					#label_vecs, label_doc_vecs, label_sim = get_AB_EmbSum(label, doc)
					label_sim = get_AB_EmbSum_sim(label, doc)
					if label == "physical relational aggression":
						label = "frustrative nonreward"
					if label == "arousal affective states heart rate":
						label = "arousal"
					if label == "sustained threat amygdala attention bias":
						label = "sustained threat"
					label_vecs = np.tile(np.expand_dims(label_vecs, axis=0), [len(doc), 1, 1])
					if params['attention_EmbSum_type'] == "sum":
						label_vecs = label_vecs.sum(axis=1)
						label_doc_vecs = np.array(label_doc_vecs).sum(axis=1)
					#label_sim = pw.cosine_similarity(np.expand_dims(label_vecs[0], axis=1), label_doc_vecs)

					title_vecs, title_doc_vecs = get_AB_EmbSum(title, doc)
					#title_vecs, title_doc_vecs, title_sim = get_AB_EmbSum(title, doc)
					title_sim = get_AB_EmbSum_sim(title, doc)
					title_vecs = np.tile(np.expand_dims(title_vecs, axis=0), [len(doc), 1, 1])
					if params['attention_EmbSum_type'] == "sum":
						title_vecs = title_vecs.sum(axis=1)
						title_doc_vecs = np.array(title_doc_vecs).sum(axis=1)
					#title_sim = pw.cosine_similarity(np.expand_dims(title_vecs[0], axis=1), title_doc_vecs)

					#val_doc_vecs = (params['label_title_weights'][0] * label_doc_vecs) + (params['label_title_weights'][1] * title_doc_vecs)
					#val_query_vecs = (params['label_title_weights'][0] * label_vecs) + (params['label_title_weights'][1] * title_vecs)
					#val_doc_vecs = np.multiply(label_sim, label_doc_vecs) + np.multiply(title_sim, title_doc_vecs)
					#val_query_vecs = np.multiply(label_sim, label_vecs) + np.multiply(title_sim, title_vecs)
					label_title_sim = softmax(np.stack([label_sim.reshape(-1), title_sim.reshape(-1)], axis=1))
					#test_feed_dict[input_sim_1] = label_sim.reshape(-1)
					#test_feed_dict[input_sim_2] = title_sim.reshape(-1)
					test_feed_dict[input_sim_1] = label_title_sim[:,0]
					#test_feed_dict[input_sim_1] = label_sim.reshape(-1)
					#test_feed_dict[input_sim_2] = label_title_sim[:,1]
					temp = np.array((label_title_sim[:,0] < label_title_sim[:,1]), dtype=np.float32)
					temp_2 = temp * np.abs(label_title_sim[:,0] - label_title_sim[:,1])
					temp_3 = temp_2 * label_title_sim[:,1]
					test_feed_dict[input_sim_2] = temp_3
					#test_feed_dict[input_sim_2] = title_sim.reshape(-1)
					test_feed_dict[unsupervised_input] = label_title_sim[:,0] * label_sim.reshape(-1) + temp_3 * title_sim.reshape(-1)
					"""
					test_query_vecs = np.concatenate([label_vecs, title_vecs], axis=1)
					test_doc_vecs = np.concatenate([label_doc_vecs, title_doc_vecs], axis=1)
					
					## Adding one-hot encoding
					val_query_vecs_one_hot, val_doc_vecs_one_hot = get_one_hot(label + " " + title, doc)

					## Adding bigrams representation
					val_query_vecs_bigram, val_doc_vecs_bigram = get_bigram(label + " " + title, doc)

					val_query_vecs_one_hot = np.concatenate([val_query_vecs_one_hot, val_query_vecs_bigram], axis=1)
					val_doc_vecs_one_hot = np.concatenate([val_doc_vecs_one_hot, val_doc_vecs_bigram], axis=1)

					val_query_vecs = np.concatenate([val_query_vecs, val_query_vecs_one_hot], axis=1)
					val_doc_vecs = np.concatenate([val_doc_vecs, val_doc_vecs_one_hot], axis=1)

					#val_feed_dict[input_x_doc_one_hot] = val_doc_vecs_one_hot
					#val_feed_dict[input_x_query_one_hot] = val_query_vecs_one_hot
					"""
				#test_feed_dict[input_x_doc] = test_doc_vecs
				#test_feed_dict[input_x_query] = test_query_vecs
				test_feed_dict[input_y] = test_doc_true_relevance_labels

				test_feed_dict[input_x_doc] = label_doc_vecs
				test_feed_dict[input_x_query] = label_vecs
				test_feed_dict[input_x_doc_2] = title_doc_vecs
				test_feed_dict[input_x_query_2] = title_vecs

				if params['include_bm25_extra_feats']:
					bm25 = BM25.BM25(doc, delimiter=' ')

					if label == "frustrative nonreward":
						label = "aggression"

					#test_relevance_score_bm25 = bm25.BM25Score(query.split())
					test_relevance_score_bm25 = bm25.BM25Score(label.split())
					test_relevance_score_bm25 = np.array(test_relevance_score_bm25, dtype=np.float32)

					if label == "aggression":
						label = "frustrative nonreward"

					test_extra_features = []
					for sent in doc:
						sent_ids = get_bm25_ids(sent.split())
						#query_ids = get_bm25_ids(query.split())
						query_ids = get_bm25_ids(label.split())
						feats = bm25.query_doc_overlap(query_ids, sent_ids)
						test_extra_features.append(feats)
					#import pdb; pdb.set_trace()
					test_relevance_score_bm25_extra = np.concatenate([np.expand_dims(test_relevance_score_bm25, axis=1), np.array(test_extra_features, dtype=np.float32)], axis=1)

					#test_feed_dict[input_x_extra_feats] = test_relevance_score_bm25_extra
					test_feed_dict[input_x_extra_feats] = test_relevance_score_bm25_extra.sum(axis=1)

				loss, score = sess_relevance.run([relevance_loss, relevance_output], feed_dict=test_feed_dict)

				AP, relevance_labels, relevance_scores, sorted_indices = eval.most_relevant(test_doc_true_relevance_labels, score)

				max_index = np.argmax(score)
				pred_label = test_doc_true_relevance_labels[max_index]

				test_scores.append(pred_label)

				if not label in test_scores_dict:
					test_scores_dict[label] = []
					test_ids_dict[label] = []
					test_sents_dict[label] = []
				test_scores_dict[label].append(pred_label)
				test_ids_dict[label].append(id)
				test_sents_dict[label].append(test_original_docs[test_doc_counter][sorted_indices[0]])
				if len(test_original_docs[test_doc_counter]) != len(doc):
					print("Mismatch lengths of original and preprocessed docs")
					import pdb; pdb.set_trace()

				# logging information
				with open(log_dir + "/reload_info_clusters_testdata.txt", "a") as f:
					f.write("Doc " + str(test_doc_counter) + ": " + str(label + " <<>> " + title) + ": " + str(id) + "\n\n")
					f.write("Average precision: " + str(AP) + "\n")
					f.write("Predicted_labels:  " + " ".join([str(int(l)) for l in relevance_labels]) + "\n")
					f.write("Predicted_probs:   " + " ".join([str(l) for l in relevance_scores]) + "\n")
					f.write("Sentence indices:  " + " ".join([str(l) for l in sorted_indices]) + "\n")
					f.write("\n=================================================================================\n\n")

			with open(os.path.join(log_dir, "results_file.txt"), "w") as f:
				for label, scores in test_scores_dict.items():
					pmids = test_ids_dict[label]
					sents = test_sents_dict[label]
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

		print("This val prec: %s, Best val prec: %s" % (val_prec, best_val_prec))
	
	with open(os.path.join(log_dir, "results_file_last_epoch.txt"), "w") as f:
		for label, scores in test_scores_dict.items():
			pmids = test_ids_dict[label]
			sents = test_sents_dict[label]
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
	#import pdb; pdb.set_trace()

print("Done.")