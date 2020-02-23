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
import BM25
from sklearn.utils.extmath import softmax

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


def perform_classification_test(train_data, val_data, test_data, c_list, classification_model="logistic", norm_before_classification=False):
	docVectors_train, train_labels = train_data
	docVectors_val, val_labels = val_data
	docVectors_test, test_labels = test_data

	if norm_before_classification:
		mean = np.mean(np.vstack((docVectors_train, docVectors_test)), axis=0)
		std = np.std(np.vstack((docVectors_train, docVectors_test)), axis=0)

		docVectors_train = (docVectors_train - mean) / std
		docVectors_test = (docVectors_test - mean) / std

	## Classification Accuracy
	val_acc = []
	val_f1 = []

	val_pred_labels = []
	val_pred_probs = []

	test_pred_probs_list = []
	test_pred_labels_list = []

	best_acc = 0.0
	clf = None
	
	for c in c_list:
		if classification_model == "logistic":
			clf = LogisticRegression(C=c)
		elif classification_model == "svm":
			#clf = SVC(C=c, kernel='precomputed')
			#clf = SVC(C=c)
			clf = SVC(C=c, probability=True, random_state=42)
		
		clf.fit(docVectors_train, train_labels)
		pred_val_labels = clf.predict(docVectors_val)
		pred_val_probs = clf.predict_proba(docVectors_val)

		acc_val = accuracy_score(val_labels, pred_val_labels)
		#f1_test = precision_recall_fscore_support(test_labels, pred_test_labels, pos_label=None, average='macro')[2]

		if acc_val > best_acc:
			best_acc = acc_val
			best_clf = clf

		val_acc.append(acc_val)

		val_pred_labels.append(pred_val_labels)
		val_pred_probs.append(pred_val_probs)

		test_pred_probs = clf.predict_proba(docVectors_test)
		test_pred_labels = clf.predict(docVectors_test)

		test_pred_labels_list.append(test_pred_labels)
		test_pred_probs_list.append(test_pred_probs)

	if classification_model == "logistic":
		return test_acc, test_f1
	elif classification_model == "svm":
		#return test_acc, test_f1, pred_test_probs
		#return test_acc, test_f1, test_pred_probs, test_pred_labels
		#return val_acc, val_f1, test_pred_probs, test_pred_labels
		return val_acc, val_f1, test_pred_probs_list, test_pred_labels_list


def reload_evaluation_f1(params, training_vectors, validation_vectors, test_vectors, suffix=""):

	### Classification - F1

	dataset = data.Dataset(params['dataset'])
	log_dir = os.path.join(params['model'], 'logs')

	if not os.path.exists(log_dir):
		os.makedirs(log_dir)

	#c_list = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
	#c_list = [1.0, 3.0, 5.0, 10.0, 100.0, 500.0, 1000.0, 10000.0]
	#c_list = [0.05, 0.5]
	#c_list = [0.001, 0.01, 0.1, 10.0]
	c_list = [0.01]

	test_acc = []
	test_f1 = []
	val_acc = []
	val_f1 = []

	test_acc_W = []
	test_f1_W = []
	val_acc_W = []
	val_f1_W = []

	y_train = np.array(
		[int(y) for y, _ in dataset.rows('training_docnade', num_epochs=1)]
	)
	y_val = np.array(
		[int(y) for y, _ in dataset.rows('validation_docnade', num_epochs=1)]
	)
	y_test = np.array(
		[int(y) for y, _ in dataset.rows('test_docnade', num_epochs=1)]
	)

	#import pdb; pdb.set_trace()

	queries = []
	with open(params['dataset'] + "/labels.txt", "r") as f:
		for line in f.readlines():
			label_tokens = line.lower().strip().split("_")
			label_tokens = [token for token in label_tokens if token]
			query = " ".join(label_tokens)
			queries.append(query)
	
	with open(params['dataset'] + "/test_ids.txt", "r") as f:
		ids = [line.strip() for line in f.readlines()]

	#id2label = {0:"Arousal", 1:"Circadian_Rhythms", 2:"Acute_Threat_Fear", 3:"Loss", 4:"Sleep_Wakefulness", 5:"Frustrative_Nonreward", 6:"Potential_Threat_Anxiety", 7:"Sustained_Threat"}
	id2label = {0:"Acute_Threat_Fear", 1:"Arousal", 2:"Circadian_Rhythms", 3:"Frustrative_Nonreward", 
				4:"Loss", 5:"Potential_Threat_Anxiety", 6:"Sleep_Wakefulness", 7:"Sustained_Threat"}

	train_data = (training_vectors, np.array(y_train, dtype=np.int32))
	validation_data = (validation_vectors, np.array(y_val, dtype=np.int32))
	test_data = (test_vectors, np.array(y_test, dtype=np.int32))

	#test_acc, test_f1 = perform_classification_test(train_data, test_data, c_list, classification_model="svm", norm_before_classification=False)
	test_acc, test_f1, pred_probs_list, pred_labels_list = perform_classification_test(train_data, validation_data,
	 		test_data, c_list, classification_model="svm", norm_before_classification=False)

	#import pdb; pdb.set_trace()
	#try:
	for c, pred_probs in zip(c_list, pred_probs_list):
		relevance_score_svm = []
		for i, label in enumerate(y_test):
			relevance_score_svm.append(pred_probs[i, int(label)])
		relevance_score_svm = np.array(relevance_score_svm)
		#except:
		#	import pdb; pdb.set_trace()

		dict_label = {int(label):[] for label in np.unique(y_test)}
		for score, id, label in zip(relevance_score_svm, ids, y_test):
			dict_label[int(label)].append([id,score])
		
		with open("task1_test_svm_classify" + str(c) + ".txt", "w") as f:
			for key in dict_label.keys():
				f.write(id2label[int(key)] + "\n")
				for id, score in dict_label[key]:
					f.write(id + "\t" + str(score) + "\n")

	#sys.exit()
	if evaluate_on_test:
		## Using BM25 Extra for relevance ranking and mAP calculation

		with open("./datasets/Task1_and_Task2_without_acronym_with_Task1_testdata_OOV_words/test.csv", "r") as f:
			file_reader = csv.reader(f, delimiter=',')
			docs = [line[1].strip() for line in file_reader]

		bm25 = BM25.BM25("./datasets/Task1_and_Task2_without_acronym_with_Task1_testdata_OOV_words/test.csv", delimiter=' ')

		def get_bm25_ids(tokens):
			ids = []
			for token in tokens:
				try:
					ids.append(bm25.dictionary.token2id[token])
				except KeyError:
					pass
			return ids

		bm25_extra_scores_list = []
		#for query in queries:
		for value in id2label.values():
			query = " ".join(value.lower().split("_")).strip()
			if query == "frustrative nonreward":
				query = "reward aggression"
			if query == "arousal":
				query += " affective states heart rate"
			query = query.split()
			scores = bm25.BM25Score(query)

			extra_features = []
			for doc in docs:
				doc = doc.split()
				#doc_ids = [bm25.dictionary.token2id[token] for token in doc]
				#query_ids = [bm25.dictionary.token2id[token] for token in query]
				doc_ids = get_bm25_ids(doc)
				query_ids = get_bm25_ids(query)

				feats = bm25.query_doc_overlap(query_ids, doc_ids)
				extra_features.append(np.sum(feats))
			#scores = np.stack([np.array(scores), np.array(extra_features)], axis=1)
			scores = np.add(np.array(scores), np.array(extra_features))

			bm25_extra_scores_list.append(scores)
		
		bm25_extra_scores_matrix = np.stack(bm25_extra_scores_list, axis=1)
		#import pdb; pdb.set_trace()
		relevance_score_bm25_extra = []
		for i, label in enumerate(y_test):
			relevance_score_bm25_extra.append(bm25_extra_scores_matrix[i, int(label)])
		relevance_score_bm25_extra = np.array(relevance_score_bm25_extra)

		dict_label = {label:[] for label in np.unique(y_test)}
		for score, id, label in zip(relevance_score_bm25_extra, ids, y_test):
			dict_label[label].append([id,score])

		with open("task1_test_bm25extra.txt", "w") as f:
			for key in dict_label.keys():
				f.write(id2label[int(key)] + "\n")
				for id, score in dict_label[key]:
					f.write(id + "\t" + str(score) + "\n")

		#combined_relevance_score = relevance_score_svm + relevance_score_bm25_extra
		pseudo_prediction = np.argmax(pred_probs_list[0], axis=1)
		combined_relevance_score = []
		counter = 0
		for svm_pred, bm25_score, svm_score, true_label in zip(pseudo_prediction, relevance_score_bm25_extra, relevance_score_svm, y_test):
		#for svm_pred, bm25_score, svm_score, true_label in zip(pred_labels_list[0], relevance_score_bm25_extra, relevance_score_svm, y_test):
			#import pdb; pdb.set_trace()
			if true_label == svm_pred:
				combined_relevance_score.append(svm_score + bm25_score)
				counter += 1
			else:
				combined_relevance_score.append(svm_score)
		print(counter)

		dict_label = {label:[] for label in np.unique(y_test)}
		for score, id, label in zip(combined_relevance_score, ids, y_test):
			dict_label[label].append([id,score])

		with open("task1_test_svm_classify_with_bm25extra.txt", "w") as f:
			for key in dict_label.keys():
				f.write(id2label[int(key)] + "\n")
				for id, score in dict_label[key]:
					f.write(id + "\t" + str(score) + "\n")

		with codecs.open("./pretrained_embeddings/biggest_vocab.vocab", "r", encoding='utf-8', errors='ignore') as f:
			total_vocab = [line.strip() for line in f.readlines()]

		prior_embedding_matrices = []

		if params['use_bio_prior']:
			#bio_embeddings_large = np.load('./pretrained_embeddings/bionlp_embeddings_biggest_vocab.npy')
			bio_embeddings_large = np.load('./pretrained_embeddings/' + params['bioemb_path'])
			prior_embedding_matrices.append(bio_embeddings_large)

		if params['use_fasttext_prior']:
			#fasttext_embeddings_large = np.load('./pretrained_embeddings/fasttext_embeddings_biggest_vocab.npy')
			fasttext_embeddings_large = np.load('./pretrained_embeddings/' + params['fttemb_path'])
			prior_embedding_matrices.append(fasttext_embeddings_large)
			
		if params['use_BOW_repesentation']:
			BOW_representations = np.eye(len(total_vocab), dtype=np.float32)
			prior_embedding_matrices.append(BOW_representations)

		total_embedding_matrix = np.concatenate(prior_embedding_matrices, axis=1)

		similarity_scores_Attention_Based_EmbSum = np.zeros((len(y_test), 8), dtype=np.float32)
		
		with open("./datasets/Task1_and_Task2_without_acronym_with_Task1_testdata_OOV_words/test.csv", "r") as f:
			file_reader = csv.reader(f, delimiter=",")
			for j, row in enumerate(file_reader):
				tokens = [total_vocab.index(word) for word in row[1].strip().split()]
				Embs = total_embedding_matrix[np.array(tokens), :]
				#for i, query in enumerate(queries):
				for k, value in enumerate(id2label.values()):
					query = " ".join(value.lower().split("_")).strip()
					if query == "frustrative nonreward":
						query = "reward aggression"
					if query == "arousal":
						query += " affective states heart rate"
					query_tokens = query.split()
					EmbSum_attns = []
					query_vecs_attns = []
					for qword in query_tokens:
						query_vector = total_embedding_matrix[total_vocab.index(qword), :]
						query_vector = np.expand_dims(query_vector, axis=0)
						query_attentions = pw.cosine_similarity(query_vector, Embs)
						#query_attentions[(query_attentions < 0.5)] = 0.0
						query_attentions = softmax(query_attentions)
						EmbSum_attentions = np.dot(query_attentions, Embs)
						EmbSum_attns.append(EmbSum_attentions)
						query_vecs_attns.append(query_vector)
					EmbSum = np.sum(EmbSum_attns, axis=0)
					#query_EmbSum_vector = np.expand_dims(query_vecs[i], axis=0)
					query_EmbSum_vector = np.sum(query_vecs_attns, axis=0)
					similarity_score = pw.cosine_similarity(query_EmbSum_vector, EmbSum)
					similarity_scores_Attention_Based_EmbSum[j, k] = similarity_score[0][0]

		relevance_score_att_embsum = []
		for i, label in enumerate(y_test):
			relevance_score_att_embsum.append(similarity_scores_Attention_Based_EmbSum[i, int(label)])
		relevance_score_att_embsum = np.array(relevance_score_att_embsum)

		#combined_relevance_score = relevance_score_svm + relevance_score_bm25_extra

		dict_label = {label:[] for label in np.unique(y_test)}
		for score, id, label in zip(relevance_score_att_embsum, ids, y_test):
			dict_label[label].append([id,score])

		with open("task1_test_att_based_embsum.txt", "w") as f:
			for key in dict_label.keys():
				f.write(id2label[int(key)] + "\n")
				for id, score in dict_label[key]:
					f.write(id + "\t" + str(score) + "\n")

		#combined_relevance_score_classify_embsum = relevance_score_att_embsum + relevance_score_svm
		pseudo_prediction = np.argmax(pred_probs_list[0], axis=1)
		combined_relevance_score_classify_embsum = []
		#positive_half_scores = []
		#negative_half_scores = []
		counter = 0
		for svm_pred, att_score, svm_score, true_label in zip(pseudo_prediction, relevance_score_att_embsum, relevance_score_svm, y_test):
		#for svm_pred, att_score, svm_score, true_label in zip(pred_labels_list[0], relevance_score_bm25_extra, relevance_score_svm, y_test):
			#import pdb; pdb.set_trace()
			if true_label == svm_pred:
				combined_relevance_score_classify_embsum.append(svm_score + att_score)
				#positive_half_scores.append()
				counter += 1
			else:
				combined_relevance_score_classify_embsum.append(svm_score)
		print(counter)

		dict_label = {label:[] for label in np.unique(y_test)}
		for score, id, label in zip(combined_relevance_score_classify_embsum, ids, y_test):
			dict_label[label].append([id,score])

		with open("task1_test_classify_att_based_embsum.txt", "w") as f:
			for key in dict_label.keys():
				f.write(id2label[int(key)] + "\n")
				for id, score in dict_label[key]:
					f.write(id + "\t" + str(score) + "\n")

		combined_relevance_score_classify_embsum_bm25extra = relevance_score_att_embsum + relevance_score_svm + relevance_score_bm25_extra

		dict_label = {label:[] for label in np.unique(y_test)}
		for score, id, label in zip(combined_relevance_score_classify_embsum_bm25extra, ids, y_test):
			dict_label[label].append([id,score])

		with open("task1_test_classify_att_based_embsum_bm25extra.txt", "w") as f:
			for key in dict_label.keys():
				f.write(id2label[int(key)] + "\n")
				for id, score in dict_label[key]:
					f.write(id + "\t" + str(score) + "\n")


def BOW_representation(tokens, vocab_size):
	vec = np.zeros(vocab_size, dtype=np.float32)
	for token in tokens:
		vec[int(token)] += 1.0
	return vec



dataset = "Task1_and_Task2_without_acronym_without_Task1_testdata_OOV_words"

evaluate_on_test = False
evaluate_on_validation = True

model_type = "SVM"
data_dir = "./datasets/" + dataset
params = {}
params['dataset'] = data_dir
params['num_classes'] = 8
params['multi_label'] = False
params['use_bio_prior'] = True
params['use_fasttext_prior'] = True
params['use_BOW_repesentation'] = True
params['split_abstract_title'] = False
params['include_sup_probs'] = True
params['attention_EmbSum_type'] = "sum"
hidden_size = 200

params['use_bio_prior_for_training'] = False
params['use_fasttext_prior_for_training'] = False
params['use_BOW_repesentation_for_training'] = True

params['model'] = os.path.join(os.getcwd(), "model", dataset)

if params['use_bio_prior_for_training']:
	params['model'] += "_bio_EmbSum"

if params['use_fasttext_prior_for_training']:
	params['model'] += "_ftt_EmbSum"

if params['use_BOW_repesentation_for_training']:
	params['model'] += "_BOW"

if params['include_sup_probs']:
	params['model'] += "_including_classification_prob"

params['bioemb_path'] = 'bionlp_embeddings_biggest_vocab.npy'
params['fttemb_path'] = 'fasttext_embeddings_biggest_vocab.npy'

with codecs.open(data_dir + "/vocab_docnade.vocab", "r", encoding='utf-8', errors='ignore') as f:
	vocab_docnade = [line.strip() for line in f.readlines()]

with codecs.open("./pretrained_embeddings/biggest_vocab.vocab", "r", encoding='utf-8', errors='ignore') as f:
	docnade_vocab_large = [line.strip() for line in f.readlines()]

prior_embedding_matrices = []

if params['use_bio_prior_for_training']:
	bio_embeddings_large = np.load('./pretrained_embeddings/' + params['bioemb_path'])
	new_bio_embs = []
	for word in vocab_docnade:
		new_bio_embs.append(bio_embeddings_large[docnade_vocab_large.index(word)])
	prior_embedding_matrices.append(np.array(new_bio_embs))

if params['use_fasttext_prior_for_training']:
	fasttext_embeddings_large = np.load('./pretrained_embeddings/' + params['fttemb_path'])
	new_ftt_embs = []
	for word in docnade_vocab:
		new_ftt_embs.append(fasttext_embeddings_large[docnade_vocab_large.index(word)])
	prior_embedding_matrices.append(np.array(new_ftt_embs))

if params['use_BOW_repesentation_for_training']:
	BOW_representations = np.eye(len(vocab_docnade), dtype=np.float32)
	prior_embedding_matrices.append(BOW_representations)

docnade_embedding_matrix = np.concatenate(prior_embedding_matrices, axis=1)

training_vecs = []
training_labels = []
with open(data_dir + "/training_docnade.csv", "r") as f:
	file_reader = csv.reader(f, delimiter=",")
	for row in file_reader:
		tokens = [int(index) for index in row[1].strip().split()]
		#if params['use_bio_prior'] or params['use_fasttext_prior']:
		Embs = docnade_embedding_matrix[np.array(tokens), :]
		EmbSum = np.sum(Embs, axis=0)
		training_vecs.append(EmbSum)
		#else:
		#	bow_representation = BOW_representation(tokens, len(vocab_docnade))
		#	training_vecs.append(bow_representation)
		training_labels.append([row[0]])

validation_vecs = []
validation_labels = []
with open(data_dir + "/validation_docnade.csv", "r") as f:
	file_reader = csv.reader(f, delimiter=",")
	for row in file_reader:
		tokens = [int(index) for index in row[1].strip().split()]
		#if params['use_bio_prior'] or params['use_fasttext_prior']:
		Embs = docnade_embedding_matrix[np.array(tokens), :]
		EmbSum = np.sum(Embs, axis=0)
		validation_vecs.append(EmbSum)
		#else:
		#	bow_representation = BOW_representation(tokens, len(vocab_docnade))
		#	validation_vecs.append(bow_representation)
		validation_labels.append([row[0]])

test_vecs = []
test_labels = []
with open(data_dir + "/test_docnade.csv", "r") as f:
	file_reader = csv.reader(f, delimiter=",")
	for row in file_reader:
		tokens = [int(index) for index in row[1].strip().split()]
		#if params['use_bio_prior'] or params['use_fasttext_prior']:
		Embs = docnade_embedding_matrix[np.array(tokens), :]
		EmbSum = np.sum(Embs, axis=0)
		test_vecs.append(EmbSum)
		#else:
		#	bow_representation = BOW_representation(tokens, len(vocab_docnade))
		#	test_vecs.append(bow_representation)
		test_labels.append([row[0]])

#import pdb; pdb.set_trace()

reload_evaluation_f1(params, training_vecs, validation_vecs, test_vecs)
print("Done.")