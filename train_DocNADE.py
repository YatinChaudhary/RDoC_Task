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

from gensim.models.keyedvectors import KeyedVectors
import sklearn.metrics.pairwise as pw
from sklearn.metrics import accuracy_score
import BM25
from sklearn.utils.extmath import softmax

os.environ['CUDA_VISIBLE_DEVICES'] = ''
#os.environ['KMP_DUPLICATE_LIB_OK']='True'

seed = 42
tf_op_seed = 1234

np.random.seed(seed)
tf.set_random_seed(seed)

home_dir = os.getenv("HOME")

dir(tf.contrib)


def loadGloveModel(gloveFile=None, params=None):
	if gloveFile is None:
		if params.hidden_size == 50:
			gloveFile = os.path.join(home_dir, "resources/pretrained_embeddings/glove.6B.50d.txt")
		elif params.hidden_size == 100:
			gloveFile = os.path.join(home_dir, "resources/pretrained_embeddings/glove.6B.100d.txt")
		elif params.hidden_size == 200:
			gloveFile = os.path.join(home_dir, "resources/pretrained_embeddings/glove.6B.200d.txt")
		elif params.hidden_size == 300:
			gloveFile = os.path.join(home_dir, "resources/pretrained_embeddings/glove.6B.300d.txt")
		else:
			print('Invalid dimension [%d] for Glove pretrained embedding matrix!!' %params.hidden_size)
			exit()

	print("Loading Glove Model")
	f = open(gloveFile, 'r')
	model = {}
	for line in f:
		splitLine = line.split()
		word = splitLine[0]
		embedding = np.array([float(val) for val in splitLine[1:]])
		model[word] = embedding
	print("Done.", len(model), " words loaded!")
	return model


def loadBioModel(BioFile=None, params=None):
	print("Loading BioNLP Model")
	#model = KeyedVectors.load_word2vec_format('./datasets/PubMed-w2v.bin', binary=True)
	model = KeyedVectors.load_word2vec_format('./datasets/PubMed-and-PMC-w2v.bin', binary=True)
	
	print("Binary model loaded!")
	return model


def train(model, dataset, params):
	log_dir = os.path.join(params.model, 'logs')
	model_dir_ir = os.path.join(params.model, 'model_ir')
	model_dir_ppl = os.path.join(params.model, 'model_ppl')
	model_dir_sup = os.path.join(params.model, 'model_sup')
	model_dir_mAP = os.path.join(params.model, 'model_mAP')

	with tf.Session(config=tf.ConfigProto(
		inter_op_parallelism_threads=params.num_cores,
		intra_op_parallelism_threads=params.num_cores,
		gpu_options=tf.GPUOptions(allow_growth=True)
	)) as session:
		avg_loss = tf.placeholder(tf.float32, [], 'loss_ph')
		tf.summary.scalar('loss', avg_loss)

		validation = tf.placeholder(tf.float32, [], 'validation_ph')
		validation_accuracy = tf.placeholder(tf.float32, [], 'validation_acc')
		tf.summary.scalar('validation', validation)
		tf.summary.scalar('validation_accuracy', validation_accuracy)

		summary_writer = tf.summary.FileWriter(log_dir, session.graph)
		summaries = tf.summary.merge_all()
		saver = tf.train.Saver(tf.global_variables())

		tf.local_variables_initializer().run()
		tf.global_variables_initializer().run()

		losses = []

		if params.input_type == "both":
			training_filename = 'training_docnade'
			validation_filename = 'validation_docnade'
			test_filename = 'test_docnade'
		elif params.input_type == "abstract":
			training_filename = 'training_docnade_abstracts'
			validation_filename = 'validation_docnade_abstracts'
			test_filename = 'test_docnade_abstracts'
		elif params.input_type == "title":
			training_filename = 'training_docnade_titles'
			validation_filename = 'validation_docnade_titles'
			test_filename = 'test_docnade_titles'
		else:
			print("Wrong value for params.input_type: ", params.input_type)
			sys.exit()

		if params.use_title_separately:
			training_title_filename = 'training_docnade_titles'
			validation_title_filename = 'validation_docnade_titles'
			test_title_filename = 'test_docnade_titles'

		# This currently streams from disk. You set num_epochs=1 and
		# wrap this call with something like itertools.cycle to keep
		# this data in memory.
		# shuffle: the order of words in the sentence for DocNADE
		
		#training_data = dataset.batches('training_docnade', params.batch_size, shuffle=True, multilabel=params.multi_label)
		training_data = dataset.batches(training_filename, params.batch_size, shuffle=True, multilabel=params.multi_label)

		if params.use_title_separately:
			training_title_data = dataset.batches(training_title_filename, params.batch_size, shuffle=True, multilabel=params.multi_label)

		id2label = {0:"Acute_Threat_Fear", 1:"Arousal", 2:"Circadian_Rhythms", 3:"Frustrative_Nonreward", 
				4:"Loss", 5:"Potential_Threat_Anxiety", 6:"Sleep_Wakefulness", 7:"Sustained_Threat"}

		with open(params.dataset + "/test_ids.txt", "r") as f:
			ids = [line.strip() for line in f.readlines()]

		best_val_IR = 0.0
		best_val_acc = 0.0
		best_val_mAP = 0.0
		best_val_nll = np.inf
		best_val_ppl = np.inf
		best_val_disc_accuracy = 0.0

		best_test_IR = 0.0
		best_test_nll = np.inf
		best_test_ppl = np.inf
		best_test_disc_accuracy = 0.0
		
		patience = params.patience

		patience_count = 0
		patience_count_ir = 0
		best_train_nll = np.inf

		training_labels = np.array(
			#[[y] for y, _ in dataset.rows('training_docnade', num_epochs=1)]
			[[y] for y, _ in dataset.rows(training_filename, num_epochs=1)]
		)
		validation_labels = np.array(
			#[[y] for y, _ in dataset.rows('validation_docnade', num_epochs=1)]
			[[y] for y, _ in dataset.rows(validation_filename, num_epochs=1)]
		)
		test_labels = np.array(
			#[[y] for y, _ in dataset.rows('test_docnade', num_epochs=1)]
			[[y] for y, _ in dataset.rows(test_filename, num_epochs=1)]
		)

		#initial_weights = session.run("embeddings_lambda_list_unclipped:0")
		#np.save(os.path.join(log_dir, "initial_sup_weights.npy"), initial_weights)

		for step in range(params.num_steps + 1):
			this_loss = -1.

			y, x, seq_lengths = next(training_data)

			train_feed_dict = {}
			train_feed_dict[model.x] = x
			train_feed_dict[model.y] = y
			train_feed_dict[model.seq_lengths] = seq_lengths

			if params.use_title_separately:
				y_title, x_title, seq_lengths_title = next(training_title_data)

				train_feed_dict[model.x_title] = x_title
				train_feed_dict[model.seq_lengths_title] = seq_lengths_title
			
			_, loss, loss_unnormed = session.run([model.opt, model.loss_normed, model.loss_unnormed], feed_dict=train_feed_dict)
			this_loss = loss
			losses.append(this_loss)

			if (step % params.log_every == 0):
				print('{}: {:.6f}'.format(step, this_loss))

			if step and (step % params.validation_ppl_freq) == 0:

				this_val_nll = []
				this_val_loss_normed = []
				# val_loss_unnormed is NLL
				this_val_nll_bw = []
				this_val_loss_normed_bw = []
				
				for val_y, val_x, val_seq_lengths in dataset.batches('validation_docnade', params.validation_bs, num_epochs=1, shuffle=True, multilabel=params.multi_label):
					
					val_loss_normed, val_loss_unnormed = session.run([model.loss_normed, model.loss_unnormed], feed_dict={
						model.x: val_x,
						model.y: val_y,
						model.seq_lengths: val_seq_lengths
					})

					this_val_nll.append(val_loss_unnormed)
					this_val_loss_normed.append(val_loss_normed)
				
				total_val_nll = np.mean(this_val_nll)
				total_val_ppl = np.exp(np.mean(this_val_loss_normed))

				if total_val_ppl < best_val_ppl:
					best_val_ppl = total_val_ppl
					print('saving: {}'.format(model_dir_ppl))
					saver.save(session, model_dir_ppl + '/model_ppl', global_step=1)

				# Early stopping
				if total_val_nll < best_val_nll:
					best_val_nll = total_val_nll
					patience_count = 0
				else:
					patience_count += 1

				print('This val PPL: {:.3f} (best val PPL: {:.3f},  best val loss: {:.3f}'.format(
					total_val_ppl,
					best_val_ppl or 0.0,
					best_val_nll
				))

				# logging information
				with open(os.path.join(log_dir, "training_info.txt"), "a") as f:
					f.write("Step: %i,	val PPL: %s,	 best val PPL: %s,	best val loss: %s\n" % 
							(step, total_val_ppl, best_val_ppl, best_val_nll))

				if patience_count > patience:
					print("Early stopping criterion satisfied.")
					break
			
			if step and (step % params.validation_ir_freq) == 0:

				## Classification accuracy

				if params.use_title_separately:
					validation_title_data = dataset.batches(validation_title_filename, params.validation_bs, num_epochs=1, shuffle=True, multilabel=params.multi_label)
					validation_filename = 'validation_docnade_abstracts'
				
				val_pred_labels = []
				val_pred_logits = []
				for val_y, val_x, val_seq_lengths in dataset.batches(validation_filename, params.validation_bs, num_epochs=1, shuffle=True, multilabel=params.multi_label):
					val_feed_dict = {}
					val_feed_dict[model.x] = val_x
					val_feed_dict[model.y] = val_y
					val_feed_dict[model.seq_lengths] = val_seq_lengths

					if params.use_title_separately:
						val_y_title, val_x_title, val_seq_lengths_title = next(validation_title_data)

						val_feed_dict[model.x_title] = val_x_title
						val_feed_dict[model.seq_lengths_title] = val_seq_lengths_title

					pred_labels, pred_logits = session.run([model.pred_labels, model.disc_output], feed_dict=val_feed_dict)

					#val_pred_labels.append(pred_labels[0][0])
					val_pred_labels.append(pred_labels[0])
					val_pred_logits.append(pred_logits[0])

				val_true_labels = [int(label[0]) for label in validation_labels]

				val_acc = accuracy_score(val_true_labels, val_pred_labels)

				if val_acc > best_val_acc:
					best_val_acc = val_acc
					print('saving: {}'.format(model_dir_sup))
					saver.save(session, model_dir_sup + '/model_sup', global_step=1)
					patience_count_ir = 0

					test_pred_labels = []
					test_pred_logits = []
					for test_y, test_x, test_seq_lengths in dataset.batches(test_filename, params.validation_bs, num_epochs=1, shuffle=True, multilabel=params.multi_label):
						test_feed_dict = {}
						test_feed_dict[model.x] = test_x
						test_feed_dict[model.y] = test_y
						test_feed_dict[model.seq_lengths] = test_seq_lengths

						if params.use_title_separately:
							test_y_title, test_x_title, test_seq_lengths_title = next(validation_title_data)

							test_feed_dict[model.x_title] = test_x_title
							test_feed_dict[model.seq_lengths_title] = test_seq_lengths_title

						pred_labels, pred_logits = session.run([model.pred_labels, model.disc_output], feed_dict=test_feed_dict)

						#test_pred_labels.append(pred_labels[0][0])
						test_pred_labels.append(pred_labels[0])
						test_pred_logits.append(pred_logits[0])

					test_pred_logits = softmax(np.array(test_pred_logits))

					test_true_labels = [int(label[0]) for label in test_labels]

					test_acc = accuracy_score(test_true_labels, test_pred_labels)

					np.save(os.path.join(log_dir, "test_pred_logits.npy"), np.array(test_pred_logits))
					np.save(os.path.join(log_dir, "test_pred_labels.npy"), np.array(test_pred_labels))
					np.save(os.path.join(log_dir, "test_true_labels.npy"), np.array(test_true_labels))

					docnade_probs = []
					for i, label in enumerate(test_true_labels):
						docnade_probs.append(test_pred_logits[i, label])
					docnade_probs = np.array(docnade_probs)

					dict_label = {int(label):[] for label in np.unique(test_true_labels)}
					for score, id, label in zip(docnade_probs, ids, test_true_labels):
						dict_label[int(label)].append([id,score])
					
					with open(os.path.join(log_dir, "task1_test_docnade_classify.txt"), "w") as f:
						for key in dict_label.keys():
							f.write(id2label[int(key)] + "\n")
							for id, score in dict_label[key]:
								f.write(id + "\t" + str(score) + "\n")

					
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
					for i, label in enumerate(test_true_labels):
						relevance_score_bm25_extra.append(bm25_extra_scores_matrix[i, int(label)])
					relevance_score_bm25_extra = np.array(relevance_score_bm25_extra)

					dict_label = {label:[] for label in np.unique(test_true_labels)}
					for score, id, label in zip(relevance_score_bm25_extra, ids, test_true_labels):
						dict_label[label].append([id,score])

					with open(os.path.join(log_dir, "task1_test_bm25extra.txt"), "w") as f:
						for key in dict_label.keys():
							f.write(id2label[int(key)] + "\n")
							for id, score in dict_label[key]:
								f.write(id + "\t" + str(score) + "\n")

					combined_relevance_score = docnade_probs + relevance_score_bm25_extra

					dict_label = {label:[] for label in np.unique(test_true_labels)}
					for score, id, label in zip(combined_relevance_score, ids, test_true_labels):
						dict_label[label].append([id,score])

					with open(os.path.join(log_dir, "task1_test_classify_with_bm25extra.txt"), "w") as f:
						for key in dict_label.keys():
							f.write(id2label[int(key)] + "\n")
							for id, score in dict_label[key]:
								f.write(id + "\t" + str(score) + "\n")
					
					#with open("./datasets/Task1_and_Task2_without_acronym_with_Task1_testdata_OOV_words/vocab_docnade.vocab", "r") as f:
					with open("./pretrained_embeddings/biggest_vocab.vocab", "r") as f:
						total_vocab = [line.strip() for line in f.readlines()]

					total_embedding_matrix = np.load('./pretrained_embeddings/fasttext_embeddings_biggest_vocab.npy')

					similarity_scores_Attention_Based_EmbSum = np.zeros((len(test_true_labels), 8), dtype=np.float32)
					
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
					for i, label in enumerate(test_true_labels):
						relevance_score_att_embsum.append(similarity_scores_Attention_Based_EmbSum[i, int(label)])
					relevance_score_att_embsum = np.array(relevance_score_att_embsum)

					#combined_relevance_score = relevance_score_svm + relevance_score_bm25_extra

					dict_label = {label:[] for label in np.unique(test_true_labels)}
					for score, id, label in zip(relevance_score_att_embsum, ids, test_true_labels):
						dict_label[label].append([id,score])

					with open(os.path.join(log_dir, "task1_test_att_based_embsum.txt"), "w") as f:
						for key in dict_label.keys():
							f.write(id2label[int(key)] + "\n")
							for id, score in dict_label[key]:
								f.write(id + "\t" + str(score) + "\n")

					combined_relevance_score_classify_embsum = relevance_score_att_embsum + docnade_probs

					dict_label = {label:[] for label in np.unique(test_true_labels)}
					for score, id, label in zip(combined_relevance_score_classify_embsum, ids, test_true_labels):
						dict_label[label].append([id,score])

					with open(os.path.join(log_dir, "task1_test_classify_att_based_embsum.txt"), "w") as f:
						for key in dict_label.keys():
							f.write(id2label[int(key)] + "\n")
							for id, score in dict_label[key]:
								f.write(id + "\t" + str(score) + "\n")

					combined_relevance_score_classify_embsum_bm25extra = relevance_score_att_embsum + docnade_probs + relevance_score_bm25_extra

					dict_label = {label:[] for label in np.unique(test_true_labels)}
					for score, id, label in zip(combined_relevance_score_classify_embsum_bm25extra, ids, test_true_labels):
						dict_label[label].append([id,score])

					with open(os.path.join(log_dir, "task1_test_classify_att_based_embsum_bm25extra.txt"), "w") as f:
						for key in dict_label.keys():
							f.write(id2label[int(key)] + "\n")
							for id, score in dict_label[key]:
								f.write(id + "\t" + str(score) + "\n")
				else:
					patience_count_ir += 1
				
				print('This val accuracy: {:.3f} (best val accuracy: {:.3f})'.format(
					val_acc,
					best_val_acc or 0.0
				))

				# logging information
				with open(os.path.join(log_dir, "training_info.txt"), "a") as f:
					f.write("Step: %i,	val accuracy: %s,	best val accuracy: %s\n" % 
							(step, val_acc, best_val_acc))

				if patience_count_ir > patience:
				#if (patience_count_ir > patience) or (step > 50):
					#final_weights = session.run("embeddings_lambda_list_unclipped:0")
					#np.save(os.path.join(log_dir, "final_sup_weights.npy"), final_weights)

					print("Early stopping criterion satisfied.")

					test_pred_labels = []
					test_pred_logits = []
					for test_y, test_x, test_seq_lengths in dataset.batches(test_filename, params.validation_bs, num_epochs=1, shuffle=True, multilabel=params.multi_label):
						test_feed_dict = {}
						test_feed_dict[model.x] = test_x
						test_feed_dict[model.y] = test_y
						test_feed_dict[model.seq_lengths] = test_seq_lengths

						if params.use_title_separately:
							test_y_title, test_x_title, test_seq_lengths_title = next(validation_title_data)

							test_feed_dict[model.x_title] = test_x_title
							test_feed_dict[model.seq_lengths_title] = test_seq_lengths_title

						pred_labels, pred_logits = session.run([model.pred_labels, model.disc_output], feed_dict=test_feed_dict)

						#test_pred_labels.append(pred_labels[0][0])
						test_pred_labels.append(pred_labels[0])
						test_pred_logits.append(pred_logits[0])

					test_pred_logits = softmax(np.array(test_pred_logits))

					test_true_labels = [int(label[0]) for label in test_labels]

					test_acc = accuracy_score(test_true_labels, test_pred_labels)

					np.save(os.path.join(log_dir, "test_pred_logits_last_epoch.npy"), np.array(test_pred_logits))
					np.save(os.path.join(log_dir, "test_pred_labels_last_epoch.npy"), np.array(test_pred_labels))
					np.save(os.path.join(log_dir, "test_true_labels_last_epoch.npy"), np.array(test_true_labels))

					docnade_probs = []
					for i, label in enumerate(test_true_labels):
						docnade_probs.append(test_pred_logits[i, label])
					docnade_probs = np.array(docnade_probs)

					dict_label = {int(label):[] for label in np.unique(test_true_labels)}
					for score, id, label in zip(docnade_probs, ids, test_true_labels):
						dict_label[int(label)].append([id,score])
					
					with open(os.path.join(log_dir, "task1_test_docnade_classify_last_epoch.txt"), "w") as f:
						for key in dict_label.keys():
							f.write(id2label[int(key)] + "\n")
							for id, score in dict_label[key]:
								f.write(id + "\t" + str(score) + "\n")

					
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
					for i, label in enumerate(test_true_labels):
						relevance_score_bm25_extra.append(bm25_extra_scores_matrix[i, int(label)])
					relevance_score_bm25_extra = np.array(relevance_score_bm25_extra)

					dict_label = {label:[] for label in np.unique(test_true_labels)}
					for score, id, label in zip(relevance_score_bm25_extra, ids, test_true_labels):
						dict_label[label].append([id,score])

					with open(os.path.join(log_dir, "task1_test_bm25extra_last_epoch.txt"), "w") as f:
						for key in dict_label.keys():
							f.write(id2label[int(key)] + "\n")
							for id, score in dict_label[key]:
								f.write(id + "\t" + str(score) + "\n")

					combined_relevance_score = docnade_probs + relevance_score_bm25_extra

					dict_label = {label:[] for label in np.unique(test_true_labels)}
					for score, id, label in zip(combined_relevance_score, ids, test_true_labels):
						dict_label[label].append([id,score])

					with open(os.path.join(log_dir, "task1_test_classify_with_bm25extra_last_epoch.txt"), "w") as f:
						for key in dict_label.keys():
							f.write(id2label[int(key)] + "\n")
							for id, score in dict_label[key]:
								f.write(id + "\t" + str(score) + "\n")
					
					#with open("./datasets/Task1_and_Task2_without_acronym_with_Task1_testdata_OOV_words/vocab_docnade.vocab", "r") as f:
					with open("./pretrained_embeddings/biggest_vocab.vocab", "r") as f:
						total_vocab = [line.strip() for line in f.readlines()]

					total_embedding_matrix = np.load('./pretrained_embeddings/fasttext_embeddings_biggest_vocab.npy')

					similarity_scores_Attention_Based_EmbSum = np.zeros((len(test_true_labels), 8), dtype=np.float32)
					
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
					for i, label in enumerate(test_true_labels):
						relevance_score_att_embsum.append(similarity_scores_Attention_Based_EmbSum[i, int(label)])
					relevance_score_att_embsum = np.array(relevance_score_att_embsum)

					#combined_relevance_score = relevance_score_svm + relevance_score_bm25_extra

					dict_label = {label:[] for label in np.unique(test_true_labels)}
					for score, id, label in zip(relevance_score_att_embsum, ids, test_true_labels):
						dict_label[label].append([id,score])

					with open(os.path.join(log_dir, "task1_test_att_based_embsum_last_epoch.txt"), "w") as f:
						for key in dict_label.keys():
							f.write(id2label[int(key)] + "\n")
							for id, score in dict_label[key]:
								f.write(id + "\t" + str(score) + "\n")

					combined_relevance_score_classify_embsum = relevance_score_att_embsum + docnade_probs

					dict_label = {label:[] for label in np.unique(test_true_labels)}
					for score, id, label in zip(combined_relevance_score_classify_embsum, ids, test_true_labels):
						dict_label[label].append([id,score])

					with open(os.path.join(log_dir, "task1_test_classify_att_based_embsum_last_epoch.txt"), "w") as f:
						for key in dict_label.keys():
							f.write(id2label[int(key)] + "\n")
							for id, score in dict_label[key]:
								f.write(id + "\t" + str(score) + "\n")

					combined_relevance_score_classify_embsum_bm25extra = relevance_score_att_embsum + docnade_probs + relevance_score_bm25_extra

					dict_label = {label:[] for label in np.unique(test_true_labels)}
					for score, id, label in zip(combined_relevance_score_classify_embsum_bm25extra, ids, test_true_labels):
						dict_label[label].append([id,score])

					with open(os.path.join(log_dir, "task1_test_classify_att_based_embsum_bm25extra_last_epoch.txt"), "w") as f:
						for key in dict_label.keys():
							f.write(id2label[int(key)] + "\n")
							for id, score in dict_label[key]:
								f.write(id + "\t" + str(score) + "\n")
					break

				#import pdb; pdb.set_trace()
				
				
				## mAP Calculation
				
				val_pred_probs = eval.softmax(np.array(val_pred_logits), axis=1)
				val_pred_probs = val_pred_probs[np.arange(len(val_pred_labels)), np.array(val_pred_labels)]

				val_mAP, _, preds_dict, probs_dict, _ = eval.evaluate_mAP(val_true_labels, val_pred_labels, val_pred_probs)

				if val_mAP > best_val_mAP:
					best_val_mAP = val_mAP
					print('saving: {}'.format(model_dir_mAP))
					saver.save(session, model_dir_mAP + '/model_mAP', global_step=1)
					patience_count_ir = 0
				else:
					patience_count_ir += 1
				
				print('This val mAP: {:.3f} (best val mAP: {:.3f})'.format(
					val_mAP,
					best_val_mAP or 0.0
				))

				# logging information
				with open(os.path.join(log_dir, "training_info.txt"), "a") as f:
					f.write("Step: %i,	val mAP: %s,	best val mAP: %s\n" % 
							(step, val_mAP, best_val_mAP))
					
				if patience_count_ir > patience:
					print("Early stopping criterion satisfied.")
					break
				
				"""
				validation_vectors = m.vectors(
					model,
					dataset.batches(
						'validation_docnade',
						params.validation_bs,
						num_epochs=1,
						shuffle=True,
						multilabel=params.multi_label
					),
					session
				)

				training_vectors = m.vectors(
					model,
					dataset.batches(
						'training_docnade',
						params.validation_bs,
						num_epochs=1,
						shuffle=True,
						multilabel=params.multi_label
					),
					session
				)

				val = eval.evaluate(
					training_vectors,
					validation_vectors,
					training_labels,
					validation_labels,
					recall=[0.02],
					num_classes=params.num_classes,
					multi_label=params.multi_label
				)[0]

				if val > best_val_IR:
					best_val_IR = val
					print('saving: {}'.format(model_dir_ir))
					saver.save(session, model_dir_ir + '/model_ir', global_step=1)
					patience_count_ir = 0
				else:
					patience_count_ir += 1
				
				print('This val IR: {:.3f} (best val IR: {:.3f})'.format(
					val,
					best_val_IR or 0.0
				))

				# logging information
				with open(os.path.join(log_dir, "training_info.txt"), "a") as f:
					f.write("Step: %i,	val IR: %s,	best val IR: %s\n" % 
							(step, val, best_val_IR))
					
				if patience_count_ir > patience:
					print("Early stopping criterion satisfied.")
					break
				"""


from gensim.models import CoherenceModel
from gensim.corpora.dictionary import Dictionary

def compute_coherence(texts, list_of_topics, top_n_word_in_each_topic_list, reload_model_dir):

	dictionary = Dictionary(texts)
	corpus = [dictionary.doc2bow(text) for text in texts]

	print('corpus len:%s' %len(corpus))
	print('dictionary:%s' %dictionary)
	# https://github.com/earthquakesan/palmetto-py
	# compute_topic_coherence: PMI and other coherence types
	# from palmettopy.palmetto import Palmetto
	# palmetto = Palmetto()

	# coherence_types = ["ca", "cp", "cv", "npmi", "uci", "umass"] # for palmetto library
	coherence_types = ["c_v"]#, 'u_mass', 'c_v', 'c_uci', 'c_npmi'] # ["c_v"] # 'u_mass', 'c_v', 'c_uci', 'c_npmi',
	avg_coh_scores_dict = {}

	best_coh_type_value_topci_indx = {}
	for top_n in top_n_word_in_each_topic_list:
		avg_coh_scores_dict[top_n]= []
		best_coh_type_value_topci_indx[top_n] = [0,  0, []] # score, topic_indx, topics words


	h_num = 0
	with open(reload_model_dir, "w") as f:
		for topic_words_all in list_of_topics:
			h_num += 1
			for top_n in top_n_word_in_each_topic_list:
				topic_words = [topic_words_all[:top_n]]
				for coh_type in coherence_types:
					try:
						print('top_n: %s Topic Num: %s \nTopic Words: %s' % (top_n, h_num, topic_words))
						f.write('top_n: %s Topic Num: %s \nTopic Words: %s\n' % (top_n, h_num, topic_words))
						# print('topic_words_top_10_abs[%s]:%s' % (h_num, topic_words_top_10_abs[h_num]))
						# PMI = palmetto.get_coherence(topic_words_top_10[h_num], coherence_type=coh_type)
						PMI = CoherenceModel(topics=topic_words, texts=texts, dictionary=dictionary, coherence=coh_type, processes=2).get_coherence()

						avg_coh_scores_dict[top_n].append(PMI)

						if PMI > best_coh_type_value_topci_indx[top_n][0]:
							best_coh_type_value_topci_indx[top_n] = [PMI, top_n, topic_words]

						print('Coh_type:%s  Topic Num:%s COH score:%s' % (coh_type, h_num, PMI))
						f.write('Coh_type:%s  Topic Num:%s COH score:%s\n' % (coh_type, h_num, PMI))

						print('--------------------------------------------------------------')
					except:
						continue
				print('========================================================================================================')

		for top_n in top_n_word_in_each_topic_list:
			print('top scores for top_%s:%s' %(top_n, best_coh_type_value_topci_indx[top_n]))
			print('-------------------------------------------------------------------')
			f.write('top scores for top_%s:%s\n' %(top_n, best_coh_type_value_topci_indx[top_n]))
			f.write('-------------------------------------------------------------------\n')

		for top_n in top_n_word_in_each_topic_list:
			print('Avg COH for top_%s topic words: %s' %(top_n, np.mean(avg_coh_scores_dict[top_n])))
			print('-------------------------------------------------------------------')
			f.write('Avg COH for top_%s topic words: %s\n' %(top_n, np.mean(avg_coh_scores_dict[top_n])))
			f.write('-------------------------------------------------------------------\n')


def get_vectors_from_matrix(matrix, batches):
	# matrix: embedding matrix of shape = [vocab_size X embedding_size]
	vecs = []
	for _, x, seq_length in batches:
		temp_vec = np.zeros((matrix.shape[1]), dtype=np.float32)
		indices = x[0, :seq_length[0]]
		for index in indices:
			temp_vec += matrix[index, :]
		vecs.append(temp_vec)
	return np.array(vecs)


from math import *
from nltk.corpus import wordnet
def square_rooted(x):
	return round(sqrt(sum([a * a for a in x])), 3)

def cosine_similarity(x, y):
	numerator = sum(a * b for a, b in zip(x, y))
	denominator = square_rooted(x) * square_rooted(y)
	return round(numerator / float(denominator), 3)


def reload_evaluation_acc_mAP(params):
	with tf.Session(config=tf.ConfigProto(
		inter_op_parallelism_threads=params['num_cores'],
		intra_op_parallelism_threads=params['num_cores'],
		gpu_options=tf.GPUOptions(allow_growth=True)
	)) as session_acc:

		dataset = data.Dataset(params['dataset'])
		log_dir = os.path.join(params['model'], 'logs')

		if not os.path.exists(log_dir):
			os.makedirs(log_dir)
			
		saver_ppl = tf.train.import_meta_graph("model/" + params['reload_model_dir'] + "model_sup/model_sup-1.meta")
		saver_ppl.restore(session_acc, tf.train.latest_checkpoint("model/" + params['reload_model_dir'] + "model_sup/"))

		graph = tf.get_default_graph()

		predicted_labels = graph.get_tensor_by_name("pred_labels:0")
		disc_logits = graph.get_tensor_by_name("disc_logits:0")

		x = graph.get_tensor_by_name("x:0")
		y = graph.get_tensor_by_name("y:0")
		seq_lengths = graph.get_tensor_by_name("seq_lengths:0")

		if params['input_type'] == "both":
			training_filename = 'training_docnade'
			validation_filename = 'validation_docnade'
			test_filename = 'test_docnade'
		elif params['input_type'] == "abstract":
			training_filename = 'training_docnade_abstracts'
			validation_filename = 'validation_docnade_abstracts'
			test_filename = 'test_docnade_abstracts'
		elif params['input_type'] == "title":
			training_filename = 'training_docnade_titles'
			validation_filename = 'validation_docnade_titles'
			test_filename = 'test_docnade_titles'

		## Classification accuracy

		#if params.use_title_separately:
		#	validation_title_data = dataset.batches(validation_title_filename, params.validation_bs, num_epochs=1, shuffle=True, multilabel=params.multi_label)
		#	validation_filename = 'validation_docnade_abstracts'
		
		val_true_labels = []
		val_pred_labels = []
		val_pred_logits = []
		for val_y, val_x, val_seq_lengths in dataset.batches(validation_filename, 1, num_epochs=1, shuffle=False, multilabel=params['multi_label']):
			val_feed_dict = {}
			val_feed_dict[x] = val_x
			val_feed_dict[y] = val_y
			val_feed_dict[seq_lengths] = val_seq_lengths

			#if params.use_title_separately:
			#	val_y_title, val_x_title, val_seq_lengths_title = next(validation_title_data)

			#	val_feed_dict[model.x_title] = val_x_title
			#	val_feed_dict[model.seq_lengths_title] = val_seq_lengths_title

			pred_labels, pred_logits = session_acc.run([predicted_labels, disc_logits], feed_dict=val_feed_dict)

			#val_pred_labels.append(pred_labels[0][0])
			val_pred_labels.append(pred_labels[0])
			val_pred_logits.append(pred_logits[0])
			val_true_labels.append(int(val_y))

		#val_true_labels = [int(label[0]) for label in validation_labels]

		val_acc = accuracy_score(val_true_labels, val_pred_labels)
		
		print('This val accuracy: {:.3f}'.format(val_acc))

		with open(os.path.join(log_dir, "reload_info_acc.txt"), "w") as f:
			f.write("val accuracy: %s\n" % (val_acc))
		
		## Using classification probability for relevance ranking and mAP calculation
		
		val_pred_probs = eval.softmax(np.array(val_pred_logits), axis=1)
		val_pred_probs_temp = val_pred_probs[np.arange(len(val_pred_labels)), np.array(val_pred_labels)]

		val_mAP, val_AP_dict, preds_dict, probs_dict, indices_dict = eval.evaluate_mAP(val_true_labels, val_pred_labels, val_pred_probs_temp)
		
		print('This val mAP: {:.3f}'.format(val_mAP))

		# logging information
		with open(os.path.join(log_dir, "reload_info_acc.txt"), "a") as f:
			f.write("val mAP: %s\n" % (val_mAP))

		with open(os.path.join(log_dir, "reload_info_clusters.txt"), "w") as f:
			for label in preds_dict.keys():
				preds = preds_dict[label]
				probs = probs_dict[label]
				preds_indices = indices_dict[label]

				sorted_indices = np.argsort(probs)[::-1]
				sorted_preds = np.array(preds)[sorted_indices]
				sorted_probs = np.array(probs)[sorted_indices]
				sorted_preds_indices = np.array(preds_indices)[sorted_indices]

				f.write("Cluster " + str(label) + "\n\n")
				f.write("Average precision: " + str(val_AP_dict[label]) + "\n")
				f.write("Predicted_labels: " + " ".join([str(l) for l in sorted_preds]) + "\n")
				f.write("Predicted_probs:  " + " ".join([str(l) for l in sorted_probs]) + "\n")
				f.write("Predicted_indices:  " + " ".join([str(l) for l in sorted_preds_indices]) + "\n")
				f.write("\n\n")
			f.write("\n\n=================================================================================\n\n")
		
		#import pdb; pdb.set_trace()
		
		# misclassified
		temp = (np.array(val_pred_labels) == np.array(val_true_labels))

		#val_pred_probs_full = eval.softmax(np.array(val_pred_logits), axis=1)

		indices = [index for index, val in enumerate(temp) if not val]

		with open(os.path.join(log_dir, "misclassified.txt"), "a") as f:
			for index in indices:
				#f.write("Val doc #" + str(index+1) + "\n")
				f.write("Val doc #" + str(index+1) + "\n")
				f.write("True label: " + str(val_true_labels[index]) + "\n")
				f.write("Pred label: " + str(val_pred_labels[index]) + "\n")
				#f.write("Prediction probs: " + " ".join([str(prob) for prob in val_pred_probs_full[index, :]]) + "\n")
				f.write("Prediction probs: " + " ".join([str(prob) for prob in val_pred_probs[index, :]]) + "\n")
				f.write("\n\n")
		
		## Using BM25 score for relevance ranking and mAP calculation

		#bm25 = BM25.BM25(params['dataset'] + "/val_docs.txt", delimiter=' ')
		bm25 = BM25.BM25(params['dataset'] + "/validation.csv", delimiter=' ')

		bm25_vocab = [word for word in bm25.dictionary.itervalues()]
		
		#with open(params['dataset'] + "/val_docs.txt", "r") as f:
		#	docs = [line.strip().split("\t")[1].lower().strip() for line in f.readlines()]

		with open(params['dataset'] + "/validation.csv", "r") as f:
			file_reader = csv.reader(f, delimiter=',')
			docs = [line[1].strip() for line in file_reader]

		queries = []
		with open(params['dataset'] + "/labels.txt", "r") as f:
			for line in f.readlines():
				label_tokens = line.lower().strip().split("_")
				label_tokens = [token for token in label_tokens if token]
				query = " ".join(label_tokens)
				queries.append(query)
				#queries.append(query_defs[query])
		
		def get_bm25_ids(tokens):
			ids = []
			for token in tokens:
				try:
					ids.append(bm25.dictionary.token2id[token])
				except KeyError:
					pass
			return ids

		bm25_scores_list = []
		for query in queries:
			query = query.split()
			scores = bm25.BM25Score(query)
			bm25_scores_list.append(scores)
		
		bm25_scores_matrix = np.stack(bm25_scores_list, axis=1)
		if params['include_sup_probs']:
			bm25_scores_matrix = np.add(eval.softmax(bm25_scores_matrix, axis=1), val_pred_probs)

		relevance_score_bm25 = []
		for i, label in enumerate(val_pred_labels):
			relevance_score_bm25.append(bm25_scores_matrix[i, label])
		relevance_score_bm25 = np.array(relevance_score_bm25)

		val_mAP, val_AP_dict, preds_dict, probs_dict, indices_dict = eval.evaluate_mAP(val_true_labels, val_pred_labels, relevance_score_bm25)
		
		print('This val mAP (with BM25): {:.3f}'.format(val_mAP))

		# logging information
		with open(os.path.join(log_dir, "reload_info_acc.txt"), "a") as f:
			f.write("val mAP (with BM25): %s\n" % (val_mAP))

		with open(os.path.join(log_dir, "reload_info_clusters_with_BM25.txt"), "w") as f:
			for label in preds_dict.keys():
				preds = preds_dict[label]
				probs = probs_dict[label]
				preds_indices = indices_dict[label]

				sorted_indices = np.argsort(probs)[::-1]
				sorted_preds = np.array(preds)[sorted_indices]
				sorted_probs = np.array(probs)[sorted_indices]
				sorted_preds_indices = np.array(preds_indices)[sorted_indices]

				f.write("Cluster " + str(label) + "\n\n")
				f.write("Average precision: " + str(val_AP_dict[label]) + "\n")
				f.write("Predicted_labels: " + " ".join([str(l) for l in sorted_preds]) + "\n")
				f.write("Predicted_probs:  " + " ".join([str(l) for l in sorted_probs]) + "\n")
				f.write("Predicted_indices:  " + " ".join([str(l) for l in sorted_preds_indices]) + "\n")
				f.write("\n\n")
			f.write("\n\n=================================================================================\n\n")
		
		## Using BM25 Extra for relevance ranking and mAP calculation

		bm25_extra_scores_list = []
		for query in queries:
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
		if params['include_sup_probs']:
			bm25_extra_scores_matrix = np.add(eval.softmax(bm25_extra_scores_matrix, axis=1), val_pred_probs)

		relevance_score_bm25_extra = []
		for i, label in enumerate(val_pred_labels):
			relevance_score_bm25_extra.append(bm25_extra_scores_matrix[i, label])
		relevance_score_bm25_extra = np.array(relevance_score_bm25_extra)

		val_mAP, val_AP_dict, preds_dict, probs_dict, indices_dict = eval.evaluate_mAP(val_true_labels, val_pred_labels, relevance_score_bm25_extra)
		
		print('This val mAP (with BM25 Extra): {:.3f}'.format(val_mAP))

		# logging information
		with open(os.path.join(log_dir, "reload_info_acc.txt"), "a") as f:
			f.write("val mAP (with BM25 Extra): %s\n" % (val_mAP))

		with open(os.path.join(log_dir, "reload_info_clusters_with_BM25_extra.txt"), "w") as f:
			for label in preds_dict.keys():
				preds = preds_dict[label]
				probs = probs_dict[label]
				preds_indices = indices_dict[label]

				sorted_indices = np.argsort(probs)[::-1]
				sorted_preds = np.array(preds)[sorted_indices]
				sorted_probs = np.array(probs)[sorted_indices]
				sorted_preds_indices = np.array(preds_indices)[sorted_indices]

				f.write("Cluster " + str(label) + "\n\n")
				f.write("Average precision: " + str(val_AP_dict[label]) + "\n")
				f.write("Predicted_labels: " + " ".join([str(l) for l in sorted_preds]) + "\n")
				f.write("Predicted_probs:  " + " ".join([str(l) for l in sorted_probs]) + "\n")
				f.write("Predicted_indices:  " + " ".join([str(l) for l in sorted_preds_indices]) + "\n")
				f.write("\n\n")
			f.write("\n\n=================================================================================\n\n")

		## Using EmbSum for relevance ranking and mAP calculation

		with codecs.open(params['docnadeVocab'], "r", encoding='utf-8', errors='ignore') as f:
			docnade_vocab = [line.strip() for line in f.readlines()]

		with codecs.open("./datasets/Task1_Augmented_Def_new/vocab_docnade.vocab", "r", encoding='utf-8', errors='ignore') as f:
			docnade_vocab_large = [line.strip() for line in f.readlines()]

		query_words = []
		for query in queries:
			query_words.extend(query.strip().split())
		query_words = np.unique(query_words)
		query_words_list = query_words.tolist()

		prior_embedding_matrices = []
		query_embedding_matrices = []

		if params['use_bio_prior']:
			bio_embeddings = np.load('./pretrained_embeddings/' + params['bioemb_path'])
			prior_embedding_matrices.append(bio_embeddings)

			bio_embeddings_large = np.load('./pretrained_embeddings/bionlp_embeddings_task1_without_acronyms_without_stopwords_augmented_def_new.npy')
			query_bio_embeddings = np.zeros((len(query_words), bio_embeddings_large.shape[1]), dtype=np.float32)
			for i, word in enumerate(query_words):
				query_bio_embeddings[i, :] = bio_embeddings_large[int(docnade_vocab_large.index(word.strip())), :]
			query_embedding_matrices.append(query_bio_embeddings)


		if params['use_fasttext_prior']:
			fasttext_embeddings = np.load('./pretrained_embeddings/' + params['fttemb_path'])
			prior_embedding_matrices.append(fasttext_embeddings)

			fasttext_embeddings_large = np.load('./pretrained_embeddings/fasttext_embeddings_task1_without_acronyms_without_stopwords_augmented_def_new.npy')
			query_fasttext_embeddings = np.zeros((len(query_words), fasttext_embeddings_large.shape[1]), dtype=np.float32)
			for i, word in enumerate(query_words):
				#import pdb; pdb.set_trace()
				query_fasttext_embeddings[i, :] = fasttext_embeddings_large[int(docnade_vocab_large.index(word.strip())), :]
			query_embedding_matrices.append(query_fasttext_embeddings)

		if params['use_BOW_repesentation']:
			#BOW_representations = np.eye(len(docnade_vocab), dtype=np.float32)
			BOW_representations = np.eye(len(docnade_vocab_large), dtype=np.float32)
			#prior_embedding_matrices.append(BOW_representations)
			BOW_representations_docs = BOW_representations[np.array([int(docnade_vocab_large.index(word)) for word in docnade_vocab]), :]
			BOW_representations_queries = BOW_representations[np.array([int(docnade_vocab_large.index(word)) for word in query_words]), :]
			prior_embedding_matrices.append(BOW_representations_docs)
			query_embedding_matrices.append(BOW_representations_queries)

		if params['use_DocNADE_W']:
			DocNADE_W = session_acc.run("embedding:0")
			prior_embedding_matrices.append(DocNADE_W)

			query_W_embeddings = np.zeros((len(query_words), DocNADE_W.shape[1]), dtype=np.float32)
			for i, word in enumerate(query_words):
				if word in docnade_vocab:
					query_W_embeddings[i, :] = DocNADE_W[int(docnade_vocab.index(word.strip())), :]
			query_embedding_matrices.append(query_W_embeddings)
		
		#import pdb; pdb.set_trace()

		docnade_embedding_matrix = np.concatenate(prior_embedding_matrices, axis=1)
		query_embedding_matrix = np.concatenate(query_embedding_matrices, axis=1)

		#import pdb; pdb.set_trace()

		query_vecs = []
		for query in queries:
			tokens = [int(query_words_list.index(word)) for word in query.strip().split()]
			Embs = query_embedding_matrix[np.array(tokens), :]
			EmbSum = np.sum(Embs, axis=0)
			query_vecs.append(EmbSum)

		#import pdb; pdb.set_trace()

		if not params['split_abstract_title']:
			validation_vecs = []
			with open(params['dataset'] + "/validation_docnade.csv", "r") as f:
				file_reader = csv.reader(f, delimiter=",")
				for row in file_reader:
					tokens = [int(index) for index in row[1].strip().split()]
					Embs = docnade_embedding_matrix[np.array(tokens), :]
					EmbSum = np.sum(Embs, axis=0)
					validation_vecs.append(EmbSum)

			similarity_scores = pw.cosine_similarity(np.array(validation_vecs), np.array(query_vecs))
		else:
			validation_abstract_vecs = []
			with open(params['dataset'] + "/validation_docnade_abstracts.csv", "r") as f:
				file_reader = csv.reader(f, delimiter=",")
				for row in file_reader:
					tokens = [int(index) for index in row[1].strip().split()]
					Embs = docnade_embedding_matrix[np.array(tokens), :]
					EmbSum = np.sum(Embs, axis=0)
					validation_abstract_vecs.append(EmbSum)

			similarity_abstract_scores = pw.cosine_similarity(np.array(validation_abstract_vecs), np.array(query_vecs))

			validation_title_vecs = []
			with open(params['dataset'] + "/validation_docnade_titles.csv", "r") as f:
				file_reader = csv.reader(f, delimiter=",")
				for row in file_reader:
					tokens = [int(index) for index in row[1].strip().split()]
					Embs = docnade_embedding_matrix[np.array(tokens), :]
					EmbSum = np.sum(Embs, axis=0)
					validation_title_vecs.append(EmbSum)

			similarity_title_scores = pw.cosine_similarity(np.array(validation_title_vecs), np.array(query_vecs))

			similarity_scores = np.add(similarity_abstract_scores, similarity_title_scores)

			validation_vecs = np.concatenate([validation_abstract_vecs, validation_title_vecs], axis=1)

		#similarity_scores = pw.cosine_similarity(np.array(validation_vecs), np.array(query_vecs))
		if params['include_sup_probs']:
			similarity_scores = np.add(eval.softmax(similarity_scores, axis=1), val_pred_probs)

		relevance_score = []
		for i, label in enumerate(val_pred_labels):
			relevance_score.append(similarity_scores[i, label])
		relevance_score = np.array(relevance_score)

		val_mAP, val_AP_dict, preds_dict, probs_dict, indices_dict = eval.evaluate_mAP(val_true_labels, val_pred_labels, relevance_score)
		
		print('This val mAP (with EmbSum): {:.3f}'.format(val_mAP))

		# logging information
		with open(os.path.join(log_dir, "reload_info_acc.txt"), "a") as f:
			f.write("val mAP (with EmbSum): %s\n" % (val_mAP))

		with open(os.path.join(log_dir, "reload_info_clusters_with_EmbSum.txt"), "w") as f:
			for label in preds_dict.keys():
				preds = preds_dict[label]
				probs = probs_dict[label]
				preds_indices = indices_dict[label]

				sorted_indices = np.argsort(probs)[::-1]
				sorted_preds = np.array(preds)[sorted_indices]
				sorted_probs = np.array(probs)[sorted_indices]
				sorted_preds_indices = np.array(preds_indices)[sorted_indices]

				f.write("Cluster " + str(label) + "\n\n")
				f.write("Average precision: " + str(val_AP_dict[label]) + "\n")
				f.write("Predicted_labels: " + " ".join([str(l) for l in sorted_preds]) + "\n")
				f.write("Predicted_probs:  " + " ".join([str(l) for l in sorted_probs]) + "\n")
				f.write("Predicted_indices:  " + " ".join([str(l) for l in sorted_preds_indices]) + "\n")
				f.write("\n\n")
			f.write("\n\n=================================================================================\n\n")
					
		## Using BM25 + EmbSum for relevance ranking and mAP calculation

		similarity_scores_combined = np.add(eval.softmax(similarity_scores, axis=1), eval.softmax(bm25_scores_matrix, axis=1))
		if params['include_sup_probs']:
			similarity_scores_combined = np.add(eval.softmax(similarity_scores_combined, axis=1), val_pred_probs)

		relevance_score_comb = []
		for i, label in enumerate(val_pred_labels):
			relevance_score_comb.append(similarity_scores_combined[i, label])
		relevance_score_comb = np.array(relevance_score_comb)

		val_mAP, val_AP_dict, preds_dict, probs_dict, indices_dict = eval.evaluate_mAP(val_true_labels, val_pred_labels, relevance_score_comb)
		
		print('This val mAP (with BM25 and EmbSum): {:.3f}'.format(val_mAP))

		# logging information
		with open(os.path.join(log_dir, "reload_info_acc.txt"), "a") as f:
			f.write("val mAP (with BM25 and EmbSum): %s\n" % (val_mAP))

		with open(os.path.join(log_dir, "reload_info_clusters_with_BM25_and_EmbSum.txt"), "w") as f:
			for label in preds_dict.keys():
				preds = preds_dict[label]
				probs = probs_dict[label]
				preds_indices = indices_dict[label]

				sorted_indices = np.argsort(probs)[::-1]
				sorted_preds = np.array(preds)[sorted_indices]
				sorted_probs = np.array(probs)[sorted_indices]
				sorted_preds_indices = np.array(preds_indices)[sorted_indices]

				f.write("Cluster " + str(label) + "\n\n")
				f.write("Average precision: " + str(val_AP_dict[label]) + "\n")
				f.write("Predicted_labels: " + " ".join([str(l) for l in sorted_preds]) + "\n")
				f.write("Predicted_probs:  " + " ".join([str(l) for l in sorted_probs]) + "\n")
				f.write("Predicted_indices:  " + " ".join([str(l) for l in sorted_preds_indices]) + "\n")
				f.write("\n\n")
			f.write("\n\n=================================================================================\n\n")
		
		## Using Attention based EmbSum for relevance ranking and mAP calculation

		similarity_scores_Attention_Based_EmbSum = np.zeros((len(validation_vecs), len(query_vecs)), dtype=np.float32)

		if not params['split_abstract_title']:
			with open(params['dataset'] + "/validation_docnade.csv", "r") as f:
				file_reader = csv.reader(f, delimiter=",")
				for j, row in enumerate(file_reader):
					tokens = [int(index) for index in row[1].strip().split()]
					Embs = docnade_embedding_matrix[np.array(tokens), :]
					for i, query in enumerate(queries):
						query_tokens = query.split()
						EmbSum_attns = []
						query_vecs_attns = []
						for qword in query_tokens:
							query_vector = query_embedding_matrix[query_words_list.index(qword), :]
							query_vector = np.expand_dims(query_vector, axis=0)
							query_attentions = pw.cosine_similarity(query_vector, Embs)
							query_attentions = eval.softmax(query_attentions)
							EmbSum_attentions = np.dot(query_attentions, Embs)
							EmbSum_attns.append(EmbSum_attentions)
							query_vecs_attns.append(query_vector)
						if params['attention_EmbSum_type'] == "sum":
							EmbSum = np.sum(EmbSum_attns, axis=0)
							query_EmbSum_vector = np.expand_dims(query_vecs[i], axis=0)
							similarity_score = pw.cosine_similarity(query_EmbSum_vector, EmbSum)
							similarity_scores_Attention_Based_EmbSum[j, i] = similarity_score[0][0]
						elif params['attention_EmbSum_type'] == "concat":
							EmbSum = np.concatenate(EmbSum_attns, axis=1)
							query_EmbSum_vector = np.concatenate(query_vecs_attns, axis=1)
							similarity_score = pw.cosine_similarity(query_EmbSum_vector, EmbSum)
							similarity_scores_Attention_Based_EmbSum[j, i] = similarity_score[0][0]
						elif params['attention_EmbSum_type'] == "max":
							max_similarity = -1.1
							for q_vec, d_vec in zip(query_vecs_attns, EmbSum_attns):
								similarity_score = pw.cosine_similarity(q_vec, d_vec)
								if similarity_score[0][0] > max_similarity:
									max_similarity = similarity_score[0][0]
							similarity_scores_Attention_Based_EmbSum[j, i] = max_similarity
		else:
			Embs_titles = []
			with open(params['dataset'] + "/validation_docnade_titles.csv", "r") as f:
				file_reader = csv.reader(f, delimiter=",")
				for j, row in enumerate(file_reader):
					tokens = [int(index) for index in row[1].strip().split()]
					Embs = docnade_embedding_matrix[np.array(tokens), :]
					Embs_titles.append(Embs)
			
			with open(params['dataset'] + "/validation_docnade_abstracts.csv", "r") as f:
				file_reader = csv.reader(f, delimiter=",")
				for j, row in enumerate(file_reader):
					tokens = [int(index) for index in row[1].strip().split()]
					Embs = docnade_embedding_matrix[np.array(tokens), :]
					Embs_title = Embs_titles[j]
					for i, query in enumerate(queries):
						query_tokens = query.split()
						EmbSum_attns = []
						EmbSum_title_attns = []
						query_vecs_attns = []
						for qword in query_tokens:
							query_vector = query_embedding_matrix[query_words_list.index(qword), :]
							query_vector = np.expand_dims(query_vector, axis=0)
							query_attentions = pw.cosine_similarity(query_vector, Embs)
							query_attentions = eval.softmax(query_attentions)
							EmbSum_attentions = np.dot(query_attentions, Embs)
							query_attentions_title = pw.cosine_similarity(query_vector, Embs_title)
							query_attentions_title = eval.softmax(query_attentions_title)
							EmbSum_attentions_title = np.dot(query_attentions_title, Embs_title)
							EmbSum_attns.append(EmbSum_attentions)
							EmbSum_title_attns.append(EmbSum_attentions_title)
							query_vecs_attns.append(query_vector)
						
						if params['attention_EmbSum_type'] == "sum":
							EmbSum = np.sum(EmbSum_attns, axis=0)
							query_EmbSum_vector = np.expand_dims(query_vecs[i], axis=0)
							similarity_score = pw.cosine_similarity(query_EmbSum_vector, EmbSum)
							similarity_scores_Attention_Based_EmbSum[j, i] += similarity_score[0][0]
						elif params['attention_EmbSum_type'] == "concat":
							EmbSum = np.concatenate(EmbSum_attns, axis=1)
							query_EmbSum_vector = np.concatenate(query_vecs_attns, axis=1)
							similarity_score = pw.cosine_similarity(query_EmbSum_vector, EmbSum)
							similarity_scores_Attention_Based_EmbSum[j, i] += similarity_score[0][0]
						elif params['attention_EmbSum_type'] == "max":
							max_similarity = -1.1
							for q_vec, d_vec in zip(query_vecs_attns, EmbSum_attns):
								similarity_score = pw.cosine_similarity(q_vec, d_vec)
								if similarity_score[0][0] > max_similarity:
									max_similarity = similarity_score[0][0]
							similarity_scores_Attention_Based_EmbSum[j, i] += max_similarity
						
						if params['attention_EmbSum_type'] == "sum":
							EmbSum = np.sum(EmbSum_title_attns, axis=0)
							query_EmbSum_vector = np.expand_dims(query_vecs[i], axis=0)
							similarity_score = pw.cosine_similarity(query_EmbSum_vector, EmbSum)
							similarity_scores_Attention_Based_EmbSum[j, i] += similarity_score[0][0]
						elif params['attention_EmbSum_type'] == "concat":
							EmbSum = np.concatenate(EmbSum_title_attns, axis=1)
							query_EmbSum_vector = np.concatenate(query_vecs_attns, axis=1)
							similarity_score = pw.cosine_similarity(query_EmbSum_vector, EmbSum)
							similarity_scores_Attention_Based_EmbSum[j, i] += similarity_score[0][0]
						elif params['attention_EmbSum_type'] == "max":
							max_similarity = -1.1
							for q_vec, d_vec in zip(query_vecs_attns, EmbSum_title_attns):
								similarity_score = pw.cosine_similarity(q_vec, d_vec)
								if similarity_score[0][0] > max_similarity:
									max_similarity = similarity_score[0][0]
							similarity_scores_Attention_Based_EmbSum[j, i] += max_similarity

		if params['include_sup_probs']:
			similarity_scores_Attention_Based_EmbSum = np.add(eval.softmax(similarity_scores_Attention_Based_EmbSum, axis=1), val_pred_probs)
		
		relevance_score_Attention_Based_EmbSum = []
		for i, label in enumerate(val_pred_labels):
			relevance_score_Attention_Based_EmbSum.append(similarity_scores_Attention_Based_EmbSum[i, label])
		relevance_score_Attention_Based_EmbSum = np.array(relevance_score_Attention_Based_EmbSum)

		val_mAP, val_AP_dict, preds_dict, probs_dict, indices_dict = eval.evaluate_mAP(val_true_labels, val_pred_labels, relevance_score_Attention_Based_EmbSum)
		
		print('This val mAP (with attention based EmbSum): {:.3f}'.format(val_mAP))

		# logging information
		with open(os.path.join(log_dir, "reload_info_acc.txt"), "a") as f:
			f.write("val mAP (with attention based EmbSum): %s\n" % (val_mAP))

		with open(os.path.join(log_dir, "reload_info_clusters_with_attention_based_EmbSum.txt"), "w") as f:
			for label in preds_dict.keys():
				preds = preds_dict[label]
				probs = probs_dict[label]
				preds_indices = indices_dict[label]

				sorted_indices = np.argsort(probs)[::-1]
				sorted_preds = np.array(preds)[sorted_indices]
				sorted_probs = np.array(probs)[sorted_indices]
				sorted_preds_indices = np.array(preds_indices)[sorted_indices]

				f.write("Cluster " + str(label) + "\n\n")
				f.write("Average precision: " + str(val_AP_dict[label]) + "\n")
				f.write("Predicted_labels: " + " ".join([str(l) for l in sorted_preds]) + "\n")
				f.write("Predicted_probs:  " + " ".join([str(l) for l in sorted_probs]) + "\n")
				f.write("Predicted_indices:  " + " ".join([str(l) for l in sorted_preds_indices]) + "\n")
				f.write("\n\n")
			f.write("\n\n=================================================================================\n\n")
		
		## Using BM25 + Attentione based EmbSum for relevance ranking and mAP calculation

		similarity_scores_Attention_Based_EmbSum_combined = np.add(eval.softmax(similarity_scores_Attention_Based_EmbSum, axis=1), eval.softmax(bm25_scores_matrix, axis=1))
		if params['include_sup_probs']:
			similarity_scores_Attention_Based_EmbSum_combined = np.add(eval.softmax(similarity_scores_Attention_Based_EmbSum_combined, axis=1), val_pred_probs)

		relevance_score_Attention_Based_EmbSum_combined = []
		for i, label in enumerate(val_pred_labels):
			relevance_score_Attention_Based_EmbSum_combined.append(similarity_scores_Attention_Based_EmbSum_combined[i, label])
		relevance_score_Attention_Based_EmbSum_combined = np.array(relevance_score_Attention_Based_EmbSum_combined)

		val_mAP, val_AP_dict, preds_dict, probs_dict, indices_dict = eval.evaluate_mAP(val_true_labels, val_pred_labels, relevance_score_Attention_Based_EmbSum_combined)
		
		print('This val mAP (with BM25 and attention based EmbSum): {:.3f}'.format(val_mAP))

		# logging information
		with open(os.path.join(log_dir, "reload_info_acc.txt"), "a") as f:
			f.write("val mAP (with BM25 and attention based EmbSum): %s\n" % (val_mAP))

		with open(os.path.join(log_dir, "reload_info_clusters_with_BM25_and_attention_based_EmbSum.txt"), "w") as f:
			for label in preds_dict.keys():
				preds = preds_dict[label]
				probs = probs_dict[label]
				preds_indices = indices_dict[label]

				sorted_indices = np.argsort(probs)[::-1]
				sorted_preds = np.array(preds)[sorted_indices]
				sorted_probs = np.array(probs)[sorted_indices]
				sorted_preds_indices = np.array(preds_indices)[sorted_indices]

				f.write("Cluster " + str(label) + "\n\n")
				f.write("Average precision: " + str(val_AP_dict[label]) + "\n")
				f.write("Predicted_labels: " + " ".join([str(l) for l in sorted_preds]) + "\n")
				f.write("Predicted_probs:  " + " ".join([str(l) for l in sorted_probs]) + "\n")
				f.write("Predicted_indices:  " + " ".join([str(l) for l in sorted_preds_indices]) + "\n")
				f.write("\n\n")
			f.write("\n\n=================================================================================\n\n")

		## Using BM25 with embeddings for relevance ranking and mAP calculation
	
		bm25_with_emb_scores_list = []
		for query in queries:
			query = query.split()
			scores = bm25.BM25Score(Query=query, 
									embedding_matrix=docnade_embedding_matrix, 
									embedding_vocab=docnade_vocab, 
									query_matrix=query_embedding_matrix, 
									query_vocab=query_words_list, 
									sim_threshold=0.20)
			bm25_with_emb_scores_list.append(scores)
		
		bm25_with_emb_scores_matrix = np.stack(bm25_with_emb_scores_list, axis=1)
		if params['include_sup_probs']:
			bm25_with_emb_scores_matrix = np.add(eval.softmax(bm25_with_emb_scores_matrix, axis=1), val_pred_probs)

		relevance_score_bm25_with_emb = []
		for i, label in enumerate(val_pred_labels):
			relevance_score_bm25_with_emb.append(bm25_with_emb_scores_matrix[i, label])
		relevance_score_bm25_with_emb = np.array(relevance_score_bm25_with_emb)

		val_mAP, val_AP_dict, preds_dict, probs_dict, indices_dict = eval.evaluate_mAP(val_true_labels, val_pred_labels, relevance_score_bm25_with_emb)
		
		print('This val mAP (with BM25 using embeddings): {:.3f}'.format(val_mAP))

		# logging information
		with open(os.path.join(log_dir, "reload_info_acc.txt"), "a") as f:
			f.write("val mAP (with BM25 using embeddings): %s\n" % (val_mAP))

		with open(os.path.join(log_dir, "reload_info_clusters_with_BM25_using_embeddings.txt"), "w") as f:
			for label in preds_dict.keys():
				preds = preds_dict[label]
				probs = probs_dict[label]
				preds_indices = indices_dict[label]

				sorted_indices = np.argsort(probs)[::-1]
				sorted_preds = np.array(preds)[sorted_indices]
				sorted_probs = np.array(probs)[sorted_indices]
				sorted_preds_indices = np.array(preds_indices)[sorted_indices]

				f.write("Cluster " + str(label) + "\n\n")
				f.write("Average precision: " + str(val_AP_dict[label]) + "\n")
				f.write("Predicted_labels: " + " ".join([str(l) for l in sorted_preds]) + "\n")
				f.write("Predicted_probs:  " + " ".join([str(l) for l in sorted_probs]) + "\n")
				f.write("Predicted_indices:  " + " ".join([str(l) for l in sorted_preds_indices]) + "\n")
				f.write("\n\n")
			f.write("\n\n=================================================================================\n\n")

		## Using BM25 using embeddings + EmbSum for relevance ranking and mAP calculation

		similarity_scores_combined = np.add(eval.softmax(similarity_scores, axis=1), eval.softmax(bm25_with_emb_scores_matrix, axis=1))
		if params['include_sup_probs']:
			similarity_scores_combined = np.add(eval.softmax(similarity_scores_combined, axis=1), val_pred_probs)

		relevance_score_comb = []
		for i, label in enumerate(val_pred_labels):
			relevance_score_comb.append(similarity_scores_combined[i, label])
		relevance_score_comb = np.array(relevance_score_comb)

		val_mAP, val_AP_dict, preds_dict, probs_dict, indices_dict = eval.evaluate_mAP(val_true_labels, val_pred_labels, relevance_score_comb)
		
		print('This val mAP (with BM25 and EmbSum): {:.3f}'.format(val_mAP))

		# logging information
		with open(os.path.join(log_dir, "reload_info_acc.txt"), "a") as f:
			f.write("val mAP (with BM25 and EmbSum): %s\n" % (val_mAP))

		with open(os.path.join(log_dir, "reload_info_clusters_with_BM25_using_embeddings_and EmbSum.txt"), "w") as f:
			for label in preds_dict.keys():
				preds = preds_dict[label]
				probs = probs_dict[label]
				preds_indices = indices_dict[label]

				sorted_indices = np.argsort(probs)[::-1]
				sorted_preds = np.array(preds)[sorted_indices]
				sorted_probs = np.array(probs)[sorted_indices]
				sorted_preds_indices = np.array(preds_indices)[sorted_indices]

				f.write("Cluster " + str(label) + "\n\n")
				f.write("Average precision: " + str(val_AP_dict[label]) + "\n")
				f.write("Predicted_labels: " + " ".join([str(l) for l in sorted_preds]) + "\n")
				f.write("Predicted_probs:  " + " ".join([str(l) for l in sorted_probs]) + "\n")
				f.write("Predicted_indices:  " + " ".join([str(l) for l in sorted_preds_indices]) + "\n")
				f.write("\n\n")
			f.write("\n\n=================================================================================\n\n")
			
		## Using BM25 using embeddings + Attention based EmbSum for relevance ranking and mAP calculation

		similarity_scores_Attention_Based_EmbSum_combined = np.add(eval.softmax(similarity_scores_Attention_Based_EmbSum, axis=1), eval.softmax(bm25_with_emb_scores_matrix, axis=1))
		if params['include_sup_probs']:
			similarity_scores_Attention_Based_EmbSum_combined = np.add(eval.softmax(similarity_scores_Attention_Based_EmbSum_combined, axis=1), val_pred_probs)

		relevance_score_Attention_Based_EmbSum_combined = []
		for i, label in enumerate(val_pred_labels):
			relevance_score_Attention_Based_EmbSum_combined.append(similarity_scores_Attention_Based_EmbSum_combined[i, label])
		relevance_score_Attention_Based_EmbSum_combined = np.array(relevance_score_Attention_Based_EmbSum_combined)

		val_mAP, val_AP_dict, preds_dict, probs_dict, indices_dict = eval.evaluate_mAP(val_true_labels, val_pred_labels, relevance_score_Attention_Based_EmbSum_combined)
		
		print('This val mAP (with BM25 and attention based EmbSum): {:.3f}'.format(val_mAP))

		# logging information
		with open(os.path.join(log_dir, "reload_info_acc.txt"), "a") as f:
			f.write("val mAP (with BM25 and attention based EmbSum): %s\n" % (val_mAP))

		with open(os.path.join(log_dir, "reload_info_clusters_with_BM25_using_embeddings_and_attention_based_EmbSum.txt"), "w") as f:
			for label in preds_dict.keys():
				preds = preds_dict[label]
				probs = probs_dict[label]
				preds_indices = indices_dict[label]

				sorted_indices = np.argsort(probs)[::-1]
				sorted_preds = np.array(preds)[sorted_indices]
				sorted_probs = np.array(probs)[sorted_indices]
				sorted_preds_indices = np.array(preds_indices)[sorted_indices]

				f.write("Cluster " + str(label) + "\n\n")
				f.write("Average precision: " + str(val_AP_dict[label]) + "\n")
				f.write("Predicted_labels: " + " ".join([str(l) for l in sorted_preds]) + "\n")
				f.write("Predicted_probs:  " + " ".join([str(l) for l in sorted_probs]) + "\n")
				f.write("Predicted_indices:  " + " ".join([str(l) for l in sorted_preds_indices]) + "\n")
				f.write("\n\n")
			f.write("\n\n=================================================================================\n\n")

		return docnade_embedding_matrix, query_embedding_matrix, query_vecs, queries, query_words_list


def reload_evaluation_f1(params, training_vectors, test_vectors, W_matrix, suffix=""):

	### Classification - F1

	dataset = data.Dataset(params['dataset'])
	log_dir = os.path.join(params['model'], 'logs')

	c_list = [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 3.0, 5.0, 10.0, 100.0, 500.0, 1000.0, 10000.0]
	#c_list = [1.0]

	test_acc = []
	test_f1 = []

	y_train = np.array(
		[y for y, _ in dataset.rows('training_docnade', num_epochs=1)]
	)
	y_test = np.array(
		[y for y, _ in dataset.rows('test_docnade', num_epochs=1)]
	)

	if not params['multi_label']:
		train_data = (training_vectors, np.array(y_train, dtype=np.int32))
		test_data = (test_vectors, np.array(y_test, dtype=np.int32))

		test_acc, test_f1 = eval.perform_classification_test(train_data, test_data, c_list)
		
		with open(os.path.join(log_dir, "reload_info_ir.txt"), "a") as f:
			f.write("\n\nTest accuracy with h vector IR: %s" % (test_acc))
		
		with open(os.path.join(log_dir, "reload_info_ir.txt"), "a") as f:
			f.write("\n\nTest F1 score with h vector IR: %s" % (test_f1))
		
	else:
		total_labels = []

		y_train_new = [label.strip().split(':') for label in y_train]
		y_test_new = [label.strip().split(':') for label in y_test]

		total_labels.extend(y_train_new)
		total_labels.extend(y_test_new)

		from sklearn.preprocessing import MultiLabelBinarizer
		mlb = MultiLabelBinarizer()
		mlb.fit(total_labels)
		y_train_one_hot = mlb.transform(y_train_new)
		y_test_one_hot = mlb.transform(y_test_new)

		train_data = (training_vectors, y_train_one_hot)
		test_data = (test_vectors, y_test_one_hot)

		test_acc, test_f1 = eval.perform_classification_test_multi(train_data, test_data, c_list)
		
		with open(os.path.join(log_dir, "reload_info_ir.txt"), "a") as f:
			f.write("\n\nTest accuracy with h vector IR: %s" % (test_acc))
		
		with open(os.path.join(log_dir, "reload_info_ir.txt"), "a") as f:
			f.write("\n\nTest F1 score with h vector IR: %s" % (test_f1))


def reload_evaluation_ir(params, training_vectors, test_vectors, W_matrix, suffix=""):

		### Information Retrieval

		dataset = data.Dataset(params['dataset'])
		log_dir = os.path.join(params['model'], 'logs')

		#ir_ratio_list = [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
		ir_ratio_list = [0.02]

		training_labels = np.array(
			[[y] for y, _ in dataset.rows('training_docnade', num_epochs=1)]
		)
		
		test_labels = np.array(
			[[y] for y, _ in dataset.rows('test_docnade', num_epochs=1)]
		)
		
		test_ir_list = eval.evaluate(
			training_vectors,
			test_vectors,
			training_labels,
			test_labels,
			recall=ir_ratio_list,
			num_classes=params['num_classes'],
			multi_label=params['multi_label']
		)

		# logging information
		with open(os.path.join(log_dir, "reload_info_ir.txt"), "a") as f:
			f.write("\n\nFractions list: %s" % (ir_ratio_list))
			f.write("\nTest IR: %s" % (test_ir_list))


def reload_evaluation_ppl(params, suffix=""):
	with tf.Session(config=tf.ConfigProto(
		inter_op_parallelism_threads=params['num_cores'],
		intra_op_parallelism_threads=params['num_cores'],
		gpu_options=tf.GPUOptions(allow_growth=True)
	)) as session_ppl:

		dataset = data.Dataset(params['dataset'])
		log_dir = os.path.join(params['model'], 'logs')
			
		saver_ppl = tf.train.import_meta_graph("model/" + params['reload_model_dir'] + "model_ppl/model_ppl-1.meta")
		saver_ppl.restore(session_ppl, tf.train.latest_checkpoint("model/" + params['reload_model_dir'] + "model_ppl/"))

		graph = tf.get_default_graph()

		x = graph.get_tensor_by_name("x:0")
		y = graph.get_tensor_by_name("y:0")
		seq_lengths = graph.get_tensor_by_name("seq_lengths:0")
		loss_normed = graph.get_tensor_by_name("loss_normed_x:0")
		loss_unnormed = graph.get_tensor_by_name("loss_unnormed_x:0")

		# TODO: Validation PPL

		this_val_nll = []
		this_val_loss_normed = []
		# val_loss_unnormed is NLL
		this_val_nll_bw = []
		this_val_loss_normed_bw = []

		this_val_disc_accuracy = []
		
		for val_y, val_x, val_seq_lengths in dataset.batches('validation_docnade', params['validation_bs'], num_epochs=1, shuffle=False, multilabel=params['multi_label']):
			
			val_loss_normed, val_loss_unnormed = session_ppl.run([loss_normed, loss_unnormed], feed_dict={
				x: val_x,
				y: val_y,
				seq_lengths: val_seq_lengths
			})

			this_val_nll.append(val_loss_unnormed)
			this_val_loss_normed.append(val_loss_normed)
		
		
		total_val_nll = np.mean(this_val_nll)
		total_val_ppl = np.exp(np.mean(this_val_loss_normed))

		print('Val PPL: {:.3f},	Val loss: {:.3f}\n'.format(
			total_val_ppl,
			total_val_nll
		))

		# logging information
		with open(os.path.join(log_dir, "reload_info_ppl_" + suffix + ".txt"), "w") as f:
			f.write("Val PPL: %s,	Val loss: %s" % 
					(total_val_ppl, total_val_nll))
		
		# TODO: Test PPL

		this_test_nll = []
		this_test_loss_normed = []
		this_test_nll_bw = []
		this_test_loss_normed_bw = []
		this_test_disc_accuracy = []
		
		for test_y, test_x, test_seq_lengths in dataset.batches('test_docnade', params['test_bs'], num_epochs=1, shuffle=False, multilabel=params['multi_label']):
			
			test_loss_normed, test_loss_unnormed = session_ppl.run([loss_normed, loss_unnormed], feed_dict={
				x: test_x,
				y: test_y,
				seq_lengths: test_seq_lengths
			})

			this_test_nll.append(test_loss_unnormed)
			this_test_loss_normed.append(test_loss_normed)
		
		total_test_nll = np.mean(this_test_nll)
		total_test_ppl = np.exp(np.mean(this_test_loss_normed))

		print('Test PPL: {:.3f},	Test loss: {:.3f}\n'.format(
			total_test_ppl,
			total_test_nll
		))

		# logging information
		with open(os.path.join(log_dir, "reload_info_ppl_" + suffix + ".txt"), "a") as f:
			f.write("\n\nTest PPL: %s,	Test loss: %s" % 
					(total_test_ppl, total_test_nll))

		W_target = session_ppl.run("embedding:0")
		
		top_n_words = 20

		# Nearest Neighbors
		with open(params['docnadeVocab'], 'r') as f:
			vocab_docnade = [w.strip() for w in f.readlines()]

		W = W_target
		
		sim_mat = pw.cosine_similarity(W, W)
		sim_mat[np.arange(len(vocab_docnade)), np.arange(len(vocab_docnade))] = -1.0

		sorted_indices = np.argsort(sim_mat, axis=1)[:, ::-1]
		
		with open(log_dir + "/nearest_neighbours.txt", "a") as f:
			for counter, indices in enumerate(sorted_indices[:, :top_n_words]):
				query_word = vocab_docnade[counter]
				nn_words = " | ".join([vocab_docnade[index] + " ( " + str(sim_mat[counter, index]) + " ) " for index in indices])
				line = query_word + " :: " + nn_words + "\n"
				f.write(line)

		
		bias_W_target = session_ppl.run("bias:0")
		U_target = session_ppl.run("U:0")
		bias_U_target = session_ppl.run("b:0")

		return W_target, bias_W_target, U_target, bias_U_target


def reload_evaluation_topics(W_target, U_target, params):

	log_dir = os.path.join(params['model'], 'logs')

	# Topics with W matrix

	top_n_topic_words = 20
	w_h_top_words_indices = []
	W_topics = W_target
	topics_list_W = []

	for h_num in range(np.array(W_topics).shape[1]):
		w_h_top_words_indices.append(np.argsort(W_topics[:, h_num])[::-1][:top_n_topic_words])

	with open(params['docnadeVocab'], 'r') as f:
		vocab_docnade = [w.strip() for w in f.readlines()]

	with open(os.path.join(log_dir, "topics_ppl_W.txt"), "w") as f:
		for w_h_top_words_indx, h_num in zip(w_h_top_words_indices, range(len(w_h_top_words_indices))):
			w_h_top_words = [vocab_docnade[w_indx] for w_indx in w_h_top_words_indx]
			
			topics_list_W.append(w_h_top_words)
			
			print('h_num: %s' % h_num)
			print('w_h_top_words_indx: %s' % w_h_top_words_indx)
			print('w_h_top_words:%s' % w_h_top_words)
			print('----------------------------------------------------------------------')

			f.write('h_num: %s\n' % h_num)
			f.write('w_h_top_words_indx: %s\n' % w_h_top_words_indx)
			f.write('w_h_top_words:%s\n' % w_h_top_words)
			f.write('----------------------------------------------------------------------\n')

	# TOPIC COHERENCE

	top_n_word_in_each_topic_list = [5, 10, 15, 20]

	text_filenames = [
		params['trainfile'],
		params['valfile'],
		params['testfile']
	]

	# read original text documents as list of words
	texts = []

	for file in text_filenames:
		print('filename:%s', file)
		for line in open(file, 'r').readlines():
			document = str(line).strip()
			texts.append(document.split())

	compute_coherence(texts, topics_list_W, top_n_word_in_each_topic_list, os.path.join(log_dir, "topics_coherence_W.txt"))


def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')


def main(args):
	args.reload = str2bool(args.reload)
	args.use_bio_prior = str2bool(args.use_bio_prior)
	args.use_fasttext_prior = str2bool(args.use_fasttext_prior)
	args.sup_projection = str2bool(args.sup_projection)
	args.sup_l2_regularization = str2bool(args.sup_l2_regularization)
	args.multi_label = str2bool(args.multi_label)
	args.run_supervised = str2bool(args.run_supervised)
	args.run_docnade = str2bool(args.run_docnade)
	args.weighted_supervised = str2bool(args.weighted_supervised)
	args.use_title_separately = str2bool(args.use_title_separately)

	if args.reload:
		with open("model/" + args.reload_model_dir + "params.json") as f:
			params = json.load(f)

		#params['trainfile'] = args.trainfile
		#params['valfile'] = args.valfile
		#params['testfile'] = args.testfile

		params['reload_model_dir'] = args.reload_model_dir

		reload_ir = False
		if os.path.isdir("model/" + args.reload_model_dir + "/model_ir"):
			reload_ir = True

		reload_ppl = False
		if os.path.isdir("model/" + args.reload_model_dir + "/model_ppl"):
			reload_ppl = True

		reload_sup = False
		if os.path.isdir("model/" + args.reload_model_dir + "/model_sup"):
			reload_sup = True

		# Reloading and evaluating on Perplexity, Topic Coherence and calculating Nearest Neighbors
		if reload_ppl:
			
			W_target, bias_W_target, U_target, bias_U_target = reload_evaluation_ppl(params, suffix="target")
			#reload_evaluation_topics(W_target, U_target, params)

		# Reloading and evaluating on Information Retrieval and Classification - F1
		if reload_ir:

			with tf.Session(config=tf.ConfigProto(
				inter_op_parallelism_threads=params['num_cores'],
				intra_op_parallelism_threads=params['num_cores'],
				gpu_options=tf.GPUOptions(allow_growth=True)
			)) as sess_ir:

				saver_ir = tf.train.import_meta_graph("model/" + args.reload_model_dir + "model_ir/model_ir-1.meta")
				saver_ir.restore(sess_ir, tf.train.latest_checkpoint("model/" + args.reload_model_dir + "model_ir/"))

				graph = tf.get_default_graph()

				x = graph.get_tensor_by_name("x:0")
				seq_lengths = graph.get_tensor_by_name("seq_lengths:0")
				last_hidden = graph.get_tensor_by_name("last_hidden:0")

				dataset = data.Dataset(params['dataset'])

				hidden_vectors_tr = []
				for tr_y, tr_x, tr_seq_lengths in dataset.batches('training_docnade', batch_size=1, num_epochs=1, shuffle=True, multilabel=params['multi_label']):
					hidden_vec = sess_ir.run([last_hidden], feed_dict={
						x: tr_x,
						seq_lengths: tr_seq_lengths
					})
					hidden_vectors_tr.append(hidden_vec[0])
				hidden_vectors_tr = np.squeeze(np.array(hidden_vectors_tr, dtype=np.float32))

				hidden_vectors_test = []
				for te_y, te_x, te_seq_lengths in dataset.batches('test_docnade', batch_size=1, num_epochs=1, shuffle=True, multilabel=params['multi_label']):
					hidden_vec = sess_ir.run([last_hidden], feed_dict={
						x: te_x,
						seq_lengths: te_seq_lengths
					})
					hidden_vectors_test.append(hidden_vec[0])
				hidden_vectors_test = np.squeeze(np.array(hidden_vectors_test, dtype=np.float32))

				W_target = sess_ir.run("embedding:0") 
			
			reload_evaluation_ir(params, hidden_vectors_tr, hidden_vectors_test, 
								W_target, suffix="target")
			"""
			reload_evaluation_f1(params, hidden_vectors_tr, hidden_vectors_test, 
								W_target, suffix="target")
			"""
		
		# Reloading and evaluating classification accuracy and mAP
		if reload_sup:
			
			params['use_bio_prior'] = False
			params['use_fasttext_prior'] = True
			params['use_BOW_repesentation'] = False
			params['use_DocNADE_W'] = False

			params['split_abstract_title'] = False
			params['include_sup_probs'] = False
			params['attention_EmbSum_type'] = "sum"

			if params['include_sup_probs']:
				params['model'] += '_including_classification_loss'

			docnade_embedding_matrix, query_embedding_matrix, query_vecs, queries, query_words_list = reload_evaluation_acc_mAP(params)

			#import pdb; pdb.set_trace()

			print("Done.")

	else:

		x = tf.placeholder(tf.int32, shape=(None, None), name='x')
		x_bw = tf.placeholder(tf.int32, shape=(None, None), name='x_bw')
		if args.multi_label:
			y = tf.placeholder(tf.string, shape=(None), name='y')
		else:
			y = tf.placeholder(tf.int32, shape=(None), name='y')
		seq_lengths = tf.placeholder(tf.int32, shape=(None), name='seq_lengths')

		if args.use_title_separately:
			x_title = tf.placeholder(tf.int32, shape=(None, None), name='x_title')
			seq_lengths_title = tf.placeholder(tf.int32, shape=(None), name='seq_lengths_title')
		else:
			x_title = None
			seq_lengths_title = None

		now = datetime.datetime.now()

		regularization_strength = 0.001

		if args.run_docnade:
			args.model += "_DocNADE"
		
		if args.run_supervised:
			args.model += "_Sup"

		if args.weighted_supervised:
			args.model += "_Weighted_" + str(args.sup_weight_init)

		if args.sup_l2_regularization:
			args.model += "_l2_reg_" + str(regularization_strength)

		if args.use_bio_prior:
			args.model += "_Emb_bio_" + str(args.lambda_glove)

		if args.use_fasttext_prior:
			args.model += "_Emb_ftt_" + str(args.lambda_glove)

		if args.W_pretrained_path or args.U_pretrained_path:
			args.model += "_Pretr_reload"
		
		args.model +=  "_Act_" + str(args.activation) + "_Hid_" + str(args.hidden_size) \
						+ "_Vocab_" + str(args.vocab_size) + "_lr_" + str(args.learning_rate) \

		if args.sup_projection:
			args.model += "_Projection_" + str(args.sup_projection_size)
		
		args.model += "_" + str(now.day) + "_" + str(now.month) + "_" + str(now.year)
		
		if not os.path.isdir(args.model):
			os.mkdir(args.model)

		docnade_vocab = args.docnadeVocab
		#with open(docnade_vocab, 'r') as f:
		with codecs.open(docnade_vocab, 'r', encoding='utf-8', errors='ignore') as f:
			vocab_docnade = [w.strip() for w in f.readlines()]

		with open(os.path.join(args.model, 'params.json'), 'w') as f:
			f.write(json.dumps(vars(args)))

		dataset = data.Dataset(args.dataset)

		docnade_embedding_matrix = None
		prior_emb_dim = None
		
		with open("./pretrained_embeddings/biggest_vocab.vocab", "r") as f:
			total_vocab = [line.strip() for line in f.readlines()]
		
		prior_embedding_matrices = []

		if args.use_bio_prior:
			bio_embeddings = np.load('./pretrained_embeddings/bionlp_embeddings_biggest_vocab.npy')
			new_bio_embs = []
			for word in vocab_docnade:
				new_bio_embs.append(bio_embeddings[total_vocab.index(word)])
			#prior_embedding_matrices.append(bio_embeddings)
			prior_embedding_matrices.append(np.array(new_bio_embs))

		if args.use_fasttext_prior:
			fasttext_embeddings = np.load('./pretrained_embeddings/fasttext_embeddings_biggest_vocab.npy')
			new_ftt_embs = []
			for word in vocab_docnade:
				new_ftt_embs.append(fasttext_embeddings[total_vocab.index(word)])
			#prior_embedding_matrices.append(fasttext_embeddings)
			prior_embedding_matrices.append(np.array(new_ftt_embs))

		if args.use_bio_prior or args.use_fasttext_prior:
			docnade_embedding_matrix = np.concatenate(prior_embedding_matrices, axis=1)
			prior_emb_dim = docnade_embedding_matrix.shape[1]

		#import pdb; pdb.set_trace()

		W_pretrained_matrix = None
		if args.W_pretrained_path:
			W_pretrained_matrix = np.load(args.W_pretrained_path)
			print("pretrained W loaded.")

		U_pretrained_matrix = None
		if args.U_pretrained_path:
			U_pretrained_matrix = np.load(args.U_pretrained_path)
			print("pretrained U loaded.")
		
		model = m.DocNADE(x, y, seq_lengths, args, \
							W_pretrained=W_pretrained_matrix, U_pretrained=U_pretrained_matrix, \
							glove_embeddings=docnade_embedding_matrix, lambda_glove=args.lambda_glove, \
							l2_reg_c=regularization_strength, prior_emb_dim=prior_emb_dim, \
							x_title=x_title, seq_lengths_title=seq_lengths_title)
		
		#import pdb; pdb.set_trace()
		print("DocNADE created")
		
		train(model, dataset, args)


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', type=str, required=True,
						help='path to model output directory')
	parser.add_argument('--dataset', type=str, required=True,
						help='path to the input dataset')
	parser.add_argument('--vocab-size', type=int, default=2000,
						help='the vocab size')
	parser.add_argument('--hidden-size', type=int, default=50,
						help='size of the hidden layer')
	parser.add_argument('--activation', type=str, default='tanh',
						help='which activation to use: sigmoid|tanh')
	parser.add_argument('--learning-rate', type=float, default=0.0004,
						help='initial learning rate')
	parser.add_argument('--num-steps', type=int, default=50000,
						help='the number of steps to train for')
	parser.add_argument('--batch-size', type=int, default=64,
						help='the batch size')
	parser.add_argument('--num-cores', type=int, default=2,
						help='the number of CPU cores to use')
	parser.add_argument('--log-every', type=int, default=10,
						help='print loss after this many steps')
	parser.add_argument('--validation-ppl-freq', type=int, default=500,
						help='print loss after this many steps')
	parser.add_argument('--num-classes', type=int, default=-1,
						help='number of classes')
	parser.add_argument('--use-bio-prior', type=str, default="False",
						help='whether to use BioNLP embeddings as prior information')
	parser.add_argument('--use-fasttext-prior', type=str, default="False",
						help='whether to use fastText embeddings as prior information')
	parser.add_argument('--docnadeVocab', type=str, default="False",
						help='path to vocabulary file used by DocNADE')
	parser.add_argument('--test-ppl-freq', type=int, default=100,
						help='print and log test PPL after this many steps')
	parser.add_argument('--test-ir-freq', type=int, default=100,
						help='print and log test IR after this many steps')
	parser.add_argument('--patience', type=int, default=10,
						help='print and log test IR after this many steps')
	parser.add_argument('--validation-bs', type=int, default=64,
						help='the batch size for validation evaluation')
	parser.add_argument('--test-bs', type=int, default=64,
						help='the batch size for test evaluation')
	parser.add_argument('--validation-ir-freq', type=int, default=500,
						help='print loss after this many steps')
	parser.add_argument('--sup-projection', type=str, default="False",
						help='whether to do projection in supervised network')
	parser.add_argument('--sup-projection-size', type=int, default=50,
						help='supervised projection hidden size')
	parser.add_argument('--sup-l2-regularization', type=str, default="False",
						help='whether to add regularization loss in supervised network')
	parser.add_argument('--reload', type=str, default="False",
						help='whether to reload model or not')
	parser.add_argument('--reload-model-dir', type=str,
						help='path for model to be reloaded')
	parser.add_argument('--run-supervised', type=str, default="False",
						help='whether to use supervised model or not')
	parser.add_argument('--run-docnade', type=str, default="False",
						help='whether to use docnade model or not')
	parser.add_argument('--weighted-supervised', type=str, default="False",
						help='whether to use weighted supervised model or not')
	parser.add_argument('--use-title-separately', type=str, default="False",
						help='whether to use titles separately or not')
	parser.add_argument('--input-type', type=str, default="both",
						help='whether to use titles/abstracts/both for training')
	parser.add_argument('--sup-weight-init', type=float, default=-1.0,
						help='initialization for weighted supervised model')
	parser.add_argument('--multi-label', type=str, default="False",
						help='whether dataset is multi-label or not')
	parser.add_argument('--trainfile', type=str, required=False,
						help='path to train text file')
	parser.add_argument('--valfile', type=str, required=False,
						help='path to validation text file')
	parser.add_argument('--testfile', type=str, required=False,
						help='path to test text file')
	parser.add_argument('--lambda-glove', type=float, default=0.0,
						help='combination weight for prior GloVe embeddings into docnade')
	parser.add_argument('--W-pretrained-path', type=str, default="",
						help='path for pretrained W matrix')
	parser.add_argument('--U-pretrained-path', type=str, default="",
						help='path for pretrained U matrix')
	parser.add_argument('--bioemb-path', type=str, default="",
						help='path for pretrained BioNLP embedding matrix')
	parser.add_argument('--fttemb-path', type=str, default="",
						help='path for pretrained BioNLP embedding matrix')


	return parser.parse_args()


query_defs = {
	"acute threat fear": "acute threat fear activation of the brainâ s defensive motivational system to promote behaviors that protect the organism from perceived danger normal fear involves a pattern of adaptive responses to conditioned or unconditioned threat stimuli exteroceptive or interoceptive fear can involve internal representations and cognitive processing and can be modulated by a variety of factors analgesia approach early development avoidance facial expressions freezing open field response inhibition response time risk assessment social approach gabaergic cells glia neurons pyramidal cells cortisol corticosterone family dopamine endogenous cannabinoids glutamate neuropeptide neurosteroids orexin oxytocin serotonin vasopressin",
	"sustained threat": "sustained threat an aversive emotional state caused by prolonged i e weeks to months exposure to internal and or external condition s state s or stimuli that are adaptive to escape or avoid the exposure may be actual or anticipated the changes in affect cognition physiology and behavior caused by sustained threat persist in the absence of the threat and can be differentiated from those changes evoked by acute threat axis hormones hippocampal microglia prefrontal anhedonia decreased appetitive behavior anxious arousal attentional bias to threat avoidance decreased libido helplessness behavior increased conflict detection increased perseverative behavior memory retrieval deficits punishment sensitivity",
	"arousal": "arousal arousal is a continuum of sensitivity of the organism to stimuli both external and internal arousal facilitates interaction with the environment in a context specific manner e g under conditions of threat some stimuli must be ignored while sensitivity to and responses to others is enhanced as exemplified in the startle reflex can be evoked by either external environmental stimuli or internal stimuli e g emotions and cognition can be modulated by the physical characteristics and motivational significance of stimuli varies along a continuum that can be quantified in any behavioral state including wakefulness and low arousal states including sleep anesthesia and coma is distinct from motivation and valence but can covary with intensity of motivation and valence may be associated with increased or decreased locomotor activity and can be regulated by homeostatic drives e g hunger sleep thirst sex acetylcholine cytokines dopamine ghrelin glutamate histamine hypocretin orexin leptin neuropeptide norepinephrine opioid oxytocin serotonin vasopressin basal forebrain nuclei central nucleus amygdala dorsal raphe lateral perifornical and dorsomedial hypothalamus locus coeruleus tuberomammillary nucleus ventral tegmental area affective states agitation cognition emotional reactivity eye blink motivated behavior motor activity sensory reactivity startle waking blood pressure breathing galvanic skin response heart rate pupil size",
	"circadian rhythms": "circadian rhythms circadian rhythms are endogenous self sustaining oscillations that organize the timing of biological systems to optimize physiology and behavior and health circadian rhythms are synchronized by recurring environmental cues anticipate the external environment allow effective response to challenges and opportunities in the physical and social environment modulate homeostasis within the brain and other central peripheral systems tissues and organs are evident across levels of organization including molecules cells circuits systems organisms and social systems drive regulated behaviors locomotor activity masking neurobehavioral functions sleep rated and waking behaviors sleep wake serotonin fibroblasts iprgc medium spiny neurons pars tuberalis cells pineal cells rods and cones cells",
	"frustrative nonreward": "frustrative nonreward reactions elicited in response to withdrawal prevention of reward i e by the inability to obtain positive rewards following repeated or sustained efforts dopamine glutamate serotonin steroids vasopressin physical and relational aggression",
	"loss": "loss state of deprivation of a motivationally significant con specific object or situation loss may be social or non social and may include permanent or sustained loss of shelter behavioral control status loved ones or relationships the response to loss may be episodic e g grief or sustained androgens estrogens glucocorticoid receptors inflammatory molecules oxytocin vasopressin amotivation anhedonia attentional bias to negative valenced information crying executive function guilt increased self focus loss of drive loss relevant recall bias morbid thoughts psychomotor retardation rumination sadness shame withdrawal worry amygdala default mode network dorsolateral prefrontal cortex habit systems striatum caudate accumbens hippocampus insula orbitofrontal cortex parietal cortex posterior cingulate gyrus reward circuitry vmpfc",
	"potential threat anxiety": "potential threat anxiety activation of a brain system in which harm may potentially occur but is distant ambiguous or low uncertain in probability characterized by a pattern of responses such as enhanced risk assessment vigilance these responses to low imminence threats are qualitatively different than the high imminence threat behaviors that characterize fear cortisol family pituitary cells",
	"sleep wakefulness": "sleep wakefulness sleep and wakefulness are endogenous recurring behavioral states that reflect coordinated changes in the dynamic functional organization of the brain and that optimize physiology behavior and health homeostatic and circadian processes regulate the propensity for wakefulness and sleep sleep is reversible typically characterized by postural recumbence behavioral quiescence and reduced responsiveness has a complex architecture with predictable cycling of states or their developmental equivalents and sleep have distinct neural substrates circuitry transmitters modulators and oscillatory properties intensity and duration is affected by homeostatic regulation is affected by experiences during wakefulness is evident at cellular circuit and system levels has restorative and transformative effects that optimize neurobehavioral functions during wakefulness acetylcholine adenosine cytokines dopamine galanin glutamate histamine hypocretin orexin norepinephrine serotonin vasopressin anterior hypothalamus and basal forebrain brainstem e g raphe lateral and perifornical hypothalamus posterior hypothalamus thalamus median thalamic nuclei reticular nucleus co sleeping intermediate admixed sleep wake states motor behaviors during sleep rest activity patterns sensory arousal threshold sex specific sleep behaviors sleep sleep deprivation and satiation sleep inertia sleep timing and variability sleep dependent neurobehavioral functions wakefulness fatigue insomnia severity index sleep quality restoration quantity sleep timing sleep modulated symptoms sleepiness",
}


if __name__ == '__main__':
	main(parse_args())
