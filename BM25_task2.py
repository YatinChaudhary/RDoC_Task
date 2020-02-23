import gensim
from gensim import corpora
import math
import operator
import numpy as np
import os, sys
import model.evaluate as eval
import sklearn.metrics.pairwise as pw
import csv

def uwords(words):
	uw = {}
	for w in words:
		uw[w] = 1
	return [w for w in uw]

def ubigrams(words):
	uw = {}
	for i in range(len(words)-1):
		uw[str(words[i]) + '_' + str(words[i+1])] = 1
	return [w for w in uw]

class BM25 :
	def __init__(self, fn_docs, delimiter=' ') :
		self.dictionary = corpora.Dictionary()
		self.DF = {}
		self.delimiter = delimiter
		self.DocTF = []
		self.DocIDF = {}
		self.N = 0
		self.DocAvgLen = 0
		self.fn_docs = fn_docs
		self.DocLen = []
		self.buildDictionary()
		self.TFIDF_Generator()

	def buildDictionary(self) :
		raw_data = []
		for line in self.fn_docs:
			raw_data.append(line.strip().split(self.delimiter))
		self.dictionary.add_documents(raw_data)

	def TFIDF_Generator(self, base=math.e) :
		docTotalLen = 0
		for line in self.fn_docs:
			doc = line.strip().split(self.delimiter)
			docTotalLen += len(doc)
			self.DocLen.append(len(doc))
			bow = dict([(term, freq*1.0/len(doc)) for term, freq in self.dictionary.doc2bow(doc)])
			for term, tf in bow.items() :
				if term not in self.DF :
					self.DF[term] = 0
				self.DF[term] += 1
			self.DocTF.append(bow)
			self.N = self.N + 1
		for term in self.DF:
			self.DocIDF[term] = math.log(1 + ((self.N - self.DF[term] +0.5) / (self.DF[term] + 0.5)), base)
		self.DocAvgLen = docTotalLen / self.N

	#def BM25Score(self, Query=[], k1=1.5, b=0.75, 
	def BM25Score(self, Query=[], k1=1.5, b=0.75, 
				embedding_matrix=None, embedding_vocab=None, 
				query_matrix=None, query_vocab=None, 
				sim_threshold=0.30) :
		query_bow = self.dictionary.doc2bow(Query)
		scores = []
		for idx, doc in enumerate(self.DocTF) :
			commonTerms = set(dict(query_bow).keys()) & set(doc.keys())
			if not embedding_matrix is None:
				#import pdb; pdb.set_trace()
				id2token = {id: word for word, id in self.dictionary.token2id.items()}
				try:
					query_words = Query
					#doc_words = [self.dictionary.id2token[id].lower().strip() for id in doc.keys()]
					doc_words = [id2token[id].lower().strip() for id in doc.keys()]
					query_vecs = query_matrix[np.array([query_vocab.index(word) for word in query_words]), :]
					#doc_vecs = embedding_matrix[np.array([embedding_vocab.index(word) for word in doc_words]), :]
					doc_vecs = np.zeros((len(doc_words), embedding_matrix.shape[1]), dtype=np.float32)
					for i, word in enumerate(doc_words):
						try:
							doc_vecs[i, :] = embedding_matrix[embedding_vocab.index(word), :]
						except ValueError:
							pass
					#import pdb; pdb.set_trace()
					sim = pw.cosine_similarity(query_vecs, doc_vecs)
					#sim = eval.softmax(pw.cosine_similarity(query_vecs, doc_vecs), axis=1)
					similar_words = []
					for row in sim:
						sim_words = [self.dictionary.token2id[doc_words[ind]] for ind, val in enumerate(row) if val >= sim_threshold]
						similar_words.extend(sim_words)
					commonTerms = commonTerms.union(set(similar_words))
				except IndexError:
					import pdb; pdb.set_trace()
			tmp_score = []
			doc_terms_len = self.DocLen[idx]
			for term in commonTerms :
				upper = (doc[term] * (k1+1))
				below = ((doc[term]) + k1*(1 - b + b*doc_terms_len/self.DocAvgLen))
				tmp_score.append(self.DocIDF[term] * upper / below)
			scores.append(sum(tmp_score))
		return scores

	def TFIDF(self) :
		tfidf = []
		for doc in self.DocTF :
			doc_tfidf  = [(term, tf*self.DocIDF[term]) for term, tf in doc.items()]
			doc_tfidf.sort()
			tfidf.append(doc_tfidf)
		return tfidf

	def Items(self) :
		# Return a list [(term_idx, term_desc),]
		items = self.dictionary.items()
		#items.sort()
		items = sorted(items, key=operator.itemgetter(1))
		return items

	def query_doc_overlap(self, qwords, dwords):

		# % Query words in doc.
		qwords_in_doc = 0
		idf_qwords_in_doc = 0.0
		idf_qwords = 0.0
		for qword in uwords(qwords):
			idf_qwords += self.DocIDF[qword]
			for dword in uwords(dwords):
				if qword == dword:
					idf_qwords_in_doc += self.DocIDF[qword]
					qwords_in_doc += 1
					break
		if len(qwords) <= 0:
			qwords_in_doc_val = 0.0
		else:
			qwords_in_doc_val = (float(qwords_in_doc) /
								float(len(uwords(qwords))))
		if idf_qwords <= 0.0:
			idf_qwords_in_doc_val = 0.0
		else:
			idf_qwords_in_doc_val = float(idf_qwords_in_doc) / float(idf_qwords)

		# % Query bigrams  in doc.
		qwords_bigrams_in_doc = 0
		idf_qwords_bigrams_in_doc = 0.0
		idf_bigrams = 0.0
		for qword in ubigrams(qwords):
			wrds = [int(w) for w in qword.split('_')]
			idf_bigrams += self.DocIDF[wrds[0]] * self.DocIDF[wrds[1]]
			for dword in ubigrams(dwords):
				if qword == dword:
					qwords_bigrams_in_doc += 1
					idf_qwords_bigrams_in_doc += (self.DocIDF[wrds[0]]
													* self.DocIDF[wrds[1]])
					break
		#if len(qwords) <= 0:
		if len(qwords) <= 1:
			qwords_bigrams_in_doc_val = 0.0
		else:
			qwords_bigrams_in_doc_val = (float(qwords_bigrams_in_doc) /
										float(len(ubigrams(qwords))))
		if idf_bigrams <= 0.0:
			idf_qwords_bigrams_in_doc_val = 0.0
		else:
			idf_qwords_bigrams_in_doc_val = (float(idf_qwords_bigrams_in_doc) /
											float(idf_bigrams))

		return [qwords_in_doc_val,
				qwords_bigrams_in_doc_val,
				idf_qwords_in_doc_val,
				idf_qwords_bigrams_in_doc_val]

if __name__ == '__main__' :
	#mycorpus.txt is as following:
	'''
	Human machine interface for lab abc computer applications
	A survey of user opinion of computer system response time
	The EPS user interface management system
	System and human system engineering testing of EPS
	Relation of user perceived response time to error measurement
	The generation of random binary unordered trees
	The intersection graph of paths in trees
	Graph IV Widths of trees and well quasi ordering
	Graph minors A survey
	'''
	data_dir = 'Task1_without_acronym'
	log_dir = "./model/" + data_dir + "_BM25"
	fn_docs = "./datasets/" + data_dir + '/val_docs.txt'

	if not os.path.exists(log_dir):
		os.makedirs(log_dir)

	bm25 = BM25(fn_docs, delimiter=' ')
	
	Query1 = 'potential threat anxiety'
	Query1 = Query1.split()
	scores1 = bm25.BM25Score(Query1)

	Query2 = 'arousal'
	Query2 = Query2.split()
	scores2 = bm25.BM25Score(Query2)

	Query3 = 'sleep wakefulness'
	Query3 = Query3.split()
	scores3 = bm25.BM25Score(Query3)

	Query4 = 'frustrative nonreward'
	Query4 = Query4.split()
	scores4 = bm25.BM25Score(Query4)

	Query5 = 'sustained threat'
	Query5 = Query5.split()
	scores5 = bm25.BM25Score(Query5)

	Query6 = 'circadian rhythms'
	Query6 = Query6.split()
	scores6 = bm25.BM25Score(Query6)

	Query7 = 'acute threat fear'
	Query7 = Query7.split()
	scores7 = bm25.BM25Score(Query7)

	Query8 = 'loss'
	Query8 = Query8.split()
	scores8 = bm25.BM25Score(Query8)

	scores = np.stack([scores1, scores2, scores3, scores4, scores5, scores6, scores7, scores8], axis=1)

	pred_labels = np.argmax(scores, axis=1)
	pred_probs = scores[np.arange(len(pred_labels)), np.array(pred_labels)]

	with open("./datasets/" + data_dir + "/labels.txt", "r") as f:
		txt_labels = [line.lower().strip() for line in f.readlines()]

	true_labels = []
	with open(fn_docs, "r") as f:
		true_labels.extend([txt_labels.index(line.strip().split("\t")[0].lower()) for line in f.readlines()])

	mAP, AP_dict, preds_dict, probs_dict = eval.evaluate_mAP(true_labels, pred_labels, pred_probs)
        
	print('This val mAP: {:.3f}'.format(mAP))

	# logging information
	with open(os.path.join(log_dir, "BM25_mAP.txt"), "w") as f:
		f.write("mAP: %s\n" % (mAP))

	"""
	labels = np.unique(true_labels)
	preds_dict = {label:[] for label in labels}
	probs_dict = {label:[] for label in labels}
	for true, pred, prob in zip(true_labels, pred_labels, pred_probs):
		preds_dict[pred].append(true)
		probs_dict[pred].append(prob)
	"""

	with open(os.path.join(log_dir, "BM25_clusters.txt"), "w") as f:
		for label in preds_dict.keys():
			preds = preds_dict[label]
			probs = probs_dict[label]

			sorted_indices = np.argsort(probs)[::-1]
			sorted_preds = np.array(preds)[sorted_indices]
			sorted_probs = np.array(probs)[sorted_indices]

			f.write("Cluster " + str(label) + "\n\n")
			f.write("Average precision: " + str(AP_dict[label]) + "\n")
			f.write("Predicted_labels: " + " ".join([str(l) for l in sorted_preds]) + "\n")
			f.write("Predicted_probs:  " + " ".join([str(l) for l in sorted_probs]) + "\n")
			f.write("\n\n")

	#tfidf = bm25.TFIDF()
	#print(bm25.Items())
	#for i, tfidfscore in enumerate(tfidf):
	#	print(i, tfidfscore)

	#import pdb; pdb.set_trace()

	#print(scores)
	print("Done.")