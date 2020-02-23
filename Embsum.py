import tensorflow as tf
import numpy as np
import os, sys
import pickle
import csv
import operator
import model.evaluate as eval
import model.data as data
import codecs

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

def loadGloveModel(gloveFile=None, hidden_size=None):
    if gloveFile is None:
        if hidden_size == 50:
            gloveFile = os.path.join(home_dir, "resources/pretrained_embeddings/glove.6B.50d.txt")
        elif hidden_size == 100:
            gloveFile = os.path.join(home_dir, "resources/pretrained_embeddings/glove.6B.100d.txt")
        elif hidden_size == 200:
            gloveFile = os.path.join(home_dir, "resources/pretrained_embeddings/glove.6B.200d.txt")
        elif hidden_size == 300:
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


def reload_evaluation_f1(params, training_vectors, validation_vectors, test_vectors, suffix=""):

    ### Classification - F1

    dataset = data.Dataset(params['dataset'])
    log_dir = os.path.join(params['model'], 'logs')

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    c_list = [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 3.0, 5.0, 10.0, 100.0, 500.0, 1000.0, 10000.0]
	#c_list = [1.0]

    test_acc = []
    test_f1 = []
    val_acc = []
    val_f1 = []

    test_acc_W = []
    test_f1_W = []
    val_acc_W = []
    val_f1_W = []

    y_train = np.array(
        [y for y, _ in dataset.rows('training_docnade', num_epochs=1)]
    )
    y_val = np.array(
        [y for y, _ in dataset.rows('validation_docnade', num_epochs=1)]
    )
    y_test = np.array(
        [y for y, _ in dataset.rows('test_docnade', num_epochs=1)]
    )

    if not params['multi_label']:
        train_data = (training_vectors, np.array(y_train, dtype=np.int32))
        validation_data = (validation_vectors, np.array(y_val, dtype=np.int32))
        test_data = (test_vectors, np.array(y_test, dtype=np.int32))

        test_acc, test_f1 = eval.perform_classification_test(train_data, test_data, c_list)
        
        with open(os.path.join(log_dir, "reload_info_ir.txt"), "a") as f:
            f.write("\n\nTest accuracy with h vector IR: %s" % (test_acc))
        
        with open(os.path.join(log_dir, "reload_info_ir.txt"), "a") as f:
            f.write("\n\nTest F1 score with h vector IR: %s" % (test_f1))
    else:
        total_labels = []

        y_train_new = [label.strip().split(':') for label in y_train]
        y_val_new = [label.strip().split(':') for label in y_val]
        y_test_new = [label.strip().split(':') for label in y_test]

        total_labels.extend(y_train_new)
        #total_labels.extend(y_val_new)
        #total_labels.extend(y_test_new)

        from sklearn.preprocessing import MultiLabelBinarizer
        mlb = MultiLabelBinarizer()
        mlb.fit(total_labels)
        y_train_one_hot = mlb.transform(y_train_new)
        y_val_one_hot = mlb.transform(y_val_new)
        y_test_one_hot = mlb.transform(y_test_new)

        train_data = (training_vectors, y_train_one_hot)
        validation_data = (validation_vectors, y_val_one_hot)
        test_data = (test_vectors, y_test_one_hot)

        test_acc, test_f1 = eval.perform_classification_test_multi(train_data, test_data, c_list)
        
        with open(os.path.join(log_dir, "reload_info_ir.txt"), "a") as f:
            f.write("\n\nTest accuracy with h vector IR: %s" % (test_acc))
        
        with open(os.path.join(log_dir, "reload_info_ir.txt"), "a") as f:
            f.write("\n\nTest F1 score with h vector IR: %s" % (test_f1))


def reload_evaluation_ir(params, training_vectors, validation_vectors, test_vectors, suffix=""):

	### Information Retrieval

	dataset = data.Dataset(params['dataset'])
	log_dir = os.path.join(params['model'], 'logs')

	if not os.path.exists(log_dir):
		os.makedirs(log_dir)

	ir_ratio_list = [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
	#ir_ratio_list = [0.02]

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


dataset = "Task1_Augmented_Def"

model_type = "EmbSum"
data_dir = "./datasets/" + dataset
params = {}
params['dataset'] = data_dir
params['num_classes'] = 8
params['multi_label'] = False
params['use_bio_prior'] = True
params['use_fasttext_prior'] = True
hidden_size = 200

params['model'] = os.path.join(os.getcwd(), "model", dataset)

if params['use_bio_prior']:
    params['model'] += "_bio_EmbSum"

if params['use_fasttext_prior']:
    params['model'] += "_ftt_EmbSum"
#else:
#    print("Cannot compute accuracy without any embeddings.")
#    sys.exit()


with codecs.open(data_dir + "/vocab_docnade.vocab", "r", encoding='utf-8', errors='ignore') as f:
    vocab_docnade = [line.strip() for line in f.readlines()]
    
prior_embedding_matrices = []

if params['use_bio_prior']:
    #bio_embeddings = np.load('./pretrained_embeddings/bionlp_embeddings_task1_without_acronyms_without_stopwords.npy')
    bio_embeddings = np.load('./pretrained_embeddings/bionlp_embeddings_task1_without_acronyms_without_stopwords_augmented_def.npy')
    prior_embedding_matrices.append(bio_embeddings)

if params['use_fasttext_prior']:
    #fasttext_embeddings = np.load('./pretrained_embeddings/fasttext_embeddings_task1_without_acronyms_without_stopwords.npy')
    fasttext_embeddings = np.load('./pretrained_embeddings/fasttext_embeddings_task1_without_acronyms_without_stopwords_augmented_def.npy')
    prior_embedding_matrices.append(fasttext_embeddings)

if params['use_bio_prior'] or params['use_fasttext_prior']:
    docnade_embedding_matrix = np.concatenate(prior_embedding_matrices, axis=1)

training_vecs = []
training_labels = []
with open(data_dir + "/training_docnade.csv", "r") as f:
    file_reader = csv.reader(f, delimiter=",")
    for row in file_reader:
        tokens = [int(index) for index in row[1].strip().split()]
        Embs = docnade_embedding_matrix[np.array(tokens), :]
        EmbSum = np.sum(Embs, axis=0)
        training_vecs.append(EmbSum)
        training_labels.append([row[0]])

validation_vecs = []
validation_labels = []
with open(data_dir + "/validation_docnade.csv", "r") as f:
    file_reader = csv.reader(f, delimiter=",")
    for row in file_reader:
        tokens = [int(index) for index in row[1].strip().split()]
        Embs = docnade_embedding_matrix[np.array(tokens), :]
        EmbSum = np.sum(Embs, axis=0)
        validation_vecs.append(EmbSum)
        validation_labels.append([row[0]])

test_vecs = []
test_labels = []
with open(data_dir + "/test_docnade.csv", "r") as f:
    file_reader = csv.reader(f, delimiter=",")
    for row in file_reader:
        tokens = [int(index) for index in row[1].strip().split()]
        Embs = docnade_embedding_matrix[np.array(tokens), :]
        EmbSum = np.sum(Embs, axis=0)
        test_vecs.append(EmbSum)
        test_labels.append([row[0]])


#reload_evaluation_ir(params, training_vecs, validation_vecs, test_vecs)
reload_evaluation_f1(params, training_vecs, validation_vecs, test_vecs)
		

print("Done.")