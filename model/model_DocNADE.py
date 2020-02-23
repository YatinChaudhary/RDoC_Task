import sys
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops

seed = 42
tf_op_seed = 1234

np.random.seed(seed)
tf.set_random_seed(seed)


def vectors(model, data, session):
    vecs = []
    for _, x, seq_lengths in data:
        vecs.extend(
            session.run([model.h], feed_dict={
                model.x: x,
                model.seq_lengths: seq_lengths
            })[0]
        )
    return np.array(vecs)


def loss(model, data, session):
    loss = []
    for _, x, seq_lengths in data:
        loss.append(
            session.run([model.loss], feed_dict={
                model.x: x,
                model.seq_lengths: seq_lengths
            })[0]
        )
    return sum(loss) / len(loss)


def gradients(opt, loss, vars, step, max_gradient_norm=None, dont_clip=[]):
    gradients = opt.compute_gradients(loss, vars)
    if max_gradient_norm is not None:
        to_clip = [(g, v) for g, v in gradients if v.name not in dont_clip]
        not_clipped = [(g, v) for g, v in gradients if v.name in dont_clip]
        gradients, variables = zip(*to_clip)
        clipped_gradients, _ = clip_ops.clip_by_global_norm(
            gradients,
            max_gradient_norm
        )
        gradients = list(zip(clipped_gradients, variables)) + not_clipped

    # Add histograms for variables, gradients and gradient norms
    for gradient, variable in gradients:
        if isinstance(gradient, ops.IndexedSlices):
            grad_values = gradient.values
        else:
            grad_values = gradient
        if grad_values is None:
            print('warning: missing gradient: {}'.format(variable.name))
        if grad_values is not None:
            tf.summary.histogram(variable.name, variable)
            tf.summary.histogram(variable.name + '/gradients', grad_values)
            tf.summary.histogram(
                variable.name + '/gradient_norm',
                clip_ops.global_norm([grad_values])
            )

    return opt.apply_gradients(gradients, global_step=step)


def linear(input, output_dim, scope=None, stddev=None, U_pretrained=None):
    const = tf.constant_initializer(0.0)

    if U_pretrained is None:
        if stddev:
            norm = tf.random_normal_initializer(stddev=stddev, seed=tf_op_seed)
        else:
            norm = tf.random_normal_initializer(
                stddev=np.sqrt(2.0 / input.get_shape()[1].value),
                seed=tf_op_seed
            )

        U = tf.get_variable(
            'U',
            [input.get_shape()[1], output_dim],
            initializer=norm
        )
    else:
        U = tf.get_variable(
            'U',
            initializer=U_pretrained
        )

    b = tf.get_variable('b', [output_dim], initializer=const)

    input_logits = tf.nn.xw_plus_b(input, U, b)
    
    return input_logits, U


def masked_sequence_cross_entropy_loss(
    x,
    seq_lengths,
    logits,
    loss_function=None,
    norm_by_seq_lengths=True,
    name=""
):
    '''
    Compute the cross-entropy loss between all elements in x and logits.
    Masks out the loss for all positions greater than the sequence
    length (as we expect that sequences may be padded).

    Optionally, also either use a different loss function (eg: sampled
    softmax), and/or normalise the loss for each sequence by the
    sequence length.
    '''
    batch_size = tf.shape(x)[0]
    labels = tf.reshape(x, [-1])

    
    max_doc_length = tf.shape(x)[1]
    mask = tf.less(
        tf.range(0, max_doc_length, 1),
        tf.reshape(seq_lengths, [batch_size, 1])
    )
    mask = tf.reshape(mask, [-1])
    mask = tf.to_float(tf.where(
        mask,
        tf.ones_like(labels, dtype=tf.float32),
        tf.zeros_like(labels, dtype=tf.float32)
    ))

    if loss_function is None:
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits,
            labels=labels
        )
    else:
        loss = loss_function(logits, labels)
    loss *= mask
    loss = tf.reshape(loss, [batch_size, -1])
    loss = tf.reduce_sum(loss, axis=1)
    loss_unnormed = loss
    if norm_by_seq_lengths:
        loss = loss / tf.to_float(seq_lengths)
    return tf.reduce_mean(loss, name="loss_normed_" + name), labels, mask, tf.reduce_mean(loss_unnormed, name="loss_unnormed_" + name)


class DocNADE(object):
    def __init__(self, x, y, seq_lengths, params,
                 W_pretrained=None, U_pretrained=None, 
                 glove_embeddings=None, lambda_glove=0.0, 
                 l2_reg_c=1.0, prior_emb_dim=0, 
                 x_title=None, seq_lengths_title=None):
        self.x = x
        self.y = y
        self.seq_lengths = seq_lengths

        self.x_title = x_title
        self.seq_lengths_title = seq_lengths_title

        batch_size = tf.shape(x)[0]
        self.b_s = tf.shape(x)
        self.lambda_glove = lambda_glove

        # Do an embedding lookup for each word in each sequence
        with tf.device('/cpu:0'):
            if W_pretrained is None:
                max_embed_init = 1.0 / (params.vocab_size * params.hidden_size)
                W = tf.get_variable(
                    'embedding',
                    [params.vocab_size, params.hidden_size],
                    initializer=tf.random_uniform_initializer(
                        maxval=max_embed_init,
                        seed=tf_op_seed
                    )
                )
            else:
                W = tf.get_variable(
                    'embedding',
                    initializer=W_pretrained
                )
            
            self.embeddings = tf.nn.embedding_lookup(W, x)
            
            if not glove_embeddings is None:
                glove_prior = tf.get_variable(
                    'glove_prior',
                    initializer=glove_embeddings,
                    trainable=False
                )
                """
                self.embeddings_prior = tf.nn.embedding_lookup(glove_prior, x)
                
                if params.run_docnade:
                    # Lambda multiplication
                    if not self.lambda_glove < 0.0:
                        self.embeddings_prior = tf.scalar_mul(self.lambda_glove, self.embeddings_prior)
                    
                    self.embeddings = tf.add(self.embeddings, self.embeddings_prior)
                """
            bias = tf.get_variable(
                'bias',
                [params.hidden_size],
                initializer=tf.constant_initializer(0)
            )

        # Compute the hidden layer inputs: each gets summed embeddings of
        # previous words
        def sum_embeddings(previous, current):
            return previous + current

        h = tf.scan(sum_embeddings, tf.transpose(self.embeddings, [1, 2, 0]))
        h = tf.transpose(h, [2, 0, 1])
        
        h = tf.concat([
            tf.zeros([batch_size, 1, params.hidden_size], dtype=tf.float32), h
        ], axis=1)
        
        self.pre_act = h

        # Apply activation
        if params.activation == 'sigmoid':
            h = tf.sigmoid(h + bias)
        elif params.activation == 'tanh':
            h = tf.tanh(h + bias)
        elif params.activation == 'relu':
            h = tf.nn.relu(h + bias)
        else:
            print('Invalid value for activation: %s' % (params.activation))
            exit()
        
        self.aft_act = h

        # Extract final state for each sequence in the batch
        indices = tf.stack([
            tf.range(batch_size),
            tf.to_int32(seq_lengths)
        ], axis=1)
        self.indices = indices

        #if not params.run_supervised:
        #    self.h = tf.gather_nd(h, indices, name='last_hidden')
        self.h = tf.gather_nd(h, indices, name='last_hidden')

        h = h[:, :-1, :]
        h = tf.reshape(h, [-1, params.hidden_size])

        ################################# SUPERVISED NETWORK ###################################

        if params.run_supervised:

            if params.run_docnade and (not glove_embeddings is None):
                #sup_matrix = tf.add(W, glove_prior, name='add_embeddings')
                #sup_matrix = tf.concat([W, glove_prior], axis=1, name='concat_embeddings')
                #sup_input_size = params.hidden_size
                #sup_input_size = params.hidden_size + prior_emb_dim
                sup_matrix = glove_prior
                sup_input_size = prior_emb_dim
            elif (not params.run_docnade) and (not glove_embeddings is None):
                sup_matrix = glove_prior
                sup_input_size = prior_emb_dim
            elif (glove_embeddings is None):
                sup_matrix = W
                sup_input_size = params.hidden_size
            else:
                print("Wrong combination of params.run_docnade and glove_prior.")
                sys.exit()

            #sup_input_size = tf.shape(sup_matrix)[1]

            if params.weighted_supervised:
                max_embed_init = 1.0 / (params.vocab_size)

                if params.sup_weight_init < 0.0:
                    weight_initializer=tf.random_uniform_initializer(
                        maxval=max_embed_init,
                        seed=tf_op_seed
                    )
                else:
                    weight_initializer=tf.constant_initializer(params.sup_weight_init)

                embeddings_lambda_list = tf.get_variable(
                    'embeddings_lambda_list_unclipped',
                    [params.vocab_size, 1],
                    initializer=weight_initializer,
                    trainable=True
                )
                #self.embeddings_lambda_list = tf.clip_by_value(embeddings_lambda_list, 0.0, 1.0, name='embeddings_lambda_list')
                self.embeddings_lambda_list = tf.sigmoid(embeddings_lambda_list, name='embeddings_lambda_list')

                sup_matrix_weighted = tf.multiply(sup_matrix, self.embeddings_lambda_list)

                self.weighted_sup_lookup = tf.nn.embedding_lookup(sup_matrix_weighted, x)

                if params.use_title_separately:
                    self.weighted_sup_title_lookup = tf.nn.embedding_lookup(sup_matrix_weighted, x_title)

                if params.run_docnade:
                    W_weighted = tf.multiply(W, self.embeddings_lambda_list)
                    self.weighted_embeddings = tf.nn.embedding_lookup(W_weighted, x)

                    weighted_h_docnade = tf.scan(sum_embeddings, tf.transpose(self.weighted_embeddings, [1, 2, 0]))
                    weighted_h_docnade = tf.transpose(weighted_h_docnade, [2, 0, 1])

                    weighted_embeddings_indices = tf.stack([
                        tf.range(batch_size),
                        tf.to_int32(seq_lengths) - 1
                    ], axis=1)

                    self.h = tf.gather_nd(weighted_h_docnade, weighted_embeddings_indices, name='last_hidden')

                    # Apply activation
                    if params.activation == 'sigmoid':
                        self.h = tf.sigmoid(self.h + bias)
                    elif params.activation == 'tanh':
                        self.h = tf.tanh(self.h + bias)
                    elif params.activation == 'relu':
                        self.h = tf.nn.relu(self.h + bias)
                    else:
                        print('Invalid value for activation: %s' % (params.activation))
                        exit()
            else:
                self.weighted_sup_lookup = tf.nn.embedding_lookup(sup_matrix, x)

                if params.use_title_separately:
                    self.weighted_sup_title_lookup = tf.nn.embedding_lookup(sup_matrix, x_title)

            weighted_h = tf.scan(sum_embeddings, tf.transpose(self.weighted_sup_lookup, [1, 2, 0]))
            weighted_h = tf.transpose(weighted_h, [2, 0, 1])
            
            #weighted_h = tf.concat([
            #    #tf.zeros([batch_size, 1, params.hidden_size], dtype=tf.float32), weighted_h
            #    tf.zeros([batch_size, 1, sup_input_size], dtype=tf.float32), weighted_h
            #], axis=1)

            sup_indices = tf.stack([
                tf.range(batch_size),
                tf.to_int32(seq_lengths) - 1
            ], axis=1)

            self.h_sup = tf.gather_nd(weighted_h, sup_indices, name='sup_input_hidden')
            
            """
            if params.run_docnade:
                # Apply activation
                if params.activation == 'sigmoid':
                    #self.h = tf.sigmoid(self.h + bias)
                    self.h = tf.sigmoid(self.h)
                elif params.activation == 'tanh':
                    #self.h = tf.tanh(self.h + bias)
                    self.h = tf.tanh(self.h)
                elif params.activation == 'relu':
                    #self.h = tf.nn.relu(self.h + bias)
                    self.h = tf.nn.relu(self.h)
                else:
                    print('Invalid value for activation: %s' % (params.activation))
                    exit()    
            """

            if params.use_title_separately:
                weighted_title_h = tf.scan(sum_embeddings, tf.transpose(self.weighted_sup_title_lookup, [1, 2, 0]))
                weighted_title_h = tf.transpose(weighted_title_h, [2, 0, 1])

                sup_indices_title = tf.stack([
                    tf.range(batch_size),
                    tf.to_int32(seq_lengths_title) - 1
                ], axis=1)

                self.h_title = tf.gather_nd(weighted_title_h, sup_indices_title, name='last_hidden_title')

                if params.run_docnade:
                    # Apply activation
                    if params.activation == 'sigmoid':
                        self.h_title = tf.sigmoid(self.h_title + bias)
                    elif params.activation == 'tanh':
                        self.h_title = tf.tanh(self.h_title + bias)
                    elif params.activation == 'relu':
                        self.h_title = tf.nn.relu(self.h_title + bias)
                    else:
                        print('Invalid value for activation: %s' % (params.activation))
                        exit()  

                self.h = tf.concat([
                    self.h_title, self.h
                ], axis=1, name='last_hidden_comb')

                sup_input_size = 2 * sup_input_size
            
            #self.disc_h = self.weighted_h

            if params.sup_projection:
                max_U1_init = 1.0 / (sup_input_size * params.sup_projection_size)
                U1 = tf.get_variable(
                    'U1_supervised',
                    [sup_input_size, params.sup_projection_size],
                    initializer=tf.random_uniform_initializer(
                        maxval=max_U1_init,
                        seed=tf_op_seed
                    ),
                    trainable=True
                )

                d1 = tf.get_variable(
                    'd1_supervised',
                    [params.sup_projection_size],
                    initializer=tf.constant_initializer(0),
                    trainable=True
                )

                #sup_hidden = tf.nn.xw_plus_b(self.h, U1, d1, name='disc_hidden')
                #sup_hidden = tf.nn.xw_plus_b(self.h_sup, U1, d1, name='disc_hidden')

                # Apply activation
                if params.activation == 'sigmoid':
                    sup_hidden = tf.sigmoid(tf.nn.xw_plus_b(self.h_sup, U1, d1, name='disc_hidden'))
                elif params.activation == 'tanh':
                    sup_hidden = tf.tanh(tf.nn.xw_plus_b(self.h_sup, U1, d1, name='disc_hidden'))
                elif params.activation == 'relu':
                    sup_hidden = tf.nn.relu(tf.nn.xw_plus_b(self.h_sup, U1, d1, name='disc_hidden'))
                else:
                    print('Invalid value for activation: %s' % (params.activation))
                    exit()

                sup_hidden = tf.concat([
                    self.h, sup_hidden
                ], axis=1, name='last_hidden_comb')
                params.sup_projection_size = params.sup_projection_size + params.hidden_size

                max_U2_init = 1.0 / (params.sup_projection_size * params.num_classes)
                U2 = tf.get_variable(
                    'U2_supervised',
                    [params.sup_projection_size, params.num_classes],
                    initializer=tf.random_uniform_initializer(
                        maxval=max_U2_init,
                        seed=tf_op_seed
                    ),
                    trainable=True
                )

                d2 = tf.get_variable(
                    'd2_supervised',
                    [params.num_classes],
                    initializer=tf.constant_initializer(0),
                    trainable=True
                )

                disc_logits = tf.nn.xw_plus_b(sup_hidden, U2, d2, name='disc_logits')

                """
                # Apply activation
                if params.activation == 'sigmoid':
                    disc_logits = tf.sigmoid(tf.nn.xw_plus_b(sup_hidden, U2, d2, name='disc_logits'))
                elif params.activation == 'tanh':
                    disc_logits = tf.tanh(tf.nn.xw_plus_b(sup_hidden, U2, d2, name='disc_logits'))
                elif params.activation == 'relu':
                    disc_logits = tf.nn.relu(tf.nn.xw_plus_b(sup_hidden, U2, d2, name='disc_logits'))
                else:
                    print('Invalid value for activation: %s' % (params.activation))
                    exit()
                """

                l2_reg_loss = tf.nn.l2_loss(U1) + tf.nn.l2_loss(d1) + tf.nn.l2_loss(U2) + tf.nn.l2_loss(d2)

            else:
                max_U_init = 1.0 / (params.hidden_size * params.num_classes)
                U = tf.get_variable(
                    'U_supervised',
                    [sup_input_size, params.num_classes],
                    initializer=tf.random_uniform_initializer(
                        maxval=max_U_init,
                        seed=tf_op_seed
                    ),
                    trainable=True
                )

                d = tf.get_variable(
                    'd_supervised',
                    [params.num_classes],
                    initializer=tf.constant_initializer(0),
                    trainable=True
                )

                #disc_logits = tf.nn.xw_plus_b(self.disc_h, U, d, name='disc_logits')
                #disc_logits = tf.nn.xw_plus_b(self.h, U, d, name='disc_logits')
                disc_logits = tf.nn.xw_plus_b(self.h_sup, U, d, name='disc_logits')

                l2_reg_loss = tf.nn.l2_loss(U) + tf.nn.l2_loss(d)
            
            self.disc_output = disc_logits

            labels = y
            disc_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels,
                logits=disc_logits,
            )
            self.disc_loss = tf.reduce_mean(disc_loss, name='disc_loss')
            self.pred_labels = tf.argmax(disc_logits, axis=1, name='pred_labels')
            
        ############################################################################################
        
        self.logits, U_new = linear(h, params.vocab_size, scope='softmax', U_pretrained=U_pretrained)
        loss_function = None
        
        self.loss_normed, self.labels, self.mask, self.loss_unnormed = masked_sequence_cross_entropy_loss(
            x,
            seq_lengths,
            self.logits,
            loss_function=loss_function,
            norm_by_seq_lengths=True,
            name="x"
        )

        #self.total_loss = tf.identity(self.loss_unnormed, name="total_loss")
        self.total_loss = 0.0

        if params.run_docnade:
            self.total_loss += self.loss_unnormed
        
        if params.run_supervised:
            self.total_loss += self.disc_loss

            if params.sup_l2_regularization:
                self.total_loss += l2_reg_c * l2_reg_loss

        # Optimiser
        step = tf.Variable(0, trainable=False)
        self.opt = gradients(
            opt=tf.train.AdamOptimizer(learning_rate=params.learning_rate),
            loss=self.total_loss,
            vars=tf.trainable_variables(),
            step=step
        )