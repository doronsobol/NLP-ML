# Copyright 2017 Benjamin Riedel
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Import relevant packages and modules
from util import *
import random
import tensorflow as tf
from tqdm import tqdm
from test_score import calc_error
import pdb
# Prompt for mode
from tf.contrib.rnn import 

flags = tf.flags
logging = tf.logging

flags.DEFINE_bool("train", False,
                          "Train the data")
flags.DEFINE_string("save_path", None,
                            "Model output directory.")
flags.DEFINE_string("data_path", None,
                            "Where the training/test data is stored.")
FLAGS = flags.FLAGS

def data_type():
    return tf.float16 if FLAGS.use_fp16 else tf.float32

class ConditionalModel(object):
  """The Conditional model model."""

  def __init__(self, is_training, batch_size, hidden_size, embedding_size):

    self.batch_size = batch_size
    self.num_steps = num_steps
    size = hidden_size
    self.embedding_size = embedding_size
    self.input = tf.placeholder(data_type(), shape=(batch_size, None, self.embedding_size))
    self.seq_len_head = tf.placeholder(tf.int32 , shape=(batch_size,))
    self.seq_len_body = tf.placeholder(tf.int32 , shape=(batch_size,))


    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to be
    # different than reported in the paper.
    head_cell = LSTMCell()
    self._initial_head_state = head_cell.zero_state(batch_size, data_type())
    self._initial_body_state = body_cell.zero_state(batch_size, data_type())

    self.head_cell = tf.nn.dynamic_rnn(attn_cell(), self.input, self.seq_len_head, initial_state=self._initial_head_state, data_type())
    self.body_cell = tf.nn.dynamic_rnn(attn_cell(), self.input, self.seq_len_body, initial_state=self._initial_body_state, data_type())

    # Simplified version of models/tutorials/rnn/rnn.py's rnn().
    # This builds an unrolled LSTM for tutorial purposes only.
    # In general, use the rnn() or state_saving_rnn() from rnn.py.
    #
    # The alternative version of the code below is:
    #
    # inputs = tf.unstack(inputs, num=num_steps, axis=1)
    # outputs, state = tf.contrib.rnn.static_rnn(
    #     cell, inputs, initial_state=self._initial_state)
    outputs = []
    state = self._initial_state
    with tf.variable_scope("RNN"):
      for time_step in range(num_steps):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        (cell_output, state) = cell(inputs[:, time_step, :], state)
        outputs.append(cell_output)

    output = tf.reshape(tf.stack(axis=1, values=outputs), [-1, size])
    softmax_w = tf.get_variable(
        "softmax_w", [size, vocab_size], dtype=data_type())
    softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
    logits = tf.matmul(output, softmax_w) + softmax_b

    # Reshape logits to be 3-D tensor for sequence loss
    logits = tf.reshape(logits, [batch_size, num_steps, vocab_size])

    # use the contrib sequence loss and average over the batches
    loss = tf.contrib.seq2seq.sequence_loss(
        logits,
        input_.targets,
        tf.ones([batch_size, num_steps], dtype=data_type()),
        average_across_timesteps=False,
        average_across_batch=True
    )

    # update the cost variables
    self._cost = cost = tf.reduce_sum(loss)
    self._final_state = state

    if not is_training:
      return

    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      config.max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(self._lr)
    self._train_op = optimizer.apply_gradients(
        zip(grads, tvars),
        global_step=tf.contrib.framework.get_or_create_global_step())

    self._new_lr = tf.placeholder(
        tf.float32, shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(self._lr, self._new_lr)

  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

  @property
  def input(self):
    return self._input

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op


class SmallConfig(object):
  """Small config."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 20
  hidden_size = 200
  max_epoch = 4
  max_max_epoch = 13
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000



mode = FALGS.train
dir = FLAGS.data_path


# Set file names
file_train_instances = dir + "/train_stances.csv"
file_train_bodies = dir + "/train_bodies.csv"
file_test_instances = dir + "/test_stances.csv"
file_test_bodies = dir + "/test_bodies.csv"
file_predictions = dir + '/predictions_test.csv'
file_test_labels = dir + "/test_stances.csv"


# Initialise hyperparameters
r = random.Random()
lim_unigram = 5000
target_size = 4
hidden_size = 100
train_keep_prob = 0.6
l2_alpha = 0.00001
learn_rate = 0.01
clip_ratio = 5
batch_size_train = 500
epochs = 90
use_hidden = input('Use hidden (y / n)? ') == 'y'
use_cosine = input('Use cosine (y / n)? ') == 'y'
just_cosine = input('Just cosine (y / n)? ') == 'y'

# Load data sets
raw_train = FNCData(file_train_instances, file_train_bodies)
raw_test = FNCData(file_test_instances, file_test_bodies)
n_train = len(raw_train.instances)
print('loaded {} instances'.format(n_train))

# Process data sets
train_heads, train_bodies, train_stances = raw_data_to_embeding(raw_train)
pdb.set_trace()
train_set, train_stances, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer = \
    pipeline_train(raw_train, raw_test, lim_unigram=lim_unigram, use_cosine=use_cosine, just_cosine=just_cosine)
feature_size = len(train_set[0])

print("feature_size is {}".format(feature_size))
test_set, test_stances = pipeline_test(raw_test, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer, use_cosine=use_cosine, just_cosine=just_cosine)
print('processed datasets')

# Define model

# Create placeholders
features_pl = tf.placeholder(tf.float32, [None, feature_size], 'features')
stances_pl = tf.placeholder(tf.int64, [None], 'stances')
keep_prob_pl = tf.placeholder(tf.float32)

# Infer batch size
batch_size = tf.shape(features_pl)[0]

# Define multi-layer perceptron
if use_hidden:
    hidden_layer = tf.nn.dropout(tf.nn.relu(tf.contrib.layers.linear(features_pl, hidden_size)), keep_prob=keep_prob_pl)
    logits_flat = tf.nn.dropout(tf.contrib.layers.linear(hidden_layer, target_size), keep_prob=keep_prob_pl)
elif not just_cosine:
    logits_flat = tf.nn.dropout(tf.contrib.layers.linear(features_pl, target_size), keep_prob=keep_prob_pl)
else:
    logits_flat = tf.contrib.layers.linear(features_pl, target_size)
    
logits = tf.reshape(logits_flat, [batch_size, target_size])

# Define L2 loss
tf_vars = tf.trainable_variables()
l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf_vars if 'bias' not in v.name]) * l2_alpha

# Define overall loss
loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, stances_pl) + l2_loss)

summary = tf.summary.scalar('loss', tf.reduce_mean(loss))



# Define prediction
softmaxed_logits = tf.nn.softmax(logits)
predict = tf.arg_max(softmaxed_logits, 1)


# Load model
if mode == 'load':
    with tf.Session() as sess:
        load_model(sess)


        # Predict
        test_feed_dict = {features_pl: test_set, keep_prob_pl: 1.0}
        test_pred = sess.run(predict, feed_dict=test_feed_dict)
    print('loaded the model')

# Train model
if mode == 'train':
    print('training...')
    # Define optimiser
    opt_func = tf.train.AdamOptimizer(learn_rate)
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tf_vars), clip_ratio)
    opt_op = opt_func.apply_gradients(zip(grads, tf_vars))

    # Perform training
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter("./summaries_dir" + '/train_' + dir,
                                      sess.graph)
        test_writer = tf.summary.FileWriter("./summaries_dir" + '/test_' + dir,
                                      sess.graph)
        sess.run(tf.global_variables_initializer())

        for epoch in tqdm(range(epochs)):
            total_loss = 0
            indices = list(range(n_train))
            r.shuffle(indices)

            for i in range(n_train // batch_size_train):
                batch_indices = indices[i * batch_size_train: (i + 1) * batch_size_train]
                batch_features = [train_set[i] for i in batch_indices]
                batch_stances = [train_stances[i] for i in batch_indices]

                batch_feed_dict = {features_pl: batch_features, stances_pl: batch_stances, keep_prob_pl: train_keep_prob}
                _, current_loss, s = sess.run([opt_op, loss, summary], feed_dict=batch_feed_dict)
                total_loss += current_loss
                train_writer.add_summary(s, epoch*(n_train // batch_size_train)+i)
            test_feed_dict = {features_pl: test_set, stances_pl: test_stances, keep_prob_pl: 1.0}
            s = sess.run(summary, feed_dict=test_feed_dict)
            test_writer.add_summary(s, epoch*(n_train // batch_size_train))
            

        # Predict
        test_feed_dict = {features_pl: test_set, keep_prob_pl: 1.0}
        test_pred = sess.run(predict, feed_dict=test_feed_dict)


# Save predictions
save_predictions(test_pred, file_predictions)
calc_error(file_predictions, file_test_labels)
