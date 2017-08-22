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
mode = input('mode (load / train)? ')
dir = input('Enter dir: ')

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
loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=stances_pl) + l2_loss)

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
