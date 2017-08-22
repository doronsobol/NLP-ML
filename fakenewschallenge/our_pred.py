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
#from util import *
import random
import tensorflow as tf
from tensorflow import TensorShape, Dimension
from tqdm import tqdm
#from test_score import calc_error
import pdb
import functools
# Prompt for mode
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
import tensorflow.contrib.learn as learn
from gensim.models.keyedvectors import KeyedVectors
from nltk import word_tokenize
from tensorflow.python import debug as tf_debug

tf.logging.set_verbosity(tf.logging.INFO)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_bool("train", False,
                          "Train the data")
tf.app.flags.DEFINE_bool("use_fp16", False,
                          "Data type to use")
tf.app.flags.DEFINE_string("save_path", None,
                            "Model output directory.")
tf.app.flags.DEFINE_string("log_dir", "log",
                            "Where the tensorboard files are located.")
tf.app.flags.DEFINE_string("data_path", None,
                            "Where the training/test data is stored.")
tf.app.flags.DEFINE_integer("batch_size", 32,
                            "batch size (default is 32).")

#filename_queue = tf.train.string_input_producer([ FLAGS.data_path + "/data.csv"])
print("loading model")
#model = KeyedVectors.load_word2vec_format('../word2vec-GoogleNews-vectors/GoogleNews-vectors-negative300.bin', binary=True)
model={}
batch_size = 32
label_ref = {'agree': 0, 'disagree': 1, 'discuss': 2, 'unrelated': 3}
#g = tf.Graph()
#with g.as_default():
filenames_train = [FLAGS.data_path + "/train_{}.tfrecords".format(i) for i in range(83)]
filenames_test = [FLAGS.data_path + "/test_{}.tfrecords".format(i) for i in range(10)]
embedding_dim = 300

def parse_serialized(serialized_example):
    context_features={
        'head_len': tf.FixedLenFeature([], tf.int64),
        'body_len': tf.FixedLenFeature([], tf.int64),
        'label': tf.FixedLenFeature([], tf.int64)
        }
    sequence_features = {
            "head": tf.FixedLenSequenceFeature([], dtype=tf.float32),
            "body": tf.FixedLenSequenceFeature([], dtype=tf.float32)
            }
    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
            serialized=serialized_example,
            context_features=context_features,
            sequence_features=sequence_features,
            name="basic_parsing"
            )
    
    head_len = context_parsed['head_len']
    raw_head = sequence_parsed['head']
    
    head = tf.reshape(raw_head,[-1, embedding_dim])
    body_len = context_parsed['body_len']
    body = tf.reshape(sequence_parsed['body'],[-1, embedding_dim])
    label = context_parsed['label']
    
    input_tensors = [head, head_len, body, body_len, label]
    names=["head", "head_len","body", "body_len", "label"]
    in_dict = dict(zip(names, input_tensors))
    return in_dict, label

def read_file_queue(filename_queue):
    reader = tf.TFRecordReader(name="TF_record_reader")
    _, serialized_example = reader.read(filename_queue, name="read_op")



def input_pipeline(filenames, batch_size, num_epochs=None):
    filename_queue = tf.train.string_input_producer(
                   filenames, num_epochs=num_epochs, shuffle=True)
    in_dict, label = read_file_queue(filename_queue)
    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * batch_size
    
    batched_data = tf.train.batch(
            tensors=in_dict,
            batch_size=batch_size,
            capacity=capacity,
            dynamic_pad=True,
            shapes=[[None, embedding_dim], [], [None, embedding_dim], [], []], name="my_padding_queue")
    return batched_data, batched_data["label"]



def dataset_input_fn(filenames, repeat):
    dataset = tf.contrib.data.TFRecordDataset(filenames)

    def parse_serialized(serialized_example):
        context_features={
            'head_len': tf.FixedLenFeature([], tf.int64),
            'body_len': tf.FixedLenFeature([], tf.int64),
            'label': tf.FixedLenFeature([], tf.int64)
            }
        sequence_features = {
                "head": tf.FixedLenSequenceFeature([], dtype=tf.float32),
                "body": tf.FixedLenSequenceFeature([], dtype=tf.float32)
                }
        context_parsed, sequence_parsed = tf.parse_single_sequence_example(
                serialized=serialized_example,
                context_features=context_features,
                sequence_features=sequence_features,
                name="basic_parsing"
                )
        
        head_len = context_parsed['head_len']
        raw_head = sequence_parsed['head']
        head = tf.reshape(raw_head,[-1, embedding_dim])
        body_len = context_parsed['body_len']
        body = tf.reshape(sequence_parsed['body'],[-1, embedding_dim])
        label = context_parsed['label']
        
        input_tensors = [head, head_len, body, body_len]
        names=["head", "head_len","body", "body_len"]
        in_dict = dict(zip(names, input_tensors))
        return in_dict, label

    dataset = dataset.map(parse_serialized)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.padded_batch(batch_size=FLAGS.batch_size,
            padded_shapes=({
                'body': TensorShape([Dimension(None), Dimension(300)]), 
                'body_len': TensorShape([]), 
                'head': TensorShape([Dimension(None), Dimension(300)]), 
                'head_len': TensorShape([])}, 
                TensorShape([]))
            )
    dataset = dataset.repeat(repeat)
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()

    return features, labels

def data_type():
    return tf.float16 if FLAGS.use_fp16 else tf.float32

def extract_axis_1(data, ind):
    """
    Get specified elements along the first axis of tensor.
    :param data: Tensorflow tensor that will be subsetted.
    :param ind: Indices to take (one for each element along axis 0 of data).
    :return: Subsetted tensor.
    """

    batch_range = tf.range(tf.cast(tf.shape(data)[0], ind.dtype), dtype=ind.dtype)
    indices = tf.stack([batch_range, ind], axis=1)
    res = tf.gather_nd(data, indices)
    return res

def conv_net_model_fn(features, labels, mode):
    input_head = features["head"]
    input_body = features["body"]
    seq_len_head = features["head_len"]
    seq_len_body = features["body_len"]
    num_of_outputs = 4
    num_of_filters = 256
    is_training = (mode == learn.ModeKeys.TRAIN)
 
    def conv_dropout_pool(input, filters, name, reuse, pool=False):
        conv = tf.layers.conv1d(inputs=input, filters=filters,
                kernel_size=3, activation=tf.nn.relu, name=name+"_conv", reuse=reuse)
        dropout = tf.layers.dropout(inputs=conv, training=is_training, name=name+"_dropout")
        if pool:
            return tf.layers.max_pooling1d(inputs=dropout, pool_size=2, strides=2, name=name+"_pool")
        else:
            return dropout

    def conv_layers(input, filters, reuse):
        l1 = conv_dropout_pool(input, filters, "layer_1", reuse, True)
        l2 = conv_dropout_pool(l1, filters, "layer_2", reuse, True)
        l3 = conv_dropout_pool(l2, filters*2, "layer_3", reuse, True)
        l4 = conv_dropout_pool(l2, filters*2, "layer_4", reuse)
        l5 = conv_dropout_pool(l2, filters*3, "layer_5", reuse)
        return l5

    body_conv = conv_layers(input_body, num_of_filters, False)
    head_conv = conv_layers(input_head, num_of_filters, True)
    max_body = tf.reduce_max(body_conv, axis=1)
    max_head = tf.reduce_max(head_conv, axis=1)

    full = tf.concat([max_body, max_head], axis=1)

    dense_1 = tf.layers.dense(inputs=full, units=1024, activation=tf.nn.relu)
    dense_2 = tf.layers.dense(inputs=dense_1, units=1024, activation=tf.nn.relu)
    dense_3 = tf.layers.dense(inputs=dense_2, units=1024, activation=tf.nn.relu)
    logits = tf.layers.dense(inputs=dense_3, units=num_of_outputs, activation=None)

    loss = None
    train_op = None
    if mode != learn.ModeKeys.INFER:
        loss = tf.losses.sparse_softmax_cross_entropy(
                labels=labels,logits=logits
                )
        tf.summary.scalar("summary_loss", loss)
    if mode == learn.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(
                loss=loss,
                global_step=tf.contrib.framework.get_global_step(),
                learning_rate=0.001,
                optimizer="Adam")
    softmax = tf.nn.softmax(logits, name="softmax_tensor")
    classes = tf.argmax(input=logits, axis=1)
    predictions = {
            "classes": classes,
            "probabilities": softmax
            }
    tf.summary.merge_all()
    eval_metric_ops = {
            "Accuracy": tf.metrics.accuracy(labels, classes)
            }
    return learn.ModelFnOps(
            mode=mode, 
            predictions=predictions, 
            loss=loss, 
            train_op=train_op,
            eval_metric_ops=eval_metric_ops)
  

def conditional_embedding_model_fn(features, labels, mode):#, params):
    input_head = features["head"]
    input_body = features["body"]
    seq_len_head = features["head_len"]
    seq_len_body = features["body_len"]
    num_of_outputs = 4
    
    with tf.variable_scope('head'):
        head_cell = tf.nn.rnn_cell.LSTMCell(600)
        outputs_head, final_state_head = tf.nn.dynamic_rnn(
                cell=head_cell, inputs=input_head, 
                sequence_length=seq_len_head, dtype=data_type()
                )

    with tf.variable_scope('body'):
        body_cell = tf.nn.rnn_cell.LSTMCell(600)
        outputs_body, final_state_body = tf.nn.dynamic_rnn(
                cell=body_cell, inputs=input_body, 
                sequence_length=seq_len_body, dtype=data_type(),
                initial_state=final_state_head
                )

    output_embedding = extract_axis_1(outputs_body, seq_len_body)
    logits = tf.layers.dense(inputs=output_embedding, units=num_of_outputs)

    loss = None
    train_op = None
    if mode != learn.ModeKeys.INFER:
        loss = tf.losses.sparse_softmax_cross_entropy(
                labels=labels,logits=logits
                )
        tf.summary.scalar("summary_loss", loss)
    if mode == learn.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(
                loss=loss,
                global_step=tf.contrib.framework.get_global_step(),
                learning_rate=0.001,
                optimizer="Adam")
    softmax = tf.nn.softmax(logits, name="softmax_tensor")
    classes = tf.argmax(input=logits, axis=1)
    predictions = {
            "classes": classes,
            "probabilities": softmax
            }
    tf.summary.merge_all()
    eval_metric_ops = {
            "Accuracy": tf.metrics.accuracy(labels, classes)
            }
    return learn.ModelFnOps(
            mode=mode, 
            predictions=predictions, 
            loss=loss, 
            train_op=train_op,
            eval_metric_ops=eval_metric_ops)

train_input_fn = functools.partial(dataset_input_fn ,filenames=filenames_train, repeat=-1)
eval_input_fn = functools.partial(dataset_input_fn ,filenames=filenames_test, repeat=1)

print("starting to fit")
tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(
              tensors=tensors_to_log, every_n_iter=10)
#a = tf.summary.merge_all()
#summarySaverHook = tf.train.SummarySaverHook(save_secs=2, output_dir=FLAGS.log_dir, scaffold=tf.train.Scaffold(), summary_op=tf.summary.merge_all())
#hooks = [tf_debug.LocalCLIDebugHook()]

#validation_monitor = tf.contrib.learn.monitors.ValidationMonitor( test_set.data, test_set.target, every_n_steps=50)



#classifier = tf.estimator.Estimator(
#classifier = learn.Estimator(model_fn=conditional_embedding_model_fn, model_dir=FLAGS.save_path)
classifier = learn.Estimator(model_fn=conv_net_model_fn, model_dir=FLAGS.save_path)
#classifier.train(input_fn=train_input_fn, steps=20000, hooks=hooks)
experiment = learn.Experiment(estimator=classifier, train_input_fn=train_input_fn, eval_input_fn=eval_input_fn, eval_hooks=[logging_hook])#, train_steps_per_iteration=10)
experiment.continuous_train_and_eval()
#experiment.train_and_evaluate()
"""
class ConditionalModel(object):

  def __init__(self, batch_size, hidden_size, embedding_size, num_of_outputs):

    self.batch_size = batch_size
    self.num_steps = num_steps
    size = hidden_size
    self.embedding_size = embedding_size
    self.input_head = tf.placeholder(data_type(), shape=(batch_size, None, self.embedding_size))
    self.input_body = tf.placeholder(data_type(), shape=(batch_size, None, self.embedding_size))
    self.seq_len_head = tf.placeholder(tf.int32 , shape=(batch_size,))
    self.seq_len_body = tf.placeholder(tf.int32 , shape=(batch_size,))
    self.labels = tf.placeholder(tf.int32 , shape=(batch_size,))

    head_cell = LSTMCell()
    body_cell = LSTMCell()

    self.outputs_head, self.final_state_head = self.head_rnn = tf.nn.dynamic_rnn(
            cell=head_cell, inputs=self.input_head, 
            sequence_length=self.seq_len_head, dtype=data_type()
            )


    self.outputs_body, self.final_state_body = tf.nn.dynamic_rnn(
            cell=body_cell, inputs=self.input_body, 
            sequence_length=self.seq_len_body, dtype=data_type(),
            initial_state=self.final_state_head
            )

    self.output_embedding = extract_axis_1(self.outputs_body, self.seq_len_body)
    self.logits = tf.layers.dense(inputs=self.output_embedding, units=num_of_outputs)
    self.softmax = tf.nn.softmax(self.logits)
    self.loss = tf.nn,sparse_softmax_cross_entropy_with_logits(
            labels=self.labels,logits=self.output
            )

    self._train_op = tf.contrib.layers.optimize_loss(
            loss=loss, global_step=tf.contrib.framework.get_global_step(),
            learning_rate=0.001, optimizer="Adam"
            )

    self._new_lr = tf.placeholder(
        tf.float32, shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(self._lr, self._new_lr)

  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op
"""

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


if __name__ == "__main__":
    tf.app.run()
