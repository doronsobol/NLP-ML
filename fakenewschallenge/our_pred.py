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
tf.app.flags.DEFINE_string("data_path", None,
                            "Where the training/test data is stored.")

#filename_queue = tf.train.string_input_producer([ FLAGS.data_path + "/data.csv"])
print("loading model")
#model = KeyedVectors.load_word2vec_format('../word2vec-GoogleNews-vectors/GoogleNews-vectors-negative300.bin', binary=True)
model={}
batch_size = 32
label_ref = {'agree': 0, 'disagree': 1, 'discuss': 2, 'unrelated': 3}
#g = tf.Graph()
#with g.as_default():
filenames_train = [FLAGS.data_path + "/train_{}.tfrecords".format(i) for i in range(3)]
filenames_test = [FLAGS.data_path + "/test_{}.tfrecords".format(i) for i in range(3)]
embedding_dim = 300

def read_file_queue(filename_queue):
    reader = tf.TFRecordReader(name="TF_record_reader")
    _, serialized_example = reader.read(filename_queue, name="read_op")
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
    
    #context = tf.contrib.learn.run_n(context_parsed, n=1, feed_dict=None)
    #print(context)
    #sequence = tf.contrib.learn.run_n(sequence_parsed, n=1, feed_dict=None)
    #print(sequence[0])
    head_len = context_parsed['head_len']
    raw_head = sequence_parsed['head']
    #raw_head = tf.Print(raw_head,[ raw_head ])
    #return raw_head_p
    #tf.contrib.learn.run_n(raw_head_p, n=1, feed_dict=None)
    #print(pppp)
    head = tf.reshape(raw_head,[-1, embedding_dim])
    body_len = context_parsed['body_len']
    body = tf.reshape(sequence_parsed['body'],[-1, embedding_dim])
    label = context_parsed['label']
    return head, head_len, body, body_len, label



def input_pipeline(filenames, batch_size, num_epochs=None):
    filename_queue = tf.train.string_input_producer(
                   filenames, num_epochs=num_epochs, shuffle=True)
    head, head_len, body, body_len, label = read_file_queue(filename_queue)
    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * batch_size
    input_tensors = [head, head_len, body, body_len, label]
    names=["head", "head_len","body", "body_len", "label"]
    in_dict = dict(zip(names, input_tensors))
    #tf.PaddingFIFOQueue(capacity=capacity,
    batched_data = tf.train.batch(
            tensors=in_dict,
            batch_size=batch_size,
            capacity=capacity,
            #dtypes=[tf.float32, tf.int64, tf.float32, tf.int64, tf.int64],
            dynamic_pad=True,
            shapes=[[None, embedding_dim], [], [None, embedding_dim], [], []], name="my_padding_queue")
    return batched_data, batched_data["label"]

#filename_queue = tf.train.string_input_producer(filenames_train[:4], num_epochs=None, shuffle=True)
#sess = tf.InteractiveSession()
#sess.run(tf.global_variables_initializer())
#tf.train.start_queue_runners(sess=sess)
#sess = tf_debug.LocalCLIDebugWrapperSession(sess, ui_type="curses")
#sess.run(read_file_queue(filename_queue))
"""
def input_fn():
    return_dict = {}
    data_body = []
    data_head = []
    head_len = []
    body_len = []
    labels = []
    while len(body_len) < batch_size:
        for i in r:
            data_body.append(np.array([model[x] for x in word_tokenize(i["body"]) if x in model]))
            data_head.append(np.array([model[x] for x in word_tokenize(i["head"]) if x in model]))
            body_len.append(data_body[-1].shape[0])
            head_len.append(data_head[-1].shape[0])
            labels.append(label_ref[i["stance"]])
            if len(body_len) >= batch_size:
                break
        if len(body_len) >= batch_size:
            break
        table = open(filename, "r", encoding='utf-8')
        r = DictReader(table)

    body_len = np.array(body_len)
    head_len = np.array(head_len)
    emmbedding_size = data_body[0].shape[1]
    maxlen_body = np.amax(body_len)
    maxlen_head = np.amax(head_len)

    for i, b in enumerate(data_body):
        data_body[i] = np.lib.pad(b, [(0,maxlen_body-b.shape[0]),(0,0)], 'constant', constant_values=0)
    data_body_np = np.stack(data_body)
    for i, b in enumerate(data_head):
        data_head[i] = np.lib.pad(b, [(0,maxlen_head-b.shape[0]),(0,0)], 'constant', constant_values=0)
    data_head_np = np.stack(data_head)
    return_dict["input_head"] = tf.constant(data_head_np)
    return_dict["input_body"] = tf.constant(data_body_np)
    return_dict["head_len"] = tf.constant(head_len)
    return_dict["body_len"] = tf.constant(body_len)
    labels = tf.constant(labels)
    print(return_dict["input_head"].shape)
    return return_dict, labels
"""    

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



def conditional_embedding_model_fn(mode, features, labels):#, params):
    #hidden_size = params["hidden_size"]
    input_head = features["head"]
    input_body = features["body"]
    seq_len_head = features["head_len"]
    seq_len_body = features["body_len"]
    num_of_outputs = 4
    #head_cell = tf.contrib.rnn.LSTMCell(600)
    #body_cell = tf.contrib.rnn.LSTMCell(600)
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
    if mode == learn.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(
                loss=loss,
                global_step=tf.contrib.framework.get_global_step(),
                learning_rate=0.001,
                optimizer="Adam")
    predictions = {
            "classes": tf.argmax(
                input=logits, axis=1),
            "probabilities": tf.nn.softmax(
                logits, name="softmax_tensor")
            }
    return tf.estimator.EstimatorSpec(
            mode=mode, predictions=predictions, loss=loss, train_op=train_op)

train_input_fn = functools.partial(input_pipeline ,filenames=filenames_train, batch_size=32)
print("starting to fit")
tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(
              tensors=tensors_to_log, every_n_iter=50)
hooks = [tf_debug.LocalCLIDebugHook()]

classifier = tf.estimator.Estimator(
              model_fn=conditional_embedding_model_fn, model_dir=FLAGS.save_path)
metrics = { 
        "accuracy": tf.contrib.learn.MetricSpec(metric_fn=tf.metrics.accuracy, prediction_key="classes"),
        }
#classifier.train(input_fn=train_input_fn, steps=20000, hooks=hooks)
classifier.train(input_fn=train_input_fn, steps=20000, hooks=[logging_hook])
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
