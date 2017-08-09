from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from gensim.models.keyedvectors import KeyedVectors
from nltk import word_tokenize
from csv import DictReader
import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("data_path", None,
                            "Where the training/test data is stored.")
print("loading model")
model = KeyedVectors.load_word2vec_format('../word2vec-GoogleNews-vectors/GoogleNews-vectors-negative300.bin', binary=True)

print("model loaded")



label_ref_rev = {0: 'agree', 1: 'disagree', 2: 'discuss', 3: 'unrelated'}
label_ref = {'agree': 0, 'disagree': 1, 'discuss': 2, 'unrelated': 3}

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _floatlist_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def convert_to(csv_file, record_file):
    filename = os.path.join(FLAGS.data_path, record_file + '.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    with open(csv_file, "r", encoding='utf-8') as table:
        r = DictReader(table)
        for line in r:
            label = label_ref[line['stance']]
            ref = np.array([model[x] for x in word_tokenize(line["head"]) if x in model])
            body = np.array([model[x] for x in word_tokenize(line["body"]) if x in model])
            ref_shape = list(ref.shape)
            body_shape = list(body.shape)
            """
            example = tf.train.Example(features=tf.train.Features(feature={
                'head': _floatlist_feature(ref.flatten()),
                'body': _floatlist_feature(body.flatten()),
                'head_shape': _int64list_feature(ref_shape),
                'body_shape': _int64list_feature(body_shape),
                'label': _int64_feature(label)}))
            """
            ex = tf.train.SequenceExample()
            ex.context.feature['head_len'].int64_list.value.append(ref.shape[0])
            ex.context.feature['body_len'].int64_list.value.append(body.shape[0])
            ex.context.feature['label'].int64_list.value.append(label)
            fl_head = ex.feature_lists.feature_list["head"]
            fl_body = ex.feature_lists.feature_list["body"]
            for e in ref.flatten():
                fl_head.feature.add().float_list.value.append(e)
            for e in body.flatten():
                fl_body.feature.add().float_list.value.append(e)
            writer.write(ex.SerializeToString())
    writer.close()

def main(argv):
    csv_train = os.path.join(FLAGS.data_path, 'traindata.csv')
    csv_test = os.path.join(FLAGS.data_path, 'traindata.csv')
    convert_to(csv_train, 'train')
    convert_to(csv_test, 'test')

if __name__ == "__main__":
    tf.app.run(main=main)
