from __future__ import division
from __future__ import print_function


import time
import tensorflow.compat.v1 as tf
import scipy.sparse as sp
import numpy as np
import os
from utils import *
from models import GCN
from gat import GAT

import pickle

import os
os.environ['CUDA_VISIBLE_DEVICES']='0'


# Set random seed
seed = 666
np.random.seed(seed)
tf.set_random_seed(seed)


# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 400, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 50, 'Tolerance for early stopping (# of epochs).')


# Load data
# adj, features, y_train_ori, y_val_ori, y_test_ori, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)
# outputs = np.load('outputs.npy')
# labels = np.zeros_like(y_test_ori)
# labels[np.arange(labels.shape[0]), outputs] = 1
labels = np.load('output/label_cora_gam.npy')
# labels = np.load('./test.npy')

print(labels)

y_train = np.zeros(labels.shape)
y_val = np.zeros(labels.shape)
y_test = np.zeros(labels.shape)
y_train[train_mask, :] = labels[train_mask, :]
y_val[val_mask, :] = labels[val_mask, :]
y_test[test_mask, :] = labels[test_mask, :]

total_edges = adj.data.shape[0]
n_node = adj.shape[0]
# Some preprocessing
features = preprocess_features(features)
# for non sparse
features = sp.coo_matrix((features[1],(features[0][:,0],features[0][:,1])),shape=features[2]).toarray()

# support = preprocess_adj(adj)
support = preprocess_adj(adj)
# for non sparse
support = [sp.coo_matrix((support[1],(support[0][:,0],support[0][:,1])),shape=support[2]).toarray()]
adj = [adj.toarray()]
num_supports = 1
model_func = GCN

bias = np.array(bl(adj[0]), dtype='float32')
# bias = np.array(np.ones(features.shape[0]), dtype='float32')

save_name = 'nat_'+FLAGS.dataset+"_gam"
# save_name = 'test'+FLAGS.dataset
if not os.path.exists(save_name):
   os.makedirs(save_name)
# Define placeholders
placeholders = {
    's': [tf.sparse_placeholder(tf.float32, shape=(n_node,n_node)) for _ in range(num_supports)],
    'adj': [tf.sparse_placeholder(tf.float32, shape=(n_node,n_node)) for _ in range(num_supports)], 
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'lmd': tf.placeholder(tf.float32),
    'mu': tf.placeholder(tf.float32),
    's': [tf.placeholder(tf.float32, shape=(n_node,n_node)) for _ in range(num_supports)],
    'adj': [tf.placeholder(tf.float32, shape=(n_node,n_node)) for _ in range(num_supports)], 
    'support': [tf.placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.placeholder(tf.float32, shape=features.shape),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}

# Create model
# for non sparse
model = model_func(placeholders, input_dim=features.shape[1], bias=bias, attack=None, logging=False)

# Initialize session
sess = tf.Session()


# Define model evaluation function
def evaluate(features, support, adj, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, adj, labels, mask, placeholders, train=True)
    feed_dict_val.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    outs_val = sess.run([model.attack_loss, model.accuracy, model.outputs], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test), outs_val[2]


# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []
# Train model
for epoch in range(FLAGS.epochs):

   t = time.time()
   # Construct feed dictionary
   feed_dict = construct_feed_dict(features, support, adj, y_train, train_mask, placeholders, train=True)
   feed_dict.update({placeholders['dropout']: FLAGS.dropout})

   # Training step
   outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

   # Validation
   cost, acc, duration, _ = evaluate(features, support, adj, y_val, val_mask, placeholders)
   cost_val.append(cost)

   # Print results
   print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
         "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
         "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

   if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
       print("Early stopping...")
       break

print("Optimization Finished!")

# Testing before attack
test_cost, test_acc, test_duration, save_label = evaluate(features, support, adj, y_test, test_mask, placeholders)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
     "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))

x = sess.run(model.B, feed_dict=feed_dict)
print(x.min())

save_label = np.argmax(save_label,1)
tmp = np.zeros_like(y_train)
tmp[np.arange(len(save_label)), save_label] = 1
tmp = y_train + tmp * (1-np.expand_dims(train_mask,1))
np.save('label_'+ FLAGS.dataset + "_gam"  + '.npy',tmp)
# np.save('label_test'  + '.npy',tmp)
print('predicted label saved at '+'label_'+ FLAGS.dataset + '.npy')
model.save(sess, save_name+'/'+save_name)
