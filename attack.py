from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import copy
import numpy as np
import os

from utils import *
from models import GCN, MLP

import pickle

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Set random seed
seed = 66
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('model_dir', 'nat_citeseer', 'saved model directory')
flags.DEFINE_string('dataset', 'citeseer', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_integer('steps', 100, 'Number of steps to attack')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of steps).')
flags.DEFINE_string('method', 'PGD', 'attack method, PGD or CW')
flags.DEFINE_float('perturb_ratio', 0.05, 'perturb ratio of total edges.')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')

# Load data
adj, features, y_train_ori, y_val_ori, y_test_ori, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)
outputs = np.load('outputs.npy')
labels = np.zeros_like(y_test_ori)
labels[np.arange(labels.shape[0]), outputs] = 1
y_train = np.zeros(labels.shape)
y_val = np.zeros(labels.shape)
y_test = np.zeros(labels.shape)
y_train[train_mask, :] = labels[train_mask, :]
y_val[val_mask, :] = labels[val_mask, :]
y_test[test_mask, :] = labels[test_mask, :]

total_edges = adj.data.shape[0]/2
n_node = adj.shape[0]

# Some preprocessing
features = preprocess_features(features)
# for non sparse
features = sp.coo_matrix((features[1], (features[0][:, 0], features[0][:, 1])), shape=features[2]).toarray()

support = preprocess_adj_gl(adj)
# support = preprocess_adj(adj)
# for non sparse
# support = sparse_to_tuple(adj + sp.eye(adj.shape[0]))
support = [sp.coo_matrix((support[1], (support[0][:, 0], support[0][:, 1])), shape=support[2]).toarray()]
num_supports = 1
model_func = GCN
bias = np.array(bl(adj.toarray()), dtype='float32')

# Define placeholders
placeholders = {
    'lmd': tf.placeholder(tf.float32),
    'mu': tf.placeholder(tf.float32),
    's': [tf.placeholder(tf.float32, shape=(n_node, n_node)) for _ in range(num_supports)],
    'adj': [tf.placeholder(tf.float32, shape=(n_node, n_node)) for _ in range(num_supports)],
    'support': [tf.placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.placeholder(tf.float32, shape=features.shape),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'label_mask_expand': tf.placeholder(tf.float32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}

# Create model
# for non sparse
model = model_func(placeholders, input_dim=features.shape[1], bias=bias, attack=FLAGS.method, logging=False)
# model = model_func(placeholders, input_dim=features.shape[1], attack='minmax', logging=False)

# Initialize session
sess = tf.Session()


# Define model evaluation function
def evaluate(features, support, adj, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, adj, labels, mask, placeholders)
    feed_dict_val.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    outs_val = sess.run([model.attack_loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)


# Init variables
sess.run(tf.global_variables_initializer())

model.load(FLAGS.model_dir, sess)
adj = adj.toarray()
lmd = 1
eps = total_edges * FLAGS.perturb_ratio
xi = 1e-5

## results before attack
test_cost, test_acc, test_duration = evaluate(features, support, adj, y_train, train_mask, placeholders)
print("Train set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
test_cost, test_acc, test_duration = evaluate(features, support, adj, y_val, val_mask, placeholders)
print("Validation set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
test_cost, test_acc, test_duration = evaluate(features, support, adj, y_test, test_mask, placeholders)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))


label = y_train
label_mask = train_mask + test_mask

original_support = copy.deepcopy(support)
feed_dict = construct_feed_dict(features, support, adj, label, label_mask, placeholders)
feed_dict.update({placeholders['lmd']: lmd})
feed_dict.update({placeholders['dropout']: FLAGS.dropout})
feed_dict.update({placeholders['adj'][i]: adj for i in range(num_supports)})
# feed_dict.update({placeholders['s'][i]: np.random.uniform(size=(n_node,n_node)) for i in range(num_supports)})
feed_dict.update({placeholders['s'][i]: np.zeros([n_node, n_node]) for i in range(num_supports)})

x1 = sess.run(model.B, feed_dict=feed_dict)

if FLAGS.method == 'CW':
    label_mask_expand = np.tile(label_mask, [label.shape[1],1]).transpose()
    feed_dict.update({placeholders['label_mask_expand']: label_mask_expand})
    C = 0.1
else:
    C = 200  # initial learning rate

if os.path.exists('label_' + FLAGS.dataset + '.npy'):
    label = np.load('label_' + FLAGS.dataset + '.npy')
else:
    ret = sess.run(model.outputs, feed_dict=feed_dict)
    ret = np.argmax(ret, 1)
    label = np.zeros_like(label)
    label[np.arange(label.shape[0]), ret] = 1
    np.save('label_' + FLAGS.dataset + '.npy', label)
feed_dict.update({placeholders['labels']: label})

print('{} attack begin:'.format(FLAGS.method))
for epoch in range(FLAGS.steps):

    t = time.time()
    # mu = C/np.sqrt(np.sqrt(epoch+1))
    mu = C / np.sqrt(epoch + 1)
    feed_dict.update({placeholders['mu']: mu})

    # modified_A = sess.run(model.modified_A, feed_dict=feed_dict)
    # hat_A = sess.run(model.hat_A, feed_dict=feed_dict)
    # rowsum = sess.run(model.rowsum, feed_dict=feed_dict)
    # d_sqrt = sess.run(model.d_sqrt, feed_dict=feed_dict)
    # d_sqrt_inv = sess.run(model.d_sqrt_inv, feed_dict=feed_dict)
    # W = sess.run(model.W, feed_dict=feed_dict)
    # d_W = sess.run(model.d_W, feed_dict=feed_dict)
    # d_W_sqrt = sess.run(model.d_W_sqrt, feed_dict=feed_dict)
    # d_W_sqrt_inv = sess.run(model.d_W_sqrt_inv, feed_dict=feed_dict)
    # laplcian = sess.run(model.laplacian, feed_dict=feed_dict)
    # support_real = sess.run(model.support_real, feed_dict=feed_dict)


    # s \in [0,1]
    if FLAGS.method == 'CW':
        a, support, l, g = sess.run([model.a, model.placeholders['support'], model.loss, model.Sgrad],
                                    feed_dict=feed_dict)
        # print('loss:', l)
    elif FLAGS.method == 'PGD':
        a, support, S, g = sess.run([model.a, model.placeholders['support'], model.upper_S_real, model.Sgrad],
                                    feed_dict=feed_dict)
    else:
        raise ValueError('invalid attack method: {}'.format(FLAGS.method))
    upper_S_update = bisection(a, eps, xi)

    feed_dict.update({placeholders['s'][i]: upper_S_update[i] for i in range(num_supports)})

    upper_S_update_tmp = upper_S_update[:]
    if epoch == FLAGS.steps - 1:
        acc_record, support_record, p_ratio_record = [], [], []
        for i in range(10):
            print('random start!')
            randm = np.random.uniform(size=(n_node, n_node))
            upper_S_update = np.where(upper_S_update_tmp > randm, 1, 0)
            feed_dict.update({placeholders['s'][i]: upper_S_update[i] for i in range(num_supports)})

            # modified_A = sess.run(model.modified_A, feed_dict=feed_dict)
            # hat_A = sess.run(model.hat_A, feed_dict=feed_dict)
            # rowsum = sess.run(model.rowsum, feed_dict=feed_dict)
            # d_sqrt = sess.run(model.d_sqrt, feed_dict=feed_dict)
            # d_sqrt_inv = sess.run(model.d_sqrt_inv, feed_dict=feed_dict)
            # W = sess.run(model.W, feed_dict=feed_dict)
            # d_W = sess.run(model.d_W, feed_dict=feed_dict)
            # d_W_sqrt = sess.run(model.d_W_sqrt, feed_dict=feed_dict)
            # d_W_sqrt_inv = sess.run(model.d_W_sqrt_inv, feed_dict=feed_dict)
            # laplcian = sess.run(model.laplacian, feed_dict=feed_dict)
            # support_real = sess.run(model.support_real, feed_dict=feed_dict)

            support = sess.run(model.placeholders['support'], feed_dict=feed_dict)
            cost, acc, duration = evaluate(features, support, adj, y_test, test_mask, placeholders)
            pr = np.count_nonzero(upper_S_update[0]) / total_edges
            if pr <= FLAGS.perturb_ratio:
                acc_record.append(acc)
                support_record.append(support)
                p_ratio_record.append(pr)
            print("Epoch:", '%04d' % (epoch + 1), "test_loss=", "{:.5f}".format(cost),
                  "test_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))
            print("perturb ratio", pr)
            print('random end!')
        # Validation
        support = support_record[np.argmin(np.array(acc_record))]
    cost, acc, duration = evaluate(features, support, adj, y_test, test_mask, placeholders)
    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "test_loss=", "{:.5f}".format(cost),
          "test_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

print("attack Finished!")
print("perturb ratio", np.count_nonzero(upper_S_update[0]) / total_edges)

#feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)




# Testing after attack

test_cost, test_acc, test_duration = evaluate(features, support, adj, y_test_ori, test_mask, placeholders)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))

modified_A = sess.run(model.hat_A[0], feed_dict=feed_dict)

with open("support.pkl", "wb") as s:
    pickle.dump(modified_A, s)
sess.close()
