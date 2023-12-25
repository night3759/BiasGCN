import tensorflow as tf

def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def masked_multiclass_margin_loss(preds, labels, mask):
    labels = tf.cast(labels, dtype=tf.bool)
    mask = tf.cast(mask, dtype=tf.bool)
    mask = tf.squeeze(mask)
    mask.set_shape([None])
    preds = tf.boolean_mask(preds, mask)
    labels = tf.boolean_mask(labels, mask)
    preds_labels = preds[labels]
    preds_max = tf.nn.relu(preds - preds_labels[:, tf.newaxis])
    preds_max = tf.reduce_max(preds_max, axis=-1)
    loss = tf.tanh(preds_max * 10)
    mask = tf.cast(mask, dtype=tf.float32)
    # mask /= tf.reduce_mean(mask)
    # loss *= mask
    return tf.reduce_mean(loss)


def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)


