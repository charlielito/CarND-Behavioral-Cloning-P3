import tensorflow as tf

### Aux functions for batchnorm layers
def conv2d_batch_norm(*args, **kwargs):

    activation = kwargs.pop("activation", None)
    batch_norm = kwargs.pop("batch_norm", {})

    net = tf.layers.conv2d(*args, **kwargs)
    net = tf.layers.batch_normalization(net, **batch_norm)

    return activation(net) if activation else net

def dense_batch_norm(*args, **kwargs):

    activation = kwargs.pop("activation", None)
    batch_norm = kwargs.pop("batch_norm", {})

    net = tf.layers.dense(*args, **kwargs)
    net = tf.layers.batch_normalization(net, **batch_norm)

    return activation(net) if activation else net


def model(images, mode, params):

    conv_dropout = params.pop("conv_droput")
    dense_dropout = params.pop("dense_dropout")
    final_height = params.pop("height")
    kernel_init = params.pop("kernel_init", None)

    # Input Layer
    input_layer = tf.reshape(images, [-1, final_height, 320, 3])

    general_ops = dict(
#         activation = activation,
        batch_norm = dict(training = mode == tf.estimator.ModeKeys.TRAIN)
#         kernel_regularizer = tf.contrib.layers.l2_regularizer(l2_regularization),
    )
                                             
    input_normalized = tf.layers.batch_normalization(input_layer, training = mode == tf.estimator.ModeKeys.TRAIN)
    
    # Convolutional Layer #1
    conv1 = conv2d_batch_norm(
      inputs=input_normalized,
      filters=32,
      kernel_size=[5, 5],
      padding="valid",
      activation=tf.nn.relu, kernel_initializer=kernel_init,
      **general_ops)
    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    pool1 = tf.layers.dropout(
        inputs=pool1, 
        rate=conv_dropout, 
        training= mode == tf.estimator.ModeKeys.TRAIN)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = conv2d_batch_norm(
      inputs=pool1,
      filters=32,
      kernel_size=[3, 3],
      padding="valid",
      activation=tf.nn.relu, kernel_initializer=kernel_init,
    **general_ops)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    pool2 = tf.layers.dropout(
        inputs=pool2, 
        rate=conv_dropout, 
        training= mode == tf.estimator.ModeKeys.TRAIN)
                                             
    # Convolutional Layer #3 and Pooling Layer #3
    conv3 = conv2d_batch_norm(
      inputs=pool2,
      filters=64,
      kernel_size=[3, 3],
      padding="valid",
      activation=tf.nn.relu, kernel_initializer=kernel_init,
    **general_ops)
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
    pool3 = tf.layers.dropout(
        inputs=pool3, 
        rate=conv_dropout, 
        training= mode == tf.estimator.ModeKeys.TRAIN)  
                                             
    # Convolutional Layer #4 and Pooling Layer #4
    conv4 = conv2d_batch_norm(
      inputs=pool3,
      filters=64,
      kernel_size=[3, 3],
      padding="valid",
      activation=tf.nn.relu, kernel_initializer=kernel_init,
    **general_ops)
    pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)
    pool4 = tf.layers.dropout(
        inputs=pool4, 
        rate=conv_dropout, 
        training= mode == tf.estimator.ModeKeys.TRAIN)  
                                             
    # Dense Layers
    pool_flat = tf.layers.flatten(conv4)
    
    dense1 = dense_batch_norm(inputs=pool_flat,
                             units=1024,
                             activation=tf.nn.relu,
                             kernel_initializer=kernel_init,
                             **general_ops)
    dense1 = tf.layers.dropout(
        inputs=dense1, 
        rate=dense_dropout, 
        training= mode == tf.estimator.ModeKeys.TRAIN)
    
    dense2 = dense_batch_norm(inputs=dense1,
                             units=512,
                             activation=tf.nn.relu,
                             kernel_initializer=kernel_init,
                             **general_ops)
    dense2 = tf.layers.dropout(
        inputs=dense2, 
        rate=dense_dropout, 
        training= mode == tf.estimator.ModeKeys.TRAIN)
                                             
    dense3 = dense_batch_norm(inputs=dense2,
                             units=128,
                             activation=tf.nn.relu,
                             kernel_initializer=kernel_init,
                             **general_ops)
    dense3 = tf.layers.dropout(
        inputs=dense3, 
        rate=dense_dropout, 
        training= mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    preds = tf.layers.dense(inputs=dense3, units=1, kernel_initializer=kernel_init, use_bias=False)

    predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "steering": preds,
    "image": input_layer
    }

    return predictions