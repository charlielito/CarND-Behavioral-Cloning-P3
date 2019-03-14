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



def constant_conv2d_transpose(net, filters, kernel_size, strides = 1, padding = "valid", kernel_fn = tf.ones):

    inputs_shape = [ dim.value for dim in net.get_shape() ]

    in_channels = inputs_shape[-1]
    batch_size = tf.shape(net)[0]

    height, width = inputs_shape[1], inputs_shape[2]
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = strides, strides if isinstance(strides, int) else strides

    # Infer the dynamic output shape:
    out_height = deconv_output_length(
        height,
        kernel_h,
        padding,
        stride_h
    )

    out_width = deconv_output_length(
        width,
        kernel_w,
        padding,
        stride_w
    )

    output_shape = (batch_size, out_height, out_width, filters)
    strides = (1, stride_h, stride_w, 1)

    output_shape_tensor = list(output_shape)

    kernel = kernel_fn([kernel_h, kernel_w, filters, in_channels])

    outputs = tf.nn.conv2d_transpose(
        net,
        kernel,
        output_shape_tensor,
        strides,
        padding = padding.upper(),
    )

    return outputs




def deconv_output_length(input_length, filter_size, padding, stride):
    """Determines output length of a transposed convolution given input length.

    Arguments:
        input_length: integer.
        filter_size: integer.
        padding: one of "same", "valid", "full".
        stride: integer.

    Returns:
        The output length (integer).
    """
    if input_length is None:
        return None
    input_length *= stride
    if padding == 'valid':
        input_length += max(filter_size - stride, 0)
    elif padding == 'full':
        input_length -= (stride + filter_size - 2)
    return input_length

def model(images, mode, params):

    conv_dropout = params.get("conv_dropout")
    dense_dropout = params.get("dense_dropout")
    final_height = params.get("height")
    kernel_init = params.get("kernel_init", None)

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

    # Convolutional Layer #5 and Pooling Layer #5
    # conv5 = conv2d_batch_norm(
    #   inputs=pool4,
    #   filters=32,
    #   kernel_size=[3, 3],
    #   padding="valid",
    #   activation=tf.nn.relu, kernel_initializer=kernel_init,
    # **general_ops)
    # pool5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[2, 2], strides=2)
    # pool5 = tf.layers.dropout(
    #     inputs=pool5, 
    #     rate=conv_dropout, 
    #     training= mode == tf.estimator.ModeKeys.TRAIN)  
                                             
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


def deep_model(images, mode, params):

    conv_dropout = params.get("conv_dropout")
    dense_dropout = params.get("dense_dropout")
    final_height = params.get("height")
    kernel_init = params.get("kernel_init", None)

    # Input Layer
    input_layer = tf.reshape(images, [-1, final_height, 320, 3])

    general_ops = dict(
#         activation = activation,
        batch_norm = dict(training = mode == tf.estimator.ModeKeys.TRAIN)
#         kernel_regularizer = tf.contrib.layers.l2_regularizer(l2_regularization),
    )
                                             
    input_normalized = tf.layers.batch_normalization(input_layer, training = mode == tf.estimator.ModeKeys.TRAIN)
    
    # Convolutional Layer #1
    conv1 = l1 = conv2d_batch_norm(
      inputs=input_normalized,
      filters=32,
      kernel_size=[3, 3],
      padding="valid",
      activation=tf.nn.relu, kernel_initializer=kernel_init,
      **general_ops)

    conv1 = tf.layers.dropout(
        inputs=conv1, 
        rate=conv_dropout, 
        training= mode == tf.estimator.ModeKeys.TRAIN)

    # Convolutional Layer #2
    conv2 = conv2d_batch_norm(
      inputs=conv1,
      filters=64,
      kernel_size=[3, 3],
      padding="valid",
      activation=tf.nn.relu, kernel_initializer=kernel_init,
      **general_ops)
    # Pooling Layer #1
    pool1 = l2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    pool1 = tf.layers.dropout(
        inputs=pool1, 
        rate=conv_dropout, 
        training= mode == tf.estimator.ModeKeys.TRAIN) 

    # Convolutional Layer #3
    conv3 = l3 = conv2d_batch_norm(
      inputs=pool1,
      filters=64,
      kernel_size=[3, 3],
      padding="valid",
      activation=tf.nn.relu, kernel_initializer=kernel_init,
      **general_ops)

    conv3 = tf.layers.dropout(
        inputs=conv3, 
        rate=conv_dropout, 
        training= mode == tf.estimator.ModeKeys.TRAIN)

    # Convolutional Layer #4 and Pooling Layer #2
    conv4 = conv2d_batch_norm(
      inputs=conv3,
      filters=128,
      kernel_size=[3, 3],
      padding="valid",
      activation=tf.nn.relu, kernel_initializer=kernel_init,
    **general_ops)
    pool2 = l4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)
    pool2 = tf.layers.dropout(
        inputs=pool2, 
        rate=conv_dropout, 
        training= mode == tf.estimator.ModeKeys.TRAIN)
                                             
    # Convolutional Layer #5 and Pooling Layer #3
    conv5 = conv2d_batch_norm(
      inputs=pool2,
      filters=128,
      kernel_size=[3, 3],
      padding="valid",
      activation=tf.nn.relu, kernel_initializer=kernel_init,
    **general_ops)
    pool3 = l5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[2, 2], strides=2)
    pool3 = tf.layers.dropout(
        inputs=pool3, 
        rate=conv_dropout, 
        training= mode == tf.estimator.ModeKeys.TRAIN)  
                                             
    # Convolutional Layer #6 and Pooling Layer #4
    conv6 = conv2d_batch_norm(
      inputs=pool3,
      filters=128,
      kernel_size=[3, 3],
      padding="valid",
      activation=tf.nn.relu, kernel_initializer=kernel_init,
    **general_ops)
    pool4 = l6 = tf.layers.max_pooling2d(inputs=conv6, pool_size=[2, 2], strides=2)
    pool4 = tf.layers.dropout(
        inputs=pool4, 
        rate=conv_dropout, 
        training= mode == tf.estimator.ModeKeys.TRAIN)  

                                             
    # Dense Layers
    pool_flat = tf.layers.flatten(conv6)
    
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


    with tf.variable_scope("mask") as scope:
        print("#######################")
        print("## Mask")
        print("#######################")

        # conv2d_transpose_ops = dict(use_bias = False, kernel_initializer = tf.ones_initializer(), trainable = False)
        conv2d_transpose_ops = dict(kernel_fn = tf.ones)

        layer = tf.nn.relu(l6); print(layer)
        mask = tf.reduce_mean(layer, axis = 3, keepdims = True); print(mask)
        mask = constant_conv2d_transpose(mask, 1, [3, 3], strides=2, **conv2d_transpose_ops); print(mask)

        layer = tf.nn.relu(l5); print(layer)
        mask = mask * tf.reduce_mean(layer, axis = 3, keepdims = True); print(mask)
        mask = constant_conv2d_transpose(mask, 1, [3, 3], strides=2, **conv2d_transpose_ops); print(mask)

        layer = tf.nn.relu(l4); print(layer)
        mask = mask * tf.reduce_mean(layer, axis = 3, keepdims = True); print(mask)
        mask = constant_conv2d_transpose(mask, 1, [3, 3], strides = 2, **conv2d_transpose_ops); print(mask)

        layer = tf.nn.relu(l3); print(layer)
        mask = mask * tf.reduce_mean(layer, axis = 3, keepdims = True); print(mask)
        mask = constant_conv2d_transpose(mask, 1, [3, 3], **conv2d_transpose_ops); print(mask)

        layer = tf.nn.relu(l2); print(layer)
        mask = mask * tf.reduce_mean(layer, axis = 3, keepdims = True); print(mask)
        mask = constant_conv2d_transpose(mask, 1, [3, 3], strides = 2, **conv2d_transpose_ops); print(mask)

        layer = tf.nn.relu(l1); print(layer)
        mask = mask * tf.reduce_mean(layer, axis = 3, keepdims = True); print(mask)
        mask = constant_conv2d_transpose(mask, 1, [3, 3], **conv2d_transpose_ops); print(mask)

        mask = mask / tf.reduce_max(mask, axis = [1, 2, 3], keepdims = True); print(mask)

        print("#######################")


        predictions["mask"] = mask

    return predictions


def nvidia_net(images, mode, params):

    conv_dropout = params.get("conv_dropout")
    dense_dropout = params.get("dense_dropout")
    final_height = params.get("height")
    kernel_init = params.get("kernel_init", None)

    # Input Layer
    input_layer = tf.reshape(images, [-1, final_height, 320, 3])

    general_ops = dict(
#         activation = activation,
        batch_norm = dict(training = mode == tf.estimator.ModeKeys.TRAIN)
#         kernel_regularizer = tf.contrib.layers.l2_regularizer(l2_regularization),
    )
                                             
    input_normalized = tf.layers.batch_normalization(input_layer, training = mode == tf.estimator.ModeKeys.TRAIN)
    
    # Convolutional Layer #1
    conv1 = l1 = conv2d_batch_norm(
      inputs=input_normalized,
      filters=36,
      kernel_size=[5, 5],
      strides = 2,
      padding="valid",
      activation=tf.nn.relu, kernel_initializer=kernel_init,
      **general_ops)

    conv1 = tf.layers.dropout(
        inputs=conv1, 
        rate=conv_dropout, 
        training= mode == tf.estimator.ModeKeys.TRAIN)

    # Convolutional Layer #2
    conv2 = l2 = conv2d_batch_norm(
      inputs=conv1,
      filters=36,
      kernel_size=[5, 5],
      strides = 2,
      padding="valid",
      activation=tf.nn.relu, kernel_initializer=kernel_init,
      **general_ops)
    pool1 = tf.layers.dropout(
        inputs=conv2, 
        rate=conv_dropout, 
        training= mode == tf.estimator.ModeKeys.TRAIN) 

    # Convolutional Layer #3
    conv3 = l3 = conv2d_batch_norm(
      inputs=pool1,
      filters=48,
      kernel_size=[5, 5],
      strides = 2,
      padding="valid",
      activation=tf.nn.relu, kernel_initializer=kernel_init,
      **general_ops)

    conv3 = tf.layers.dropout(
        inputs=conv3, 
        rate=conv_dropout, 
        training= mode == tf.estimator.ModeKeys.TRAIN)

    # Convolutional Layer #4 and Pooling Layer #2
    conv4 = l4 = conv2d_batch_norm(
      inputs=conv3,
      filters=64,
      kernel_size=[3, 3],
      padding="valid",
      activation=tf.nn.relu, kernel_initializer=kernel_init,
    **general_ops)
    pool2 = tf.layers.dropout(
        inputs=conv4, 
        rate=conv_dropout, 
        training= mode == tf.estimator.ModeKeys.TRAIN)
                                             
    # Convolutional Layer #5 and Pooling Layer #3
    conv5 = l5 = conv2d_batch_norm(
      inputs=pool2,
      filters=64,
      kernel_size=[3, 3],
      padding="valid",
      activation=tf.nn.relu, kernel_initializer=kernel_init,
    **general_ops)
    pool3 = tf.layers.dropout(
        inputs=conv5, 
        rate=conv_dropout, 
        training= mode == tf.estimator.ModeKeys.TRAIN)  

    # Dense Layers
    pool_flat = tf.layers.flatten(pool3)
    
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


    with tf.variable_scope("mask") as scope:
        print("#######################")
        print("## Mask")
        print("#######################")

        # conv2d_transpose_ops = dict(use_bias = False, kernel_initializer = tf.ones_initializer(), trainable = False)
        conv2d_transpose_ops = dict(kernel_fn = tf.ones)

        layer = tf.nn.relu(l5); print(layer)
        mask = tf.reduce_mean(layer, axis = 3, keepdims = True); print(mask)
        mask = constant_conv2d_transpose(mask, 1, [3, 3], **conv2d_transpose_ops); print(mask)

        layer = tf.nn.relu(l4); print(layer)
        mask = mask * tf.reduce_mean(layer, axis = 3, keepdims = True); print(mask)
        mask = constant_conv2d_transpose(mask, 1, [3, 3], **conv2d_transpose_ops); print(mask)

        layer = tf.nn.relu(l3); print(layer)
        mask = mask * tf.reduce_mean(layer, axis = 3, keepdims = True); print(mask)
        mask = constant_conv2d_transpose(mask, 1, [5, 5], strides = 2, **conv2d_transpose_ops); print(mask)

        layer = tf.nn.relu(l2); print(layer)
        mask = mask * tf.reduce_mean(layer, axis = 3, keepdims = True); print(mask)
        mask = constant_conv2d_transpose(mask, 1, [6, 6], strides = 2, **conv2d_transpose_ops); print(mask)

        layer = tf.nn.relu(l1); print(layer)
        mask = mask * tf.reduce_mean(layer, axis = 3, keepdims = True); print(mask)
        mask = constant_conv2d_transpose(mask, 1, [6, 6], strides = 2, **conv2d_transpose_ops); print(mask)

        mask = mask / tf.reduce_max(mask, axis = [1, 2, 3], keepdims = True); print(mask)

        print("#######################")


        predictions["mask"] = mask

    return predictions