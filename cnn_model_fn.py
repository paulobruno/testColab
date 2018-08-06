import tensorflow as tf

def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel
  input_layer = tf.reshape(features["x"], [-1, 16, 16, 1])

  # Convolutional Layer #1
  # Computes 64 features using a 11x11 filter with ReLU activation.
  # Input Tensor Shape: [batch_size, 16, 16, 1]
  # Output Tensor Shape: [batch_size, 28, 28, 64]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=64,
      kernel_size=[5, 5],
      strides=1,
      padding="same",
      activation=tf.nn.relu,
      kernel_initializer=tf.contrib.layers.xavier_initializer(),
      bias_initializer=tf.zeros_initializer())

#  print('conv1: ' + str(conv1.shape))

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 28, 28, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 32]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
  
#  print('pool1: ' + str(pool1.shape))

  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 14, 14, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 64]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=192,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu,
      kernel_initializer=tf.contrib.layers.xavier_initializer(),
      bias_initializer=tf.zeros_initializer())

#  print('conv2: ' + str(conv2.shape))
  
  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 14, 14, 64]
  # Output Tensor Shape: [batch_size, 7, 7, 64]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

#  print('pool2: ' + str(pool2.shape))
  
  conv3 = tf.layers.conv2d(
      inputs=pool2,
      filters=384,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      kernel_initializer=tf.contrib.layers.xavier_initializer(),
      bias_initializer=tf.zeros_initializer())
      
 # print('conv3: ' + str(conv3.shape))
  
  conv4 = tf.layers.conv2d(
      inputs=conv3,
      filters=384,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      kernel_initializer=tf.contrib.layers.xavier_initializer(),
      bias_initializer=tf.zeros_initializer())
      
 # print('conv4: ' + str(conv4.shape))
  
  conv5 = tf.layers.conv2d(
      inputs=conv4,
      filters=256,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      kernel_initializer=tf.contrib.layers.xavier_initializer(),
      bias_initializer=tf.zeros_initializer())
      
#  print('conv5: ' + str(conv5.shape))
  
  pool5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[2, 2], strides=2)
  
#  print('pool5: ' + str(pool5.shape))
  
  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 7, 7, 64]
  # Output Tensor Shape: [batch_size, 7 * 7 * 64]
  pool5_flat = tf.reshape(pool5, [-1, 2 * 2 * 256])
  
#  print('pool5_flat: ' + str(pool5_flat.shape))

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 7 * 7 * 64]
  # Output Tensor Shape: [batch_size, 1024]
  fc6 = tf.layers.dense(inputs=pool5_flat, units=4096, activation=tf.nn.relu)
  
#  print('fc6: ' + str(fc6.shape))
  
  fc7 = tf.layers.dense(inputs=fc6, units=4096, activation=tf.nn.relu)
  
#  print('fc7: ' + str(fc7.shape))
  
  fc8 = tf.layers.dense(inputs=fc7, units=1000, activation=tf.nn.relu)
  
#  print('fc8: ' + str(fc8.shape))

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=fc8, rate=0.7, training=mode == tf.estimator.ModeKeys.TRAIN)
      
#  print('dropout: ' + str(dropout.shape))

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 10]
  logits = tf.layers.dense(inputs=dropout, units=2)
  
#  print("logits: " + str(logits.shape))
  
  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
  
  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
  
  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
  
  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

