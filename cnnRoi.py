from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from google.colab import files

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel
  input_layer = tf.reshape(features["x"], [-1, 16, 16, 1])

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 28, 28, 1]
  # Output Tensor Shape: [batch_size, 28, 28, 32]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 28, 28, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 32]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 14, 14, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 64]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 14, 14, 64]
  # Output Tensor Shape: [batch_size, 7, 7, 64]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 7, 7, 64]
  # Output Tensor Shape: [batch_size, 7 * 7 * 64]
  pool2_flat = tf.reshape(pool2, [-1, 4 * 4 * 64])

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 7 * 7 * 64]
  # Output Tensor Shape: [batch_size, 1024]
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 10]
  logits = tf.layers.dense(inputs=dropout, units=2)
  
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


def main(unused_argv):
  # Load training and eval data
  #mnist = tf.contrib.learn.datasets.load_dataset("mnist")
  #train_data = mnist.train.images  # Returns np.array
  #train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
  #eval_data = mnist.test.images  # Returns np.array
  #eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

  try:
    x = np.loadtxt('dataset.txt', dtype=int, delimiter=' ')
  except IOError:
    uploaded = files.upload()
    x = np.loadtxt('dataset.txt', dtype=int, delimiter=' ')
    
  try:
    y = np.loadtxt('roilist.txt', dtype=int)
  except IOError:
    uploaded = files.upload()
    y = np.loadtxt('roilist.txt', dtype=int)
  

  # PB: aleatorizacao do conjunto completo
  y = np.reshape(y, (-1,1))
  c = np.concatenate((x,y), axis=1)
  np.random.shuffle(c)
  c = np.hsplit(c, np.array([256,257]))
  x = c[0]
  y = c[1]

  x = x.astype(float) / 255.0
  
  
  x = np.split(x, [12000,14809])
  y = np.split(y, [12000,14809])
  
  train_data = x[0]
  train_labels = np.asarray(y[0])
    
  eval_data = x[1]
  eval_labels = np.asarray(y[1])

  # Create the Estimator
  roi_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir="models/roi_convnet_model_new")

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  #tensors_to_log = {"probabilities": "softmax_tensor"}
  #logging_hook = tf.train.LoggingTensorHook(
  #    tensors=tensors_to_log, every_n_iter=1000)

  # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      batch_size=100,
      num_epochs=None,
      shuffle=True)
  roi_classifier.train(
      input_fn=train_input_fn,
      steps=10000)
      #hooks=[logging_hook])

  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data},
      y=eval_labels,
      num_epochs=1,
      shuffle=False)
  eval_results = roi_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)

  
  # PB: classificar novos blocos
  #filename = '105_8.tif.bmp_text.txt'
  #try:
  #  new_samples = np.loadtxt(filename, dtype=int, delimiter=' ')
  #except IOError:
  #  uploaded = files.upload()
  #  new_samples = np.loadtxt(filename, dtype=int, delimiter=' ')
    
  #new_samples = new_samples.astype(float) / 255.0
  
  #predict_input_fn = tf.estimator.inputs.numpy_input_fn(
  #    x={"x": new_samples},
  #    num_epochs=1,
  #    shuffle=False)

  #predictions = list(roi_classifier.predict(input_fn=predict_input_fn))
  #predicted_classes = [p["classes"] for p in predictions]

  #print(
  #    "New Samples, Class Predictions:    {}\n"
  #    .format(predicted_classes))

if __name__ == "__main__":
  tf.app.run()
