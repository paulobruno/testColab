from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from cnn_model_fn import cnn_model_fn

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


def main(unused_argv):

  x = np.loadtxt('dataset.txt', dtype=int, delimiter=' ')
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
      steps=50000)
      #hooks=[logging_hook])

  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data},
      y=eval_labels,
      num_epochs=1,
      shuffle=False)
  eval_results = roi_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)


if __name__ == "__main__":
  tf.app.run()
