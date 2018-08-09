from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from cnn_model_fn import cnn_model_fn

import numpy as np
import tensorflow as tf
import sys
import time
import glob
  
tf.logging.set_verbosity(tf.logging.INFO)


def main(unused_argv):

  if len(sys.argv) > 1:
    path = sys.argv[1]
  else:
    print ('Erro: nenhum arquivo passado')
    sys.exit()
      
  # Create the Estimator
  roi_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir="models/roi_convnet_model_new")
    
  files = glob.glob(path + '*.txt')
    
  for filename in files:
    print("filename: " + filename)
    new_samples = np.loadtxt(filename, dtype=int, delimiter=' ')
    new_samples = new_samples.astype(float) / 255.0
      
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": new_samples},
      num_epochs=1,
      shuffle=False)
      
    start_time = time.time()
    predictions = list(roi_classifier.predict(input_fn=predict_input_fn))
    print("--- %s seconds ---" % (time.time() - start_time))
  
    predicted_classes = [p["classes"] for p in predictions]
    
    np.savetxt(filename + "_out.txt", np.transpose(predicted_classes), fmt='%d', newline=' ')
  

if __name__ == "__main__":
  tf.app.run()
