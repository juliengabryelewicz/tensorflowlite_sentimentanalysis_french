import numpy as np
import os

from tflite_model_maker import model_spec
from tflite_model_maker import text_classifier
from tflite_model_maker.config import ExportFormat, QuantizationConfig
from tflite_model_maker.text_classifier import AverageWordVecSpec
from tflite_model_maker.text_classifier import DataLoader
import tensorflow as tf

assert tf.__version__.startswith('2')
tf.get_logger().setLevel('ERROR')

mb_spec = model_spec.get('average_word_vec')

import pandas as pd

def replace_label(original_file, new_file):
  df = pd.read_csv(original_file, sep='\t')
  label_map = {0: 'negative', 1: 'positive'}
  df.replace({'label': label_map}, inplace=True)
  df.to_csv(new_file)

train_data = DataLoader.from_csv(
      filename='train.csv',
      text_column='sentence',
      label_column='label',
      model_spec=mb_spec,
      is_training=True)
test_data = DataLoader.from_csv(
      filename='test.csv',
      text_column='sentence',
      label_column='label',
      model_spec=mb_spec,
      is_training=False)


model = text_classifier.create(train_data, model_spec=mb_spec, epochs=50)

loss, acc = model.evaluate(test_data)

model.summary()

config = QuantizationConfig.for_float16()

model.export(export_dir='average_word_vec', export_format=[ExportFormat.VOCAB], quantization_config=config)

accuracy = model.evaluate_tflite('average_word_vec/model.tflite', test_data)

print('TFLite model accuracy: ', accuracy)

