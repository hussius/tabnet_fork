# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from fire import Fire
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python.util import deprecation

deprecation._PRINT_DEPRECATION_WARNINGS = False
import tabnet_model


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["KMP_AFFINITY"] = "none"
np.random.seed(1)
tf.compat.v1.set_random_seed(1)


def pandas_input_fn(df,
                    label_column,
                    dataset_info,
                    num_epochs,
                    shuffle,
                    batch_size,
                    n_buffer=50):

    dataframe = df.copy()
    labels = dataframe.pop(label_column)
    if dataset_info['task'] == 'classification':
        labels = [dataset_info['class_map'][val] for val in labels]
    # Change dtype of int categoricals to str to avoid errors with integer categoricals
    for col in dataset_info['cat_columns']:
        if dataframe[col].dtype == int:
            dataframe[col] = [str(i) for i in dataframe[col]]
    if dataset_info['task'] == 'classification':
        labels = tf.cast(labels, tf.int32)
    else:
        labels = tf.cast(labels, tf.float32)
    dataset = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=n_buffer)

    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    return dataset


def csv_input_fn(data_file,
                 label_column,
                 all_columns,
                 num_epochs,
                 shuffle,
                 batch_size,
                 n_buffer=50,
                 n_parallel=16):
  """Function to read the input file and return the dataset."""

  def parse_csv(value_column):
    columns = tf.decode_csv(value_column, record_defaults=defaults)
    features = dict(zip(all_columns, columns))
    label = features.pop(label_column)
    classes = tf.cast(label, tf.int32) - 1
    return features, classes

  # Extract lines from input files using the Dataset API.
  dataset = tf.data.TextLineDataset(data_file)

  if shuffle:
    dataset = dataset.shuffle(buffer_size=n_buffer)

  dataset = dataset.map(parse_csv, num_parallel_calls=n_parallel)

  # Repeat after shuffling, to prevent separate epochs from blending together.
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.batch(batch_size)
  return dataset


def prepare_dataset(df, categorical_features, target_name, task, embedding_dim=1):

    all_columns = df.columns
    label_column = target_name
    if type(categorical_features) == str:
        cat_columns = categorical_features.split(',')
    elif type(categorical_features) == str:
        cat_columns = list(categorical_features)
    else:
        cat_columns = categorical_features
    num_columns = set(all_columns) - set(cat_columns) - {label_column}
    n_unique = {col: df[col].nunique() for col in cat_columns}
    emb_dim = {col: embedding_dim for col in cat_columns}

    feature_columns = []
    for col in list(df.columns.drop(label_column)):
        if col in num_columns:
            feature_columns.append(tf.feature_column.numeric_column(col))
        else:
            feature_columns.append(tf.feature_column.embedding_column(
                tf.feature_column.categorical_column_with_hash_bucket(
                    col, hash_bucket_size=int(3 * n_unique[col])),
                dimension=emb_dim[col]))

    dataset_info = {}
    if task == 'classification':
        class_map = {name: idx for (idx, name) in enumerate(df[label_column].unique())}
        dataset_info['class_map'] = class_map
    dataset_info['num_classes'] = df[label_column].nunique()
    dataset_info['num_features'] = len(num_columns) + sum([v for (k, v) in emb_dim.items()])
    dataset_info['feature_columns'] = feature_columns
    dataset_info['task'] = task
    dataset_info['cat_columns'] = cat_columns

    return dataset_info


def main(csv_path, target_name, task='classification', categorical_features=[],
         feature_dim=128, output_dim=64, batch_size=512, virtual_batch_size=512, batch_momentum=0.7, gamma=1.5,
         n_steps=6, max_steps=25, lr=0.02, decay_every=500, lambda_sparsity=0.0001):

  all_data = pd.read_csv(csv_path)
  trainval_df, test_df = train_test_split(all_data, test_size=0.2)
  train_df, val_df = train_test_split(trainval_df, test_size=0.1)

  dataset_info = prepare_dataset(all_data, categorical_features, target_name, task)

  # TabNet model
  tabnet = tabnet_model.TabNet(
      columns=dataset_info['feature_columns'],
      num_features=dataset_info['num_features'],
      feature_dim=feature_dim,
      output_dim=output_dim,
      num_decision_steps=n_steps,
      relaxation_factor=gamma,
      batch_momentum=batch_momentum,
      virtual_batch_size=virtual_batch_size,
      num_classes=dataset_info['num_classes'])

  label_column = target_name

  # Training parameters
  max_steps = max_steps
  display_step = 5
  val_step = 5
  save_step = 5
  init_localearning_rate = lr
  decay_every = decay_every
  decay_rate = 0.95
  batch_size = batch_size
  sparsity_loss_weight = lambda_sparsity
  gradient_thresh = 2000.0

  # Input sampling
  train_batch = pandas_input_fn(
      train_df,
      label_column,
      dataset_info,
      num_epochs=100000,
      shuffle=True,
      batch_size=batch_size,
      n_buffer=1)
  val_batch = pandas_input_fn(
      val_df,
      label_column,
      dataset_info,
      num_epochs=10000,
      shuffle=False,
      batch_size=batch_size,
      n_buffer=1)
  test_batch = pandas_input_fn(
      test_df,
      label_column,
      dataset_info,
      num_epochs=10000,
      shuffle=False,
      batch_size=batch_size,
      n_buffer=1)

  train_iter = train_batch.make_initializable_iterator()
  val_iter = val_batch.make_initializable_iterator()
  test_iter = test_batch.make_initializable_iterator()

  feature_train_batch, label_train_batch = train_iter.get_next()
  feature_val_batch, label_val_batch = val_iter.get_next()
  feature_test_batch, label_test_batch = test_iter.get_next()

  # Define the model and losses

  encoded_train_batch, total_entropy = tabnet.encoder(
      feature_train_batch, reuse=False, is_training=True)

  if task == 'classification':

      logits_orig_batch, _ = tabnet.classify(
          encoded_train_batch, reuse=False)

      softmax_orig_key_op = tf.reduce_mean(
          tf.nn.sparse_softmax_cross_entropy_with_logits(
              logits=logits_orig_batch, labels=label_train_batch))

      train_loss_op = softmax_orig_key_op + sparsity_loss_weight * total_entropy

  else:

      predictions = tabnet.regress(
          encoded_train_batch, reuse=False
      )

      #l2_loss = tf.reduce_mean(
      #    tf.nn.l2_loss(t = tf.subtract(predictions, label_train_batch)) / tf.to_float(tf.size(predictions))
      #)

      l2_loss = tf.reduce_mean(tf.square(tf.subtract(predictions, label_train_batch)))

      train_loss_op = l2_loss + sparsity_loss_weight * total_entropy

  tf.compat.v1.summary.scalar("Total loss", train_loss_op)

  # Optimization step
  global_step = tf.compat.v1.train.get_or_create_global_step()
  learning_rate = tf.compat.v1.train.exponential_decay(
      init_localearning_rate,
      global_step=global_step,
      decay_steps=decay_every,
      decay_rate=decay_rate)
  optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
  update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    gvs = optimizer.compute_gradients(train_loss_op)
    capped_gvs = [(tf.clip_by_value(grad, -gradient_thresh,
                                    gradient_thresh), var) for grad, var in gvs]
    train_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)

  # Model evaluation

  # Validation performance
  encoded_val_batch, _ = tabnet.encoder(
      feature_val_batch, reuse=True, is_training=True)

  val_op = None

  if task == 'classification':
    _, prediction_val = tabnet.classify(
        encoded_val_batch, reuse=True)

    predicted_labels = tf.cast(tf.argmax(prediction_val, 1), dtype=tf.int32)
    val_eq_op = tf.equal(predicted_labels, label_val_batch)
    val_acc_op = tf.reduce_mean(tf.cast(val_eq_op, dtype=tf.float32))
    tf.compat.v1.summary.scalar("Val accuracy", val_acc_op)
    val_op = val_acc_op

  else:
    predictions = tabnet.regress(
          encoded_val_batch, reuse=True
      )

    val_loss_op = tf.reduce_mean(tf.square(tf.subtract(predictions, label_train_batch)))

    val_op = val_loss_op
    tf.compat.v1.summary.scalar("Validation loss", val_loss_op)

  # Test performance
  encoded_test_batch, _ = tabnet.encoder(
      feature_test_batch, reuse=True, is_training=True)
  test_op = None

  if task == 'classification':
    _, prediction_test = tabnet.classify(
        encoded_test_batch, reuse=True)

    predicted_labels = tf.cast(tf.argmax(prediction_test, 1), dtype=tf.int32)
    test_eq_op = tf.equal(predicted_labels, label_test_batch)
    test_acc_op = tf.reduce_mean(tf.cast(test_eq_op, dtype=tf.float32))
    tf.compat.v1.summary.scalar("Test accuracy", test_acc_op)
    test_op = test_acc_op

  else:
      predictions = tabnet.regress(
          encoded_test_batch, reuse=True
      )

      test_loss_op = tf.reduce_mean(tf.square(tf.subtract(predictions, label_test_batch)))

      tf.compat.v1.summary.scalar("Test loss", test_loss_op)
      test_op = test_loss_op

  # Training setup
  model_name = "tabnet"
  init = tf.initialize_all_variables()
  init_local = tf.compat.v1.local_variables_initializer()
  init_table = tf.compat.v1.tables_initializer(name="Initialize_all_tables")
  saver = tf.compat.v1.train.Saver()
  summaries = tf.compat.v1.summary.merge_all()

  with tf.compat.v1.Session() as sess:
    summary_writer = tf.compat.v1.summary.FileWriter("./tflog/" + model_name, sess.graph)

    sess.run(init)
    sess.run(init_local)
    sess.run(init_table)
    sess.run(train_iter.initializer)
    sess.run(val_iter.initializer)
    sess.run(test_iter.initializer)

    for step in range(1, max_steps + 1):
      if step % display_step == 0:
        _, train_loss, merged_summary = sess.run(
            [train_op, train_loss_op, summaries])
        summary_writer.add_summary(merged_summary, step)
        print("Step " + str(step) + " , Training Loss = " +
              "{:.4f}".format(train_loss))
      else:
        _ = sess.run(train_op)

      if step % val_step == 0:
        feed_arr = [
            vars()["summaries"],
            vars()[f"val_op"],
            vars()[f"test_op"]
        ]

        val_arr = sess.run(feed_arr)
        merged_summary = val_arr[0]
        val_acc = val_arr[1]

        print("Step " + str(step) + " , Val Metric = " +
              "{:.4f}".format(val_acc))
        summary_writer.add_summary(merged_summary, step)

      if step % save_step == 0:
        saver.save(sess, "./checkpoints/" + model_name + ".ckpt")


if __name__ == "__main__":
  Fire(main)
