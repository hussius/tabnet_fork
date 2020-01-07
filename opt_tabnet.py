from functools import partial
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from fire import Fire
from hyperopt import fmin, hp, STATUS_OK, tpe
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python.util import deprecation

deprecation._PRINT_DEPRECATION_WARNINGS = False
import tabnet_model


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KMP_AFFINITY'] = "none"
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
    labels = [dataset_info['class_map'][val] for val in labels]
    # Change dtype of int categoricals to str to avoid errors with integer categoricals
    for col in dataset_info['cat_columns']:
        if dataframe[col].dtype == int:
            dataframe[col] = [str(i) for i in dataframe[col]]
    labels = tf.cast(labels, tf.int32)
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


def prepare_dataset(df, categorical_features, target_name, embedding_dim=1):

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
    class_map = {name: idx for (idx, name) in enumerate(df[label_column].unique())}
    dataset_info['class_map'] = class_map
    dataset_info['num_classes'] = df[label_column].nunique()
    dataset_info['num_features'] = len(num_columns) + sum([v for (k, v) in emb_dim.items()])
    dataset_info['feature_columns'] = feature_columns
    dataset_info['cat_columns'] = cat_columns

    return dataset_info


def train_and_evaluate(params,
                       batch_size,
                       virtual_batch_size,
                       max_steps,
                       lr,
                       decay_every,
                       target_name,
                       dataset_info,
                       train_df,
                       val_df):

  tf.compat.v1.reset_default_graph()
  print(params)

  # TabNet model
  tabnet = tabnet_model.TabNet(
      columns=dataset_info['feature_columns'],
      num_features=dataset_info['num_features'],
      feature_dim=int(params['n_a']),
      output_dim=int(params['n_a']), # Same dims for feature and output
      num_decision_steps=int(params['n_steps']),
      relaxation_factor=params['gamma'],
      batch_momentum=params['batch_momentum'],
      virtual_batch_size=virtual_batch_size,
      num_classes=dataset_info['num_classes'])

  label_column = target_name

  # Training parameters
  max_steps = max_steps
  display_step = 5
  val_step = 5
  init_localearning_rate = lr
  decay_every = decay_every
  decay_rate = 0.95
  batch_size = batch_size
  sparsity_loss_weight = params['lambda']
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

  train_iter = train_batch.make_initializable_iterator()
  val_iter = val_batch.make_initializable_iterator()

  feature_train_batch, label_train_batch = train_iter.get_next()
  feature_val_batch, label_val_batch = val_iter.get_next()

  # Define the model and losses

  encoded_train_batch, total_entropy = tabnet.encoder(
      feature_train_batch, is_training=True)

  logits_orig_batch, _ = tabnet.classify(
      encoded_train_batch)

  softmax_orig_key_op = tf.reduce_mean(
      tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=logits_orig_batch, labels=label_train_batch))

  train_loss_op = softmax_orig_key_op + sparsity_loss_weight * total_entropy

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
      feature_val_batch, is_training=True)

  val_op = None
  _, prediction_val = tabnet.classify(
      encoded_val_batch)
  predicted_labels = tf.cast(tf.argmax(prediction_val, 1), dtype=tf.int32)
  val_eq_op = tf.equal(predicted_labels, label_val_batch)
  val_acc_op = tf.reduce_mean(tf.cast(val_eq_op, dtype=tf.float32))
  val_op = val_acc_op


  # Training setup
  init = tf.initialize_all_variables()
  init_local = tf.compat.v1.local_variables_initializer()
  init_table = tf.compat.v1.tables_initializer(name="Initialize_all_tables")
  summaries = tf.compat.v1.summary.merge_all()

  with tf.compat.v1.Session() as sess:
    sess.run(init)
    sess.run(init_local)
    sess.run(init_table)
    sess.run(train_iter.initializer)
    sess.run(val_iter.initializer)

    early_stop_steps = 25
    best_val_acc = -1

    for step in range(1, max_steps + 1):
      if step % display_step == 0:
        _, train_loss, merged_summary = sess.run(
            [train_op, train_loss_op, summaries])
      else:
        _ = sess.run(train_op)

      if step % val_step == 0:
        feed_arr = [
            vars()["summaries"],
            vars()[f"val_op"],
        ]

        val_arr = sess.run(feed_arr)
        merged_summary = val_arr[0]
        val_acc = val_arr[1]
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_step = step
        if (step - best_val_step) > early_stop_steps:
            break

    print(f'Best validation accuracy: {best_val_acc}')
    return -1*best_val_acc


def main(csv_path, target_name,
         categorical_features=[], val_frac=0.25, test_frac=0.25,
         emb_size=1):

    all_data = pd.read_csv(csv_path)
    trainval_df, test_df = train_test_split(all_data, test_size=test_frac, stratify=all_data[target_name])
    val_frac_after_test_split = val_frac / (1 - test_frac)
    train_df, val_df = train_test_split(trainval_df, test_size=val_frac_after_test_split)
    dataset_info = prepare_dataset(all_data, categorical_features, target_name, embedding_dim=emb_size)

    params = {'lambda': hp.choice('lambda', [0.01, 0.001, 0.0001, 0.00001]),
              'n_steps': hp.quniform('n_steps', 3, 10, 1),
              'n_a': hp.quniform('n_a', 8, 128, 8),
              'gamma': hp.uniform('gamma', 0.1, 3),
              'batch_momentum': hp.uniform('batch_momentum', 0.0, 1.0)
              }

    opt_fn = partial(train_and_evaluate,
                     batch_size=4096,
                     virtual_batch_size=128,
                     max_steps=2000,
                     lr=0.2,
                     decay_every=500,
                     target_name=target_name,
                     dataset_info=dataset_info,
                     train_df=train_df,
                     val_df=val_df
                     )

    # Now opt_fn is a function of a single variable, params
    best = fmin(
        opt_fn,
        params,
        algo=tpe.suggest,
        max_evals=100,
    )

    print(f'Best parameter setting: {best}')

if __name__ == "__main__":
  Fire(main)
