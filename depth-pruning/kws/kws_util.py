import os
import argparse
import matplotlib.pyplot as plt
import math
import tensorflow as tf
from tensorflow import keras
import tensorflow_model_optimization as tfmot

import pathlib

def parse_command():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        type=str,
        default=os.path.join(os.getenv('HOME'), 'data'),
        help="""\
        Where to download the speech training data to. Or where it is already saved.
        """)
    parser.add_argument(
        '--bg_path',
        type=str,
        default=os.path.join(os.getenv('PWD')),
        help="""\
        Where to find background noise folder.
        """)
    parser.add_argument(
        '--background_volume',
        type=float,
        default=0.1,
        help="""\
        How loud the background noise should be, between 0 and 1.
        """)
    parser.add_argument(
        '--background_frequency',
        type=float,
        default=0.8,
        help="""\
        How many of the training samples have background noise mixed in.
        """)
    parser.add_argument(
        '--silence_percentage',
        type=float,
        default=10.0,
        help="""\
        How much of the training data should be silence.
        """)
    parser.add_argument(
        '--unknown_percentage',
        type=float,
        default=10.0,
        help="""\
        How much of the training data should be unknown words.
        """)
    parser.add_argument(
        '--time_shift_ms',
        type=float,
        default=100.0,
        help="""\
        Range to randomly shift the training audio by in time.
        """)
    parser.add_argument(
        '--sample_rate',
        type=int,
        default=16000,
        help='Expected sample rate of the wavs',)
    parser.add_argument(
        '--clip_duration_ms',
        type=int,
        default=1000,
        help='Expected duration in milliseconds of the wavs',)
    parser.add_argument(
        '--window_size_ms',
        type=float,
        default=30.0,
        help='How long each spectrogram timeslice is',)
    parser.add_argument(
        '--window_stride_ms',
        type=float,
        default=20.0,
        help='How long each spectrogram timeslice is',)
    parser.add_argument(
        '--feature_type',
        type=str,
        default="mfcc",
        choices=["mfcc", "lfbe", "td_samples"],
        help='Type of input features. Valid values: "mfcc" (default), "lfbe", "td_samples"',)
    parser.add_argument(
        '--dct_coefficient_count',
        type=int,
        default=10,
        help='How many MFCC or log filterbank energy features')
    parser.add_argument(
        '--epochs',
        type=int,
        default=36,
        help='How many epochs to train',)
    parser.add_argument(
        '--num_train_samples',
        type=int,
        default=-1, # 85511,
    help='How many samples from the training set to use',)
    parser.add_argument(
        '--num_val_samples',
        type=int,
        default=-1, # 10102,
    help='How many samples from the validation set to use',)
    parser.add_argument(
        '--num_test_samples',
        type=int,
        default=-1, # 4890,
    help='How many samples from the test set to use',)
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help='How many items to train with at once',)
    parser.add_argument(
        '--num_bin_files',
        type=int,
        default=1000,
        help='How many binary test files for benchmark runner to create',)
    parser.add_argument(
        '--bin_file_path',
        type=str,
        default=os.path.join(os.getenv('HOME'), 'kws_test_files'),
        help="""\
        Directory where plots of binary test files for benchmark runner are written.
        """)
    parser.add_argument(
        '--model_architecture',
        type=str,
        default='ds_cnn',
        help='What model architecture to use')
    parser.add_argument(
        '--run_test_set',
        type=bool,
        default=True,
        help='In train.py, run model.eval() on test set if True')
    parser.add_argument(
        '--saved_model_path',
        type=str,
        default='trained_models/kws_model.h5',
        help='In quantize.py, path to load pretrained model from; in train.py, destination for trained model')
    parser.add_argument(
        '--model_init_path',
        type=str,
        default=None,
        help='Path to load pretrained model for evaluation or starting point for training')
    parser.add_argument(
        '--tfl_file_name',
        default='trained_models/kws_model.tflite',
        help='File name to which the TF Lite model will be saved (quantize.py) or loaded (eval_quantized_model)')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.00001,
        help='Initial LR',)
    parser.add_argument(
        '--lr_sched_name',
        type=str,
        default='step_function',
        help='lr schedule scheme name to be picked from lr.py')  
    parser.add_argument(
        '--plot_dir',
        type=str,
        default='./plots',
        help="""\
        Directory where plots of accuracy vs Epochs are stored
        """)
    parser.add_argument(
        '--target_set',
        type=str,
        default='test',
        help="""\
        For eval_quantized_model, which set to measure.
        """)

  # Pruning Flags
    parser.add_argument(
        '--attach_tensor_name',
        help='Name of a tensor in the base model where the auxiliary network will be attached',
        type=str)
    parser.add_argument(
        '--aux_arch',
        help='Artchitecture for the auxiliary network (dscnn32_16, dscnn16_16_16, dscnn32, conv, dense)',
        type=str)
    parser.add_argument(
        '--train_dir',
        type=str,
        default='aux_models/test_models',
        help='Directory to write event logs and checkpoint.')
    parser.add_argument(
        '--ckpt_name',
        type=str)
    parser.add_argument(
        '--nopretrained',
        help='Set to true to use uninitialized weights for the base model (architecture is matched)',
        type=str,
        default="False")
    parser.add_argument(
        '--base_model_path',
        help='Path of the base model in TF SavedModel format',
        type=str,
        default="False")
    parser.add_argument(
        '--frozen_head',
        help='Freeze the head (base model layers) during training of auxiliary network',
        type=str,
        default="True")

    parser.add_argument(
        '--pruning_scheme',
        help='Pruning scheme to use (constant or polynomial)',
        type=str)
    parser.add_argument(
        '--epochs_recover',
        help='Additional epochs to train after reaching end sparsity, used only for polynomial training scheme',
        type=int)
    parser.add_argument(
        '--start_sparsity',
        help='Starting sparsity used in polynomial training scheme',
        type=float)
    parser.add_argument(
        '--end_sparsity',
        help='Final sparsity of trained model',
        type=float)

    Flags, unparsed = parser.parse_known_args()
    return Flags, unparsed


def plot_training(plot_dir,history):
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plt.subplot(2,1,1)
    plt.plot(history.history['sparse_categorical_accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_sparse_categorical_accuracy'], label='Val Accuracy')
    plt.title('Accuracy vs Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.subplot(2,1,2)
    plt.plot(history.history['loss'], label='Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.grid(True)
    plt.legend(loc="upper left")
    plt.savefig(plot_dir+'/acc.png')

def step_function_wrapper(batch_size):
    def step_function(epoch, lr):
        if (epoch < 12):
            return 0.0005
        elif (epoch < 24):
            return 0.0001
        elif (epoch < 36):
            return 0.00002
        else:
            return 0.00001
    return step_function

def get_callbacks(args):
    lr_sched_name = args.lr_sched_name
    batch_size = args.batch_size
    initial_lr = args.learning_rate

    train_dir = pathlib.Path(args.train_dir + "/" + args.ckpt_name)
    csv_logger = tf.keras.callbacks.CSVLogger(args.train_dir + "/" + args.ckpt_name + "/" + args.ckpt_name + '.log')

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=(train_dir / (args.ckpt_name + "_{val_sparse_categorical_accuracy:.4f}")),
        monitor='val_sparse_categorical_accuracy',
        mode='max',
        save_best_only=True)

    callbacks = [csv_logger, model_checkpoint_callback]
    if(lr_sched_name == "step_function"):
        callbacks = [csv_logger, model_checkpoint_callback, keras.callbacks.LearningRateScheduler(step_function_wrapper(batch_size),verbose=1)]
    return callbacks

def get_pruning_callbacks(args):
    lr_sched_name = args.lr_sched_name
    batch_size = args.batch_size
    initial_lr = args.learning_rate

    train_dir = pathlib.Path(args.train_dir + "/" + args.ckpt_name)
    csv_logger = tf.keras.callbacks.CSVLogger(args.train_dir + "/" + args.ckpt_name + "/" + args.ckpt_name + '.log')

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=(train_dir / (args.ckpt_name + "_{val_sparse_categorical_accuracy:.4f}")),
        monitor='val_sparse_categorical_accuracy',
        mode='max',
        save_best_only=True)

    callbacks = [csv_logger, model_checkpoint_callback]
    if(lr_sched_name == "step_function"):
        callbacks = [csv_logger, model_checkpoint_callback, tfmot.sparsity.keras.UpdatePruningStep(), keras.callbacks.LearningRateScheduler(step_function_wrapper(batch_size),verbose=1)]
    return callbacks

    

