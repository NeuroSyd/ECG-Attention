#import sys
#import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (ModelCheckpoint,
                             TensorBoard, ReduceLROnPlateau,
                             CSVLogger, EarlyStopping)

#from tensorflow.keras.backend import set_session
from model_attn import get_model_attn
import argparse
#from tensorflow.keras.utils import HDF5Matrix
#import pandas as pd
#import h5py
#import numpy as np
from datasets_attn import ECGSequence


if __name__ == "__main__":
    # Get data and train
    parser = argparse.ArgumentParser(description='Train neural network.')
    parser.add_argument('path_to_hdf5', type=str,
                        help='path to hdf5 file containing tracings')
    parser.add_argument('path_to_csv', type=str,
                        help='path to csv file containing annotations')
    parser.add_argument('--val_split', type=float, default=0.02,
                        help='number between 0 and 1 determining how much of'
                             ' the data is to be used for validation. The remaining '
                             'is used for validation. Default: 0.02')
    parser.add_argument('--dataset_name', type=str, default='tracings',
                        help='name of the hdf5 dataset containing tracings')
    args = parser.parse_args()
    # Optimization settings
    loss = 'binary_crossentropy'
    lr = 0.001
    batch_size = 64
    opt = Adam(lr)
    callbacks = [ReduceLROnPlateau(monitor='val_loss',
                                   factor=0.1,
                                   patience=7,
                                   min_lr=lr / 100),
                 EarlyStopping(patience=9,  # Patience should be larger than the one in ReduceLROnPlateau
                               min_delta=0.00001)]
                               
    train_seq, valid_seq = ECGSequence.get_train_and_val(
        args.path_to_hdf5, args.dataset_name, args.path_to_csv, batch_size, args.val_split)
                                   
    # Set session and compile model
    #config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    #set_session(tf.Session(config=config))
    
    # If you are continuing an interrupted section, uncomment line bellow:
    #   model = keras.models.load_model(PATH_TO_PREV_MODEL, compile=False)
    
    model = get_model_attn(train_seq.n_classes)
    #config = model.get_config()
    
    # At loading time, register the custom objects with a `custom_object_scope`:
    ##custom_objects = {"attentionLayer": attentionLayer}
    #from keras.utils import CustomObjectScope
    #from keras.models import load_model
    #with CustomObjectScope({"attention": attention}):
      #model = load_model('model_attn.hdf5')
    ##with keras.utils.custom_object_scope(custom_objects):
      ##new_model = keras.Model.from_config(config)
    
      ##new_model.compile(loss=loss, optimizer=opt)
    model.compile(loss=loss, optimizer=opt)
    
    # Get annotations
    #y = pd.read_csv(args.path_to_csv).values
    # Get tracings
    #f = h5py.File(args.path_to_hdf5, "r")
    #x = f[args.dataset_name]

    # Create log
    callbacks += [TensorBoard(log_dir='./logs_attn', write_graph=False),
                  CSVLogger('training_attn.log', append=False)]  # Change append to true if continuing training
    # Save the BEST and LAST model
    callbacks += [ModelCheckpoint('./backup_model_attn.hdf5'),
                  ModelCheckpoint('./backup_model_attn_best.hdf5', save_best_only=True)]
    
    # Train neural network
    history = model.fit(train_seq,
                        epochs=70,
                        initial_epoch=0,  # If you are continuing a interrupted section change here
                        callbacks=callbacks,
                        validation_data=valid_seq,
                        verbose=1)
    # Save final result
    model.save("./model_attn_TNMG.hdf5")
    #f.close()
