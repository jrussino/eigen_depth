#!/usr/bin/env python

from __future__ import print_function

print('### IMPORTING MODULES ###')

import cv2
import datetime
import numpy as np
import os
import time
import yaml

from keras.models import Model, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Reshape, merge
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import History, ModelCheckpoint
from keras import backend as K
K.clear_session() # workaround for session bug; see https://github.com/tensorflow/tensorflow/issues/3388

np.random.seed(None) 
dateTimeStr = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H%M%S')


## TRAINED COARSE
MODEL_FILE = '/home/jrussino/sandbox/eigen_depth/models/2017-05-09-230038/depth_coarse_model_2017-05-09-230038.json'
WEIGHTS_FILE = '/home/jrussino/sandbox/eigen_depth/models/2017-05-09-230038/depth_coarse_weights_2017-05-09-230038.h5'
## TRAINED FINE
#MODEL_FILE = '/home/jrussino/sandbox/eigen_depth/models/2017-05-09-205427/depth_fine_model_2017-05-09-205427.json'
#WEIGHTS_FILE = '/home/jrussino/sandbox/eigen_depth/models/2017-05-09-205427/depth_fine_weights_2017-05-09-205427.h5'


""" UTILS """
def reshapeAndScale(data):
    if len(data.shape) == 3:
        data = data.reshape(data.shape[0], data.shape[2], data.shape[1])
    else:
        data = data.reshape(data.shape[0], data.shape[2], data.shape[1], data.shape[3])
    data = data.astype('float32')
    data /= 255
    return data

def loadData(dataDir):
    X_files = [os.path.join(dataDir, f) for f in os.listdir(dataDir) if '_image' in f] 
    X = np.array([cv2.pyrDown(cv2.imread(f)) for f in X_files])
    X = reshapeAndScale(X)

    Y_files = [f.replace('_image', '_depth') for f in X_files]
    Y = np.array([cv2.pyrDown(cv2.pyrDown(cv2.pyrDown(cv2.imread(f, 0)))) for f in Y_files])
    Y = reshapeAndScale(Y)
    return X, Y



""" DEPTHPREDICTOR """
class DepthPredictor(object):
    def __init__(self, configFile):
        with open(configFile) as cfg:
            self.config = yaml.load(cfg)
        self.MODE = self.config['MODE']

    def scale_invariant_error(self, y_true, y_pred):
        first_log = K.log(K.clip(y_pred, K.epsilon(), np.inf) + 1.)
        second_log = K.log(K.clip(y_true, K.epsilon(), np.inf) + 1.)
        return K.mean(K.square(first_log - second_log), axis=-1) - self.config['LAMBDA'] * K.square(K.mean(first_log - second_log, axis=-1))

    def train_coarse(self):
        # Input:
        #TODO check and enforce self.config['IMG_ROWS'], self.config['IMG_COLS'] are even
        inputs = Input(shape=(int(self.config['IMG_ROWS']/2), int(self.config['IMG_COLS']/2), 3))

        # Coarse 1:
        # 11x11 conv, 4 stride, ReLU activation, 2x2 pool
        coarse_1 = Convolution2D(96, (11, 11), strides=(4,4), padding='same', kernel_initializer='uniform', input_shape=(1,self.config['IMG_ROWS']/2, self.config['IMG_COLS']/2), name='coarse_1')(inputs)
        coarse_1 = Activation('relu')(coarse_1)
        coarse_1 = MaxPooling2D(pool_size=(2, 2))(coarse_1)

        # Coarse 2:
        # 5x5 conv, 1 stride, ReLU activation, 2x2 pool
        coarse_2 = Convolution2D(256, (5, 5), padding='same', kernel_initializer='uniform', name='coarse_2')(coarse_1)
        coarse_2 = Activation('relu')(coarse_2)
        coarse_2 = MaxPooling2D(pool_size=(2, 2))(coarse_2)

        # Coarse 3:
        # 3x3 conv, 1 stride, ReLU activation, no pool
        coarse_3 = Convolution2D(384, (3, 3), padding='same', kernel_initializer='uniform', name='coarse_3')(coarse_2)
        coarse_3 = Activation('relu')(coarse_3)

        # Coarse 4:
        # 3x3 conv, 1 stride, ReLU activation, no pool
        coarse_4 = Convolution2D(384, (3, 3), padding='same', kernel_initializer='uniform', name='coarse_4')(coarse_3)
        coarse_4 = Activation('relu')(coarse_4)

        # Coarse 5:
        # 3x3 conv, 1 stride, ReLU activation, 2x2 pool?
        coarse_5 = Convolution2D(256, (3, 3), padding='same', kernel_initializer='uniform', name='coarse_5')(coarse_4)
        coarse_5 = Activation('relu')(coarse_5)
        coarse_5 = MaxPooling2D(pool_size=(2, 2))(coarse_5)

        # Coarse 6:
        # Fully-connected, ReLU activation, followed by dropout
        coarse_6 = Flatten(name='coarse_6')(coarse_5)
        coarse_6 = Dense(4096, kernel_initializer='uniform')(coarse_6)
        coarse_6 = Activation('relu')(coarse_6)
        coarse_6 = Dropout(0.5)(coarse_6)

        # Coarse 7:
        # Fully-connected, linear activation
        coarse_7 = Dense((int(self.config['IMG_ROWS']/8))*(int(self.config['IMG_COLS']/8)), kernel_initializer='uniform', name='coarse_7')(coarse_6) #XXX
        coarse_7 = Activation('linear')(coarse_7)
        coarse_7 = Reshape((int(self.config['IMG_ROWS']/8), int(self.config['IMG_COLS']/8)))(coarse_7) #XXX

        # compile the model
        #TODO compile model once and save (separate script)
        print('### COMPILING MODEL ###')
        model = Model(input=inputs, output=coarse_7)
        model.compile(loss=self.scale_invariant_error, optimizer=SGD(lr=self.config['LEARNING_RATE'], momentum=self.config['MOMENTUM']), metrics=['accuracy'])
        model.summary()

        # save the model architecture to file
        print('### SAVING MODEL ARCHITECTURE ###')
        modelDir = dateTimeStr;
        os.mkdir(os.path.join(self.config['OUTDIR'], modelDir))
        modelFile = os.path.join(self.config['OUTDIR'], modelDir, 'depth_coarse_model_{}.json'.format(dateTimeStr))
        print(model.to_json(), file=open(modelFile, 'w'))

        # load and preprocess the data
        print('### LOADING DATA ###')
        X_train, Y_train = loadData(os.path.join(self.config['DATA_DIR'], 'train/'))
        X_test , Y_test = loadData(os.path.join(self.config['DATA_DIR'], 'test/'))
        print('X_train shape:', X_train.shape)
        print('Y_train shape:', Y_train.shape)
        print(X_train.shape[0], 'train samples')
        print(X_test.shape[0], 'test samples')

        # train the model
        #TODO use validation_split instead of using separate (test) data
        print('### TRAINING ###')
        history_cb = History()
        checkpointFile = os.path.join(self.config['OUTDIR'], modelDir, 'coarse-weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5') 
        checkpoint_cb = ModelCheckpoint(filepath=checkpointFile, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto')
        model.fit(X_train, Y_train, epochs=self.config['EPOCHS'], batch_size=self.config['BATCH_SIZE'],
                verbose=1, validation_data=(X_test, Y_test), callbacks=[history_cb, checkpoint_cb])
        histFile = os.path.join(self.config['OUTDIR'], modelDir, 'depth_coarse_hist_{}.h5'.format(dateTimeStr))

        # save the model weights to file
        print('### SAVING TRAINED MODEL ###')
        print(history_cb.history, file=open(histFile, 'w'))
        weightsFile = os.path.join(self.config['OUTDIR'], modelDir, 'depth_coarse_weights_{}.h5'.format(dateTimeStr))
        model.save_weights(weightsFile)

        # evaluate the trained model
        print('sleeping for 5 seconds...')
        time.sleep(5)
        print('### LOADING THE MODEL WEIGHTS ###')
        model_json = open(modelFile, 'r').read()
        model2 = model_from_json(model_json)
        model2.load_weights(weightsFile)
        model2.compile(loss=self.scale_invariant_error, optimizer=SGD(lr=self.config['LEARNING_RATE'], momentum=self.config['MOMENTUM']), metrics=['accuracy'])

        # evaluate the model
        print('### EVALUATING ###')
        score = model2.evaluate(X_test, Y_test, verbose=1)
        print('Test score:', score[0])
        print('Test accuracy:', score[1])

    def train_fine(self):
        # load coarse model
        print('### LOADING SAVED MODEL AND WEIGHTS ###')
        model_json = open(MODEL_FILE, 'r').read()
        model = model_from_json(model_json)
        model.load_weights(WEIGHTS_FILE)
        
        # freeze training on coarse layers
        for layer in model.layers:
            layer.trainable = False

        # modify with additional fine layers
        print('### UPDATING MODEL ###')
        # Input:
        inputs = model.inputs[0]

        # Fine 1:
        # 9x9 conv, 2 stride, ReLU activation, 2x2 pool
        fine_1 = Convolution2D(63, (9, 9), padding='same', kernel_initializer='uniform', strides=(2,2), input_shape=(1, int(self.config['IMG_ROWS']/2), int(self.config['IMG_COLS']/2)), name='fine_1_conv')(inputs) #XXX
        fine_1 = Activation('relu', name='fine_1_relu')(fine_1)
        fine_1 = MaxPooling2D(pool_size=(2, 2), name='fine_1_pool')(fine_1)

        # Fine 2:
        # Concatenation with Coarse 7
        coarse_out = model.outputs[0]
        coarse_out = Reshape((int(self.config['IMG_ROWS']/8), int(self.config['IMG_COLS']/8), 1), name='coarse_out_reshape')(coarse_out) #XXX
        fine_2 = merge([fine_1, coarse_out], mode='concat', concat_axis=3, name='fine_2_merge')

        # Fine 3:
        # 5x5 conv, 1 stride, ReLU activation, no pool
        fine_3 = Convolution2D(64, (5, 5), padding='same', kernel_initializer='uniform', strides=(1,1), name='fine_3_conv')(fine_2)
        fine_3 = Activation('relu', name='fine_3_relu')(fine_3)

        # Fine 4:
        # 5x5 conv, 1 stride, linear activation, no pool
        fine_4 = Convolution2D(1, (5, 5), padding='same', kernel_initializer='uniform', strides=(1,1), name='fine_4_conv')(fine_3)
        fine_4 = Activation('linear', name='fine_4_linear')(fine_4)
        fine_4 = Reshape((int(self.config['IMG_ROWS']/8), int(self.config['IMG_COLS']/8)), name='fine_4_reshape')(fine_4) #XXX

        # compile the model
        print('### COMPILING MODEL ###')
        model = Model(input=inputs, output=fine_4)
        model.compile(loss=self.scale_invariant_error, optimizer=SGD(lr=self.config['LEARNING_RATE'], momentum=self.config['MOMENTUM']), metrics=['accuracy'])
        model.summary()

        # save the model architecture to file
        print('### SAVING MODEL ARCHITECTURE ###')
        modelDir = dateTimeStr;
        os.mkdir(os.path.join(self.config['OUTDIR'], modelDir))
        modelFile = os.path.join(self.config['OUTDIR'], modelDir, 'depth_fine_model_{}.json'.format(dateTimeStr))
        print(model.to_json(), file=open(modelFile, 'w'))

        # load and preprocess the data
        print('### LOADING DATA ###')
        X_train, Y_train = loadData(os.path.join(self.config['DATA_DIR'], 'train/'))
        X_test , Y_test = loadData(os.path.join(self.config['DATA_DIR'], 'test/'))
        print('X_train shape:', X_train.shape)
        print('Y_train shape:', Y_train.shape)
        print(X_train.shape[0], 'train samples')
        print(X_test.shape[0], 'test samples')

        # train the model
        #TODO use validation_split instead of using separate (test) data
        print('### TRAINING ###')
        history_cb = History()
        checkpointFile = os.path.join(self.config['OUTDIR'], modelDir, 'fine-weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5') 
        checkpoint_cb = ModelCheckpoint(filepath=checkpointFile, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto')
        model.fit(X_train, Y_train, epochs=self.config['EPOCHS'], batch_size=self.config['BATCH_SIZE'],
                verbose=1, validation_data=(X_test, Y_test), callbacks=[history_cb, checkpoint_cb])
        histFile = os.path.join(self.config['OUTDIR'], modelDir, 'depth_fine_hist_{}.json'.format(dateTimeStr))

        # save the model weights to file
        print('### SAVING TRAINED MODEL ###')
        print(history_cb.history, file=open(histFile, 'w'))
        weightsFile = os.path.join(self.config['OUTDIR'], modelDir, 'depth_fine_weights_{}.h5'.format(dateTimeStr))
        model.save_weights(weightsFile)

        # evaluate the trained model
        print('sleeping for 5 seconds...')
        time.sleep(5)
        print('### LOADING THE MODEL WEIGHTS ###')
        model_json = open(modelFile, 'r').read()
        model2 = model_from_json(model_json)
        model2.load_weights(weightsFile)
        model2.compile(loss=self.scale_invariant_error, optimizer=SGD(lr=self.config['LEARNING_RATE'], momentum=self.config['MOMENTUM']), metrics=['accuracy'])

        # evaluate the model
        print('### EVALUATING ###')
        score = model2.evaluate(X_test, Y_test, verbose=1)
        print('Test score:', score[0])
        print('Test accuracy:', score[1])

    def train(self):
        pass

    def eval(self):
        # load coarse model
        print('### LOADING SAVED MODEL AND WEIGHTS ###')
        model_json = open(MODEL_FILE, 'r').read()
        model = model_from_json(model_json)
        model.load_weights(WEIGHTS_FILE)
        model.compile(loss=self.scale_invariant_error, optimizer=SGD(lr=self.config['LEARNING_RATE'], momentum=self.config['MOMENTUM']), metrics=['accuracy'])
        model.summary()

        # load and preprocess the data
        print('### LOADING DATA ###')
        X_train, Y_train = loadData(os.path.join(self.config['DATA_DIR'], 'train/'))
        X_test , Y_test = loadData(os.path.join(self.config['DATA_DIR'], 'test/'))
        print('X_train shape:', X_train.shape)
        print('Y_train shape:', Y_train.shape)
        print(X_train.shape[0], 'train samples')
        print(X_test.shape[0], 'test samples')

        # evaluate the model
        print('### EVALUATING ###')
        score = model.evaluate(X_test, Y_test, verbose=1)
        print('Test score:', score[0])
        print('Test accuracy:', score[1])


depthPredictor = DepthPredictor('./config.yml')
print(depthPredictor.MODE)
if depthPredictor.MODE == 'train_coarse':
    depthPredictor.train_coarse()

if depthPredictor.MODE == 'train_fine':
    depthPredictor.train_fine()

if depthPredictor.MODE == 'eval':
    depthPredictor.eval()
