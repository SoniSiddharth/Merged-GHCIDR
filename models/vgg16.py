from keras.models import Sequential  
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization
from keras import optimizers, regularizers
import numpy as np  
import pandas as pd
import sys
sys.path.append("..")
from utils import *
import argparse
import os,sys

def define_model():
	global args
	model = Sequential()
	weight_decay = args.weight_decay
	model.add(Conv2D(64, (3, 3), padding='same',input_shape=args.imageSize,kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.3))
	model.add(Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())

	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.4))

	model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())

	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.4))

	model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.4))

	model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())

	model.add(MaxPooling2D(pool_size=(2, 2)))


	model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.4))

	model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.4))

	model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())

	model.add(MaxPooling2D(pool_size=(2, 2)))


	model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.4))

	model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.4))

	model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())

	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.5))

	model.add(Flatten())
	model.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())

	model.add(Dropout(0.5))
	model.add(Dense(args.noOfClasses))
	model.add(Activation('softmax'))
	sgd = optimizers.SGD(lr=args.lr, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])
	return model

def run_test_harness():
	global args
	trainX, trainY, testX, testY = args.getOriginalData(mode=1)
	if(args.fullDataset=='No'):
		trainX = []
		trainY = []
		for i in args.reducedData:
			trainX.append(i[0])
			trainY.append(i[1])
		trainX = np.array(trainX, dtype="float32")
		trainY = np.array(trainY, dtype="float32")
		# testX = testX.reshape(len(testX),s1,s2,s3)
	s1,s2,s3 = args.imageSize
	trainX = trainX.reshape(len(trainX),s1,s2,s3)
	trainY = np_utils.to_categorical(trainY, args.noOfClasses)
	model = define_model()    
	print(trainX.shape)
	print(testX.shape)
	print(trainY.shape)
	print(testY.shape)
	history = model.fit(trainX, trainY, epochs=args.epochs, batch_size=args.batchSize, validation_data=(testX, testY))
	# evaluate model
	_, acc = model.evaluate(testX, testY)
	print('> %.3f' % (acc * 100.0))

parser = argparse.ArgumentParser()
parser.add_argument("-epochs", default='100', type=str)
parser.add_argument("-datasetName", default='MNIST', type=str)
parser.add_argument("-variantName", default='RHC', type=str)
parser.add_argument("-lr", default='0.001', type=float)
parser.add_argument("-batchSize", default=64, type=int) 
parser.add_argument("-fullDataset", default='No', type=str) 
parser.add_argument("-weightdecay",default=0.0005,type=float)
args = parser.parse_args()
Datasets = { 
			"MNIST":[getMnistData,(28,28,1),10],
			"FMNIST":[getFMnistData,(28,28,1),10], 
			"CIFAR10":[getCifar10Data,(32,32,3),10],
			"TinyImagenet":[getTinyImagenetData,(64,64,3),100]
			}
# args.uniqueClasses
getOriginalData, imageSize, noOfClasses = Datasets[args.datasetName]
if(args.variantName == "MGHCIDR"):
    path =  "./datasetPickle/" +args.datasetName + '_' + args.variantName + '_'+ str(args.beta)+".pickle"
else:
    path =  "./datasetPickle/" +args.datasetName + '_' + args.variantName +".pickle"

reducedData = loadFromPickle(path)
args.getOriginalData = getOriginalData
args.imageSize = imageSize
args.noOfClasses = noOfClasses
args.reducedData = reducedData
run_test_harness()