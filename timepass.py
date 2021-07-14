# baseline model with dropout on the cifar10 dataset
import sys
from matplotlib import pyplot
from keras.datasets import fashion_mnist, mnist
# from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.optimizers import SGD
from keras.datasets import mnist
from keras.models import Sequential  
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils      
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, Flatten
from keras.layers.normalization import BatchNormalization
from keras.datasets import cifar10
from keras.utils import np_utils

%matplotlib inline
import numpy as np  
import pandas as pd
import matplotlib.pyplot as plt      
import random
import pickle                       
import cv2
def RHC_Output():
    from keras.datasets import mnist
    from keras.models import Sequential  
    from keras.layers.core import Dense, Dropout, Activation
    from keras.utils import np_utils      
    from keras.preprocessing.image import ImageDataGenerator
    from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, Flatten
    from keras.layers.normalization import BatchNormalization
    from keras.datasets import cifar10
    from keras.utils import np_utils
    
    %matplotlib inline
    import numpy as np  
    import pandas as pd
    import matplotlib.pyplot as plt      
    import random
    import pickle                       
    import cv2
    
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    X_train = X_train.astype('float32')         # change integers to 32-bit floating point numbers
    X_test = X_test.astype('float32')
    X_train /= 255                              # normalize each value for each pixel for the entire vector for each input
    X_test /= 255
    uniqueClasses = len(np.unique(y_train))
    Y_train = y_train
    Y_test = np_utils.to_categorical(y_test, uniqueClasses)

    # to load the reduced data from pickle object

    dbfile = open('/content/drive/MyDrive/ML project/Dataset Pickle/final_datasets/FMNIST_algo_Centroid.pickle', 'rb')      
    CondensedSet = pickle.load(dbfile) 
    reducedData = CondensedSet
    print(len(reducedData))
    Y_train = []
    X_train = []
    for i in reducedData:
      Y_train.append(i[1])
      X_train.append(i[0])
    X_train = np.array(X_train)

    X_test = X_test.astype('float32')
    labels = 10
    Y_train = np_utils.to_categorical(Y_train, labels)
    Y_test = np_utils.to_categorical(y_test, labels)
    trainX = X_train.reshape(len(X_train),28,28,1)
    testX = X_test.reshape(10000,28,28,1)
    print("datasets are ready")
    return trainX, Y_train, testX, Y_test

# load train and test dataset
def load_dataset():
  # load dataset
  (trainX, trainY), (testX, testY) = mnist.load_data()
  # one hot encode target values
  labels = 10
  trainY = np_utils.to_categorical(trainY,labels)
  testY = np_utils.to_categorical(testY,labels)
  return trainX, trainY, testX, testY

# scale pixels
def prep_pixels(train, test):
  # convert from integers to floats
  train_norm = train.astype('float32')
  test_norm = test.astype('float32')
  train_norm = train_norm.reshape(len(train_norm),28,28,1)
  test_norm = test_norm.reshape(10000,28,28,1)
  # normalize to range 0-1
  train_norm = train_norm / 255.0
  test_norm = test_norm / 255.0
  # return normalized images
  return train_norm, test_norm

# define cnn model
def define_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(28, 28, 1)))
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.2))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.2))
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dropout(0.2))
	model.add(Dense(10, activation='softmax'))
	# compile model
	opt = SGD(learning_rate=0.001, momentum=0.9)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model

# plot diagnostic learning curves
def summarize_diagnostics(history):
	# plot loss
	pyplot.subplot(211)
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color='blue', label='train')
	pyplot.plot(history.history['val_loss'], color='orange', label='test')
	# plot accuracy
	pyplot.subplot(212)
	pyplot.title('Classification Accuracy')
	pyplot.plot(history.history['accuracy'], color='blue', label='train')
	pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
	# save plot to file
	filename = sys.argv[0].split('/')[-1]
	pyplot.savefig(filename + '_plot.png')
	pyplot.close()

    
# run the test harness for evaluating a model
def run_test_harness():
	# load dataset
	trainX, trainY, testX, testY = load_dataset()
#   (trainX, trainY), (testX, testY) = fashion_mnist.load_data()
  # print(trainY.shape)
  # print(trainX.shape)
	# prepare pixel data
	trainX, testX = prep_pixels(trainX, testX)
# 	uniqueClasses = 10
# 	trainY = np_utils.to_categorical(trainY, uniqueClasses)
# 	testY = np_utils.to_categorical(testY, uniqueClasses)
  # define model
	model = define_model()
  # fit model
	print(trainX.shape)
	print(testX.shape)
	print(trainY.shape)
	print(testY.shape)
	history = model.fit(trainX, trainY, epochs=100, batch_size=64, validation_data=(testX, testY))
	# evaluate model
	_, acc = model.evaluate(testX, testY)
	print('> %.3f' % (acc * 100.0))
	# learning curves
	# summarize_diagnostics(history)

# entry point, run the test harness
for i in range(5):
  run_test_harness()

import numpy as np                   
import matplotlib.pyplot as plt      
import random                        
from keras.datasets import mnist
from keras.models import Sequential  
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils      

# Y_train = []
# X_train = []
# for i in CondensedSet:
#   Y_train.append(i[1])
#   X_train.append(i[0])
X_train = np.array(X_train)
X_test = X_test.astype('float32')
labels = 10
Y_train = np_utils.to_categorical(Y_train, labels)

for j in range(5):
  # Y_test = np_utils.to_categorical(y_test, labels)
  model = Sequential()
  model.add(Dense(512, input_shape=(784,)))
  model.add(Activation('relu'))
  model.add(Dropout(0.2))
  model.add(Dense(512))
  model.add(Activation('relu'))
  model.add(Dropout(0.2))
  model.add(Dense(10))
  model.add(Activation('softmax'))
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  print(type(X_train))
  model.fit(X_train, Y_train,batch_size=64, epochs=100,verbose=1)

  print(X_train.shape)
  print(X_test.shape)
  print(Y_train.shape)
  print(Y_test.shape)
  score = model.evaluate(X_test, Y_test)
  print(score[1])

  import sys
from matplotlib import pyplot
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import categorical_crossentropy
import keras
from keras.datasets import cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import optimizers
import numpy as np
from keras.layers.core import Lambda
from keras import backend as K
from keras import regularizers
import pickle
def RHC_Output():
    %matplotlib inline
    import numpy as np                   
    import matplotlib.pyplot as plt      
    import random 
    import pickle                       
    %matplotlib inline
    import numpy as np  
    import pandas as pd 
    import matplotlib.pyplot as plt      
    import random   
    print("first")
    import pickle
    dbfile = open("./Copy of tiny-imagenet.pickle","rb")
    X_train, y_train, X_test, y_test = pickle.load(dbfile)
    X_train = []
    y_train = []
#     X_train = X_train.astype('float32')         # change integers to 32-bit floating point numbers
    X_test = X_test.astype('float32')
#     X_train /= 255                              # normalize each value for each pixel for the entire vector for each input
    X_test /= 255
    uniqueClasses = len(np.unique(y_train))
    Y_train = y_train
    Y_test = to_categorical(y_test, uniqueClasses)

    dbfile = open('./Copy of tinyImagenet_GHCIDRCompleteLinkage_50_alpha40_AlmostRHC.pickle', 'rb')      
    CondensedSet = pickle.load(dbfile) 
    print(len(CondensedSet))
    print("second")
    Y_train = []
    X_train = []
    for i in CondensedSet:
        Y_train.append(i[1])
        X_train.append(i[0])
    X_train = np.array(X_train).astype("float32")
    X_test = np.array(X_test).astype("float32")
    labels = 100
    Y_train = to_categorical(Y_train, labels)
    Y_test = to_categorical(y_test, labels)
    trainX = X_train.reshape(len(X_train),64,64,3)
    testX = X_test.reshape(10000,64,64,3)
    print("thrid")
    return trainX, Y_train,testX,Y_test
    

import pickle
# load train and test dataset
def load_dataset():
	# load dataset
	dbfile = open("../input/fulltinyimagenet/tiny-imagenet.pickle","rb")
	X_train, y_train, X_test, y_test = pickle.load(dbfile)
	# one hot encode target values
	y_train = to_categorical(y_train)
	y_test = to_categorical(y_test)
	return X_train, y_train, X_test, y_test

# scale pixels
def prep_pixels(train, test):
	# convert from integers to floats
	train_norm = train.astype('float32')
	test_norm = test.astype('float32')
	# normalize to range 0-1
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	# return normalized images
	return train_norm, test_norm

# define cnn model
def define_model():
    model = Sequential()
    weight_decay = 0.0005

    model.add(Conv2D(64, (3, 3), padding='same',
                     input_shape=[64,64,3],kernel_regularizer=regularizers.l2(weight_decay)))
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
    model.add(Dense(100))
    model.add(Activation('softmax'))
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])

    return model

# plot diagnostic learning curves
def summarize_diagnostics(history):
	# plot loss
	pyplot.subplot(211)
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color='blue', label='train')
	pyplot.plot(history.history['val_loss'], color='orange', label='test')
	# plot accuracy
	pyplot.subplot(212)
	pyplot.title('Classification Accuracy')
	pyplot.plot(history.history['accuracy'], color='blue', label='train')
	pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
	# save plot to file
	filename = sys.argv[0].split('/')[-1]
	pyplot.savefig(filename + '_plot.png')
	pyplot.close()

    
# run the test harness for evaluating a model
def run_test_harness():
	# load dataset
	trainX, trainY, testX, testY = RHC_Output()
	model = define_model()
	print("Running")
	history = model.fit(trainX, trainY, epochs=100, batch_size=64, validation_data=(testX, testY))
	_, acc = model.evaluate(testX, testY)
	print('> %.3f' % (acc * 100.0))

run_test_harness() 