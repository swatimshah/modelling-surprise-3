from numpy import loadtxt
from numpy import savetxt
import numpy
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot
from numpy.random import seed
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import Sequential, save_model, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D,AveragePooling1D
from numpy import mean
from tensorflow.random import set_seed
from keras.constraints import min_max_norm
from keras.regularizers import L2
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization
from sklearn.preprocessing import RobustScaler
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import tensorflow


# setting the seed
seed(1)
set_seed(1)

rScaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(20, 100-20), unit_variance=True)

my_epochs = loadtxt('E:\\recording-car\\my_epochs.csv', delimiter=',', skiprows=1)
print(my_epochs.shape)

# shuffle the training data
numpy.random.seed(2) 
numpy.random.shuffle(my_epochs)
print(my_epochs.shape)

index1 = 3823

# split the training data between training and validation
tensorflow.compat.v1.reset_default_graph()
X_train_tmp, X_test_tmp, Y_train_tmp, Y_test_tmp = train_test_split(my_epochs[0:index1, :], my_epochs[0:index1, -1], random_state=1, test_size=0.3, shuffle = False)
print(X_train_tmp.shape)
print(X_test_tmp.shape)

# augment train data
choice = X_train_tmp[:, -1] == 0.
X_total_1 = numpy.append(X_train_tmp, X_train_tmp[choice, :], axis=0)
X_total_2 = numpy.append(X_total_1, X_train_tmp[choice, :], axis=0)
X_total_3 = numpy.append(X_total_2, X_train_tmp[choice, :], axis=0)
X_total_4 = numpy.append(X_total_3, X_train_tmp[choice, :], axis=0)
X_total = numpy.append(X_total_4, X_train_tmp[choice, :], axis=0)
print(X_total.shape)

# data balancing for train data
sm = SMOTE(random_state = 2)
X_train_keep, Y_train_keep = sm.fit_resample(X_total, X_total[:, -1].ravel())
print("After OverSampling, counts of label '1': {}".format(sum(Y_train_keep == 1)))
print("After OverSampling, counts of label '0': {}".format(sum(Y_train_keep == 0)))
print(X_train_keep.shape)

train_data = numpy.append(X_train_keep, Y_train_keep.reshape(len(Y_train_keep), 1), axis=1)
numpy.random.shuffle(train_data)


#=======================================
 
# Data Pre-processing - scale data using robust scaler

input = rScaler.fit_transform(train_data[:, 0:20])
testinput = rScaler.fit_transform(X_test_tmp[:,0:20])
Y_train = train_data[:, -1]
Y_test = X_test_tmp[:, -1]

#=====================================

# Model configuration

print(len(input))
print(len(testinput))

input = input.reshape(len(input), 1, 20)
input = input.transpose(0, 2, 1)
print (input.shape)

testinput = testinput.reshape(len(testinput), 1, 20)
testinput = testinput.transpose(0, 2, 1)
print (testinput.shape)

# Create the model
model=Sequential()
model.add(Conv1D(filters=32, kernel_size=4, kernel_regularizer=L2(0.001), bias_regularizer=L2(0.001), activity_regularizer = L2(0.001), kernel_constraint=min_max_norm(min_value=-1, max_value=1), padding='valid', activation='relu', strides=1, input_shape=(20, 1)))
model.add(Conv1D(filters=32, kernel_size=4, kernel_regularizer=L2(0.001), bias_regularizer=L2(0.001), activity_regularizer = L2(0.001), kernel_constraint=min_max_norm(min_value=-1, max_value=1), padding='valid', activation='relu', strides=1))
model.add(Conv1D(filters=32, kernel_size=6, kernel_regularizer=L2(0.001), bias_regularizer=L2(0.001), activity_regularizer = L2(0.001), kernel_constraint=min_max_norm(min_value=-1, max_value=1), padding='valid', activation='relu', strides=1))
model.add((GlobalAveragePooling1D()))
model.add(Dense(2, activation='softmax'))

model.summary()

# Compile the model   
adam = Adam(learning_rate=0.000005)
model.compile(loss=sparse_categorical_crossentropy, optimizer=adam, metrics=['accuracy'])

# simple early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=300)
mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

hist = model.fit(input, Y_train, batch_size=32, epochs=200, verbose=1, validation_data=(testinput, Y_test), steps_per_epoch=None, callbacks=[es, mc])

# evaluate the model
predict_y = model.predict(testinput)
Y_hat_classes=numpy.argmax(predict_y,axis=-1)

matrix = confusion_matrix(Y_test, Y_hat_classes)
print(matrix)


# plot training and validation history
pyplot.plot(hist.history['loss'], label='tr_loss')
pyplot.plot(hist.history['val_loss'], label='val_loss')
pyplot.plot(hist.history['accuracy'], label='tr_accuracy')
pyplot.plot(hist.history['val_accuracy'], label='val_accuracy')
pyplot.legend()
pyplot.xlabel("No of iterations")
pyplot.ylabel("Accuracy and loss")
pyplot.show()

#==================================

model.save("E:\\recording-car\\model_conv1d.h5")

# load the saved model
saved_model = load_model('E:\\recording-car\\best_model.h5')
# evaluate the model
_, train_acc = saved_model.evaluate(input, Y_train, verbose=1)
_, test_acc = saved_model.evaluate(testinput, Y_test, verbose=1)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

# evaluate the model
predict_y = saved_model.predict(testinput)
Y_hat_classes=numpy.argmax(predict_y,axis=-1)

matrix = confusion_matrix(Y_test, Y_hat_classes)
print(matrix)


#==================================












