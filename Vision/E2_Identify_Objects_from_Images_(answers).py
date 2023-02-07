# baseline DCN model
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from tensorflow.keras.optimizers import RMSprop

"""
Q16: Before fitting the model, show your teacher the code you used to define the model
    described here. (Question 16, 6 points)
A16: see the code

Q17: Plot the training history and show it to your teacher (Question 17, 2 points)
A17: see plots

Q18: Discuss with your group, then describe to your teacher, how the training history
     differs from the convolutional model for digit recognition and why. (Question 18, 5 points)
A18: Similarly to the previous DCN with dropout, but unlike the previous DCN without drop out, 
    the training and validation performance matches clossly during the training process, with small deviations.
    Intrestingly, validation performance in this DCN model is even slighlty better than the trainings performance. 
    Noticeably, this DCN shows a strongly reduced maximal performances (for object recognition), 
    than the previous model for hand-written digit recognition:
        
    Digit_DCN:  loss: 0.0443 - accuracy: 0.9870 - val_loss: 0.0383 - val_accuracy: 0.9894    
    Object_DCN: loss: 0.8883 - accuracy: 0.6876 - val_loss: 0.8256 - val_accuracy: 0.7104
    
    This might be attributable to the complexity of the task. Similarly, the performance plateau is reach 
    fairly fast for digit recognition, while for this model an extended training period (e.g. epoch =30) 
    might still allow further accuracy improvements. In addition, the initial performance is worse for the 
    object_DCN supporting the claim that differences in the observed performance are attributable to task complexity.

Q19: Discuss with your group, then describe to your teacher, how the time taken for each
     training epoch differs from the convolutional model for digit recognition. Give several
     factors that may contribute to this difference (Question 19, 4 points)
A19: The epoch times of the object-recognition-DCN differs from the previous DCN by several 
    seconds (~10s difference; now 86s). Mossible reasons might be: More layers, large fully 
    connected layer, 3 color channels, different optimizer
"""
# load train and test dataset
def load_dataset():
	# load dataset
    (trainX, trainY), (testX, testY) = cifar10.load_data()
	# reshape dataset to have a single channel
    trainX = trainX.reshape((trainX.shape[0], 32, 32, 3))
    testX = testX.reshape((testX.shape[0], 32, 32, 3))
	# one hot encode target values
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    return trainX, trainY, testX, testY

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

# define model
def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation = 'relu', input_shape = (32, 32, 3), padding = 'same'))
    model.add(Conv2D(32, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(32, (3, 3), activation = 'relu', padding = 'same'))
    model.add(Conv2D(32, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation = 'softmax'))
    # compile model
    model.compile(optimizer=RMSprop(lr = 0.0001, decay = 1e-6), loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model

def evaluate_model(trainX, trainY, testX, testY):
    # define model
    model = define_model()
    model.summary()
    # fit model
    history = model.fit(trainX, trainY, epochs = 20, batch_size = 32, verbose = 1, validation_data = (testX, testY), shuffle = 'TRUE')
    score = model.evaluate(testX, testY, verbose = 0)
    # list accuracy and loss  all data in history
    print(score)
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


# run the test harness for evaluating a model
def run_DCN():
	# load dataset
    trainX, trainY, testX, testY = load_dataset()
	# prepare pixel data
    trainX, testX = prep_pixels(trainX, testX)
	# evaluate model
    evaluate_model(trainX, trainY, testX, testY)
	 
# entry point, run the SCN
run_DCN()
