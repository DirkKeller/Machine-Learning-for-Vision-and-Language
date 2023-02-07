# baseline DCN model
import matplotlib.pyplot as plt
from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from tensorflow.keras.optimizers import Adadelta

"""
Q10: Plot the training history and show this to your teacher. (Question 10, 2 points)
A10: see history and plots
    
Q11: Discuss with your group, then describe to your teacher, how the training history
     differs from the previous model, for the training and validation sets. What does this
     tell us about the generalisation of the model? (Question 11, 5 points)
A11: All three measures, the initial performance (epoche =1), the maximum performance 
     (loss/accuracy at epoche =6) and the difference between training and validation set 
     are superior for the DCN, when compared to MLP.
     Intitial:   MLP: loss: 0.3220 - accuracy: 0.9106 - val_loss: 0.1661 - val_accuracy: 0.9528
                 DCN: loss: 0.1256 - accuracy: 0.9605 - val_loss: 0.0521 - val_accuracy: 0.9852
     Maximum:    MLP: loss: 0.0121 - accuracy: 0.9968 - val_loss: 0.0889 - val_accuracy: 0.9784
                 DCN: loss: 0.0056 - accuracy: 0.9984 - val_loss: 0.0451 - val_accuracy: 0.9905
     T/V-D:      MLP: loss: 0.0734 - accuracy: 0.0247
                 DCN: loss: 0.0395 - accuracy: 0.0078 
     The DCN performance is overall superior, overfitting is minimal/absent and hence its generalizability
     is better (or excellent) for this task.

Q12: Show your teacher what values you get for the model’s accuracy and loss. (Question12, 2 points)
A1: loss: 0.0056 - accuracy: 0.9984 - val_loss: 0.0451 - val_accuracy: 0.9905
    
Q13: Discuss with your group, then describe to your teacher, whether this accuracy is
     sufficient for some uses of automatic hand-written digit classification. (Question 13, 5 points)
A13: An misclassifcation rate of 0.0095 would be considered as sufficient for hand-written digit classification.
     The FP, FN rate would be very low, although 1 n ~105 pcitures would be classfied incorrectly, 
     thus again it depends on the goal of the tasks (e.g. classify bank chaques (not sufficient), 
     drawing board number recognition (sufficient))

Q14: Discuss with your group, then describe to your teacher, how the training history
     differs from the previous (convolutional) model, for both the training and validation
     sets, and for the time taken to run each model epoch (Question 14, 3 points)
A14: The droupout model has lower training accuracy, while performing similiar on the validation test set. 
     The difference between trainings and validation performance is less pronounced (training performance 
     exceeding validation performance), hence the model is less prone to overfit to the idiosyncracies of the training set.
     The computation time is slighly increased from ~ 68s to 72s.
     
     With_dropout:    loss: 0.0056 - accuracy: 0.9984 - val_loss: 0.0451 - val_accuracy: 0.9905
     Without_dropout: loss: 0.0443 - accuracy: 0.9870 - val_loss: 0.0383 - val_accuracy: 0.9894
    
Q15: Discuss with your group, then describe to your teacher, what this tells us about the
     generalisation of the two models. (Question 15, 3 points)
A15: With Dropout, the training process essentially drops out neurons in a neural network, 
     as well as the connections or synapses, and hence no data flows through these neurons anymore. 
     They are temporarily removed from the network (e.g. Srivastava, 2014). This process repeats every epoch 
     and hence sampling thinned networks is occurs frequently. Dropout, then, prevents these co-adaptations by 
     making the presence of other hidden neurons unreliable. Neurons simply cannot rely on other units to 
     correct their mistakes, which reduces the number of co-adaptations that do not generalize to unseen data, 
     and thus presumably reduces overfitting as well. The removal of neurons and synapses, and hence the data flow 
     that originally passed through these neurons during training, is performed at random (e.g. using Bernoulli variables), 
     with a parameter P that is tunable. The units are temporarily removed from the network during every epoch, 
     which generally results in sampling thinned networks. The last step is to sample the ‘thinned’ network from the global 
     architecture and used it for training. 
     This can be seen in the smaller difference between the training and validation loss and accuracy for the model with 
     dropout, as compared to the model without
"""

# load train and test dataset
def load_dataset():
	# load dataset
    (trainX, trainY), (testX, testY) = mnist.load_data()
	# reshape dataset to have a single channel
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))
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
    model.add(Conv2D(32, (3, 3), activation = 'relu', input_shape = (28, 28, 1)))
    model.add(Conv2D(64, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation = 'softmax'))
    # compile model
    model.compile(optimizer = Adadelta(learning_rate = 1), loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model

def evaluate_model(trainX, trainY, testX, testY):
    # define model
    model = define_model()
    model.summary()
    # fit model
    history = model.fit(trainX, trainY, epochs = 6, batch_size = 32, verbose = 1, validation_split = 0.2)
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
	 
# entry point, run the DCN
run_DCN()
