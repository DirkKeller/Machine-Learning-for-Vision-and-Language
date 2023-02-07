# baseline MLP model 
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop

"""
##### more concrete
Q1: Discuss with your group, then describe to your teacher, a list of applications where 
    automatic recognition of hand-written numbers would be useful. (Question 1, 3)
    points)
A1: Digitalize hand written letters, manual bookkeeping, or archive material (e.g. bookkeeping)

Q2: Show your teacher the text from your console, with how long it took for each epoch 
    to run and the training performance history. (Question 2, 5 points)
A2: 3-5s for each epoche, with an initial high/low trainings accuracy/loss of 
    loss: loss: 0.3940 - accuracy: 0.8868 respectively. At epoche 12 reaching: loss of 0.2644 - accuracy: 0.9265 
    
Q3: Plot the training history and show this to your teacher (Question 3, 3 points) 
A3: see plots

 ### improve describe the changes in accuracy for validation and loss
Q4: Discuss with your group, then describe to your teacher, how the accuracy on the 
    training and validation sets progress differently across epochs, and what this tells us 
    about the generalisation of the model. (Question 4, 5 points).  
A4: After epoche 8 the trainings loss intersects with the validation loss, which already have reached the performens plateau  
    several epoches earlier. Trainings loss for classification is very good at
    epoche 12 (loss: 0.2644 - accuracy: 0.9265) with a neglectable 0.01009 difference to the validation (or test) set (val_loss: 0.2745 - val_accuracy: 0.9258),
    indicating good generalizability with respect to the maximum performance, with very minimal overfitting to the idosyncracies of trainings data. 

Q5: Evaluate the model performance on the test set using the following command: 
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0) 
    Show your teacher what values you get for the model’s accuracy and loss. (Question 
    5, 2 points) 
A5: Verbose = 0 yields lower run times for an epoche (e.g. 2-3s) with some what higher/lower accuracy/loss (might not be robust). 
    For instance, when 0: loss: 0.2653 - accuracy: 0.9252 - val_loss: 0.2732 - val_accuracy: 0.9286
                  when 1: loss: 0.3940 - accuracy: 0.8868 - val_loss: 0.3117 - val_accuracy: 0.9130
    
Q6: Discuss with your group, then describe to your teacher, whether this accuracy is 
    sufficient for some uses of automatic hand-written digit classification. (Question 6, 5 
    points) 
A6: Overall generalization is good (e.g. 92% accuracy), however good is subjective and depends on the context of the application of the model 
    
    EXAMPLE: A 92% accuracy rate might considered as (very) good but for instance (1) tumor detection sensitivity or (2)
    error detection in power plant having a missclassifcation (e.g. FP, FN) of 8% might be considered as not sufficient.

Q7: In the previous model, we did not specify an activation function for our hidden layer, 
    so it used the default linear activation. Discuss with your group, then describe to your 
    teacher, how linear activation of units limits the possible computations this model can 
    perform. (Question 7, 5 points) 
A7: The rectification operation usually contains a non-linear function, such as a signmoid, tanh or ReLu function, 
    otherwise the pattern analysis is limited to linear patterns, since the sum of linear functions is a linear function 
    with introducing a non linear function new patters can be analyse that go beyond linear features such as nonlinear relationships.
    In ReLu the activation threshold is often limited, for computational means and neuronal plausability.

Q8: Now make a similar model with a rectified activation in the first hidden layer.
    Plot the training history and show it to your teacher (Question 8, 2 points) 
A8: The epoche time is between 2-4s with initial performance of: 
    loss: 0.3181 - accuracy: 0.9110 - val_loss: 0.1662 - val_accuracy: 0.9545
    and final performance of:
    loss: 0.0128 - accuracy: 0.9965 - val_loss: 0.0842 - val_accuracy: 0.9779
    
Q9: Discuss with your group, then describe to your teacher, how this training history 
    differs from the previous model, for the training and validation sets. Describe what 
    this tells us about the generalisation of the model. 
    (Question 9, 5 points) 
A9: Both the test and the trainings set achieve higher accurac/lower loss on both training (0.9965) and test set (0.9779), 
    than the linear activation function. In addition traing period extends for some epoches reaching the plateau later, 
    while training and test performance already intersect at epoch 1, where, unlike the validation performance, the trainings
    performance increases thereafter, indicating some degree of overfitting with respect to the maximum performance    
"""

# load train and test dataset
def load_dataset():
	# load dataset
    data =  mnist.load_data()
    trainX = data[0][0]
    trainY = data[0][1]
    testX = data[1][0]
    testY = data[1][1]
    #np.array([(trainX, trainY)]), np.array([(testX, testY)]) = mnist.load_data()
	# reshape dataset to have a single channel
    trainX = np.ndarray.reshape(trainX, (60000, 28 * 28))
    testX = np.ndarray.reshape(testX, (10000, 28 * 28))
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
    #model.add(Dense(256, input_shape = (28 * 28, )))
    model.add(Dense(256, activation = 'relu', 
                         input_shape = (28 * 28, ))) #Q8
    model.add(Dense(10,  activation = 'softmax'))
    # compile model
    model.compile(optimizer = RMSprop(), 
                  loss = 'categorical_crossentropy', 
                  metrics = ['accuracy'])
    return model

def evaluate_model(trainX, trainY, testX, testY):
    # define model
    model = define_model()
    model.summary()
    # fit model
    history = model.fit(trainX, 
                        trainY, 
                        batch_size = 128, 
                        epochs = 12, 
                        verbose = 1, 
                        validation_split = 0.2)
    #score = model.evaluate(testX, testY, verbose = 0) 
    score = model.evaluate(testX, 
                           testY, 
                           verbose = 1) #Q5
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

def run_MLP():
	# load dataset
    trainX, trainY, testX, testY = load_dataset()
	# prepare pixel data
    trainX, testX = prep_pixels(trainX, testX)
	# evaluate model
    evaluate_model(trainX, trainY, testX, testY)
	 
# entry point, run the MLP
run_MLP()