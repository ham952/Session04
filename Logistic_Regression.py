import numpy as np
import cv2
from Utils import *
import matplotlib.pyplot as plt
import h5py
from lr_utils import load_dataset

#######################################################################################
################ Loading the data (cat/non-cat) #######################################
#######################################################################################
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# Example of a picture
index = 25
image = train_set_x_orig[index]
#plt.imshow(image)
#plt.show()
print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")

print "Train_set_Y Shape : ",train_set_y.shape             # (1L, 209L)
print "Train_set_X Shape : ",train_set_x_orig.shape        # (209L, 64L, 64L, 3L)
print "X Shape : ",train_set_x_orig[index].shape       # (64L, 64L, 3L)

'''
- m_train (number of training examples)
- m_test (number of test examples)
- num_px (= height = width of a training image)
'''

m_train = train_set_x_orig.shape[0]     # 209
m_test =  test_set_x_orig.shape[0]      # 50
num_px = train_set_x_orig.shape[1]      # 64

print "\nNumber of training examples: m_train = " + str(m_train)
print "Number of testing examples: m_test = " + str(m_test)
print "Height/Width of each image: num_px = " + str(num_px)
print "Each image is of size:" "(" + str(num_px) + ", " + str(num_px) + ", 3)"

#######################################################################################
##################### Data Pre-processing  ############################################
#######################################################################################

# Reshape the training and test examples into single vectors of shape (num_px x num_px x 3, 1)

# A trick when you want to flatten a matrix X of shape (a,b,c,d) to a matrix X_flatten of shape (bxcxd, a) is to use:
# X_flatten = X.reshape(X.shape[0], -1).T

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T

print "\ntrain_set_x_flatten shape: " + str(train_set_x_flatten.shape)
print "train_set_y shape: " + str(train_set_y.shape)
print "test_set_x_flatten shape: " + str(test_set_x_flatten.shape)
print "test_set_y shape: " + str(test_set_y.shape)

# Let's standardize our dataset
train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.
#train_set_x = normalizeRows(train_set_x_flatten)
#test_set_x = normalizeRows(test_set_x_flatten)

#ToDo : use of normalize row functions

#######################################################################################
#################### Model Training / Testing #########################################
#######################################################################################


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    """
    Builds the logistic regression model using "initialize_with_zeros","optimize" and "predict" functions from Utils

    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations

    Returns:
    d -- dictionary containing information about the model.
    """


    # initialize parameters with zeros
    m = X_train.shape[0]
    w, b = initialize_with_zeros(m)

    # Gradient descent
    parameters, grads, costs = optimize(w,b,X_train,Y_train,num_iterations,learning_rate,print_cost)

    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]

    # Predict test/train set examples
    Y_prediction_test = predict(w,b,X_test)
    Y_prediction_train = predict(w,b,X_train)


    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d

d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)

# Plot learning curve (with costs)
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()



learning_rates = [0.01, 0.001, 0.0001]
models = {}
for i in learning_rates:
    print ("learning rate is: " + str(i))
    models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 1500, learning_rate = i, print_cost = False)
    print ('\n' + "-------------------------------------------------------" + '\n')

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))

plt.ylabel('cost')
plt.xlabel('iterations (hundreds)')

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()






# We preprocess the image to fit your algorithm.
## START CODE HERE ## (PUT YOUR IMAGE NAME)
my_image = "my_image1.jpg"   # change this to the name of your image file
## END CODE HERE ##

# We preprocess the image to fit your algorithm.
#fname = "images/" + my_image
fname = my_image
#image = np.array(ndimage.imread(fname, flatten=False))
image = cv2.imread(fname)
image = image/255.
my_image = cv2.resize(image, dsize=(num_px,num_px), interpolation=cv2.INTER_CUBIC).reshape((1, num_px*num_px*3)).T
my_predicted_image,probability = predict_mod(d["w"], d["b"], my_image)

plt.imshow(image)
plt.show()
print"y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture with "+ str(np.squeeze(probability)) + " %" +"  Probability"