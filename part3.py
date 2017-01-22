import glob
import numpy as np
from matplotlib.pyplot import *
from scipy.misc import imread
from scipy.misc import imsave
from scipy.misc import imresize

############################################################################
#   Part 3
############################################################################


#######################load the data for Hader(actor)#######################
h_train = []
h_validation = []
h_test = []

hader_train = glob.glob('data/male/hader/train/*')
hader_validation = glob.glob('data/male/hader/validation/*')
hader_test = glob.glob('data/male/hader/test/*')

for i in range(0, 100):
    try:
        img = imread(hader_train[i])
        img = img.reshape(-1)
        h_train.append(img)
    except Exception:
        pass

for i in range(0, 10):
    try:
        img = imread(hader_validation[i])
        img = img.reshape(-1)
        h_validation.append(img)
        img = imread(hader_test[i])
        img = img.reshape(-1)
        h_test.append(img)
    except Exception:
        pass

hader_train = np.array(h_train)
hader_validation = np.array(h_validation)
hader_test = np.array(h_test)
hader_m, hader_n = hader_train.shape



#######################load the data for Carell(actor)#######################
c_train = []
c_validation = []
c_test = []

carell_train = glob.glob('data/male/carell/train/*')
carell_validation = glob.glob('data/male/carell/validation/*')
carell_test = glob.glob('data/male/carell/test/*')

for i in range(0, 100):
    try:
        img = imread(carell_train[i])
        img = img.reshape(-1)
        c_train.append(img)
    except Exception:
        pass

for i in range(0, 10):
    try:
        img = imread(carell_validation[i])
        img = img.reshape(-1)
        c_validation.append(img)
        img = imread(carell_test[i])
        img = img.reshape(-1)
        c_test.append(img)
    except Exception:
        pass

carell_train = np.asarray(c_train)
carell_validation = np.asarray(c_validation)
carell_test = np.asarray(c_test)
carell_m, carell_n = carell_train.shape



#######################done loading data#######################

#######apply linear classification algorithm###################

#Needed function for linear classifier
np.random.seed()

#output the prediction
#take in data and parameter
#output a vector of prediction for each data example
def computePrediction(X, theta):
    m, n = X.shape
    theta_feature, output_col = theta.shape
    assert (theta_feature == n), "theta feature size and X size does not align"
    
    prediction = X.dot(theta)
    return prediction
 
 
#compute the cost function
#cost function J = 1 / (2m) * sum(h_theta(x) - y) ** 2
def computeCost(X, theta, Y):
    m, n = X.shape
    h_theta_x = computePrediction(X, theta)
    cost = 1 / float(2 * m) * sum(np.square(h_theta_x - Y))
    return cost


#compute the gradient of the cost function w.r.t. each theta parameter
#apply analytical formula
#the output should be a vector of values
def computeGradient(X, theta, Y):
    m, n = X.shape
    h_theta_x = computePrediction(X, theta)
    temp = np.multiply(X, (h_theta_x - Y))
    gradient = sum(temp, 0) / float(m)
    return gradient.T
    

#gradient descent algorithm
#the output of the function should give a trained model: a vector of parameters
def gradientDescent(computePrediction, computeCost, computeGradient, X, theta, Y, max_iter):
    print "Starting Gradient Descent..."
    learning_rate = 0.0000001
    EPS = 1e-6
    
    prev_theta = theta - 2 * EPS
    gradient_val = computeGradient(X, theta, Y)
    theta = theta - learning_rate * np.asmatrix(gradient_val).T
    
    iter = 0
    while np.linalg.norm(prev_theta - theta, 2) > EPS and iter < max_iter:
        prev_theta = theta
        gradient_val = computeGradient(X, theta, Y)
        
        theta = theta - learning_rate * np.asmatrix(gradient_val)
        iter = iter + 1
    
    model = theta
    print "Number of iterations until convergence: " + str(iter)
    return model

#############Train the data##############
max_iter = 5000
#############set up the data#############
#add a bias term in the training data
dummy_ones = np.ones((hader_m + carell_m, 1))
train_data = np.vstack((hader_train, carell_train))
train_data = np.hstack((train_data, dummy_ones))

#set up the target value. assume hader is 1 and carell output 0
dummy_ones = np.ones((hader_m, 1))
dummy_zero = np.zeros((carell_m, 1))
target = np.vstack((dummy_ones, dummy_zero))

#set up the theta parameter
assert(hader_n == carell_n), "Two sets of data's features do not align"
theta = np.random.rand(hader_n + 1, 1) / 100000.0

#########apply the learning algorithm#########
model = gradientDescent(computePrediction, computeCost, computeGradient, train_data, theta, target, max_iter)
print "#############################################"
print "Training data cost", computeCost(train_data, theta, target)


##########test for validation and test set data############
print "#############################################"
dummy_ones = np.ones((20, 1))
validation_data = np.vstack((hader_validation, carell_validation))
test_data = np.vstack((hader_test, carell_test))
validation_data = np.hstack((validation_data, dummy_ones))
test_data = np.hstack((test_data, dummy_ones))
print "The result w.r.t. the validation set"
#print computePrediction(validation_data, model)

dummy_ones = np.ones((10, 1))
dummy_zero = np.zeros((10, 1))
vali_target = np.vstack((dummy_ones, dummy_zero))
print "Validation cost", computeCost(validation_data, theta, vali_target)

print "#############################################"
print "Test data Result:"
print "Hader Correct ", sum(computePrediction(test_data, model)[0:10] >=0.5)
print "Carell Correct", sum(computePrediction(test_data, model)[11:20] < 0.5)

theta = np.random.rand(1024, 1) / 100000.0
dummy_ones = np.ones((2, 1))
dummy_zero = np.zeros((2, 1))
target = np.vstack((dummy_ones, dummy_zero))
twoimage_data = hader_train[4:6][:]
twoimage_data = np.vstack((twoimage_data, carell_train[:2][:]))
twoimage_model = gradientDescent(computePrediction, computeCost, computeGradient, twoimage_data, theta, target, max_iter)
twoimage_model = np.reshape(twoimage_model, (32, 32))

print twoimage_data.shape

model = model[:][0:1024]
model = np.reshape(model, (32, 32))




fig = figure()
a = fig.add_subplot(1,2,1)
imgplot = imshow(model)
a.set_title("Trained full model")
a = fig.add_subplot(1,2,2)
imgplot = imshow(twoimage_model)
a.set_title("2 images from each person")
show()



