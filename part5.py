import glob
from numpy import *
from scipy.misc import imread
from scipy.misc import imsave
from scipy.misc import imresize


####################part 5####################

####################load the data####################

####################load drescher####################
d_train = []
d_validation = []
d_test = []

drescher_train = glob.glob('data/female/drescher/train/*')
drescher_validation = glob.glob('data/female/drescher/validation/*')
drescher_test = glob.glob('data/female/drescher/test/*')

for i in range(0, 100):
    try:
        img = imread(drescher_train[i])
        img = img.reshape(-1)
        d_train.append(img)
    except Exception:
        pass

for i in range(0, 10):
    try:
        img = imread(drescher_validation[i])
        img = img.reshape(-1)
        d_validation.append(img)
        img = imread(drescher_test[i])
        img = img.reshape(-1)
        d_test.append(img)
    except Exception:
        pass

drescher_train = array(d_train)
drescher_validation = array(d_validation)
drescher_test = array(d_test)
drescher_m, drescher_n = asmatrix(drescher_train).shape

####################load ferrera####################
f_train = []
f_validation = []
f_test = []

ferrera_train = glob.glob('data/female/ferrera/train/*')
ferrera_validation = glob.glob('data/female/ferrera/validation/*')
ferrera_test = glob.glob('data/female/ferrera/test/*')

for i in range(0, 100):
    try:
        img = imread(ferrera_train[i])
        img = img.reshape(-1)
        f_train.append(img)
    except Exception:
        pass

for i in range(0, 10):
    try:
        img = imread(ferrera_validation[i])
        img = img.reshape(-1)
        f_validation.append(img)
        img = imread(ferrera_test[i])
        img = img.reshape(-1)
        f_test.append(img)
    except Exception:
        pass

ferrera_train = array(f_train)
ferrera_validation = array(f_validation)
ferrera_test = array(f_test)
ferrera_m, ferrera_n = asmatrix(ferrera_train).shape

####################load chenoweth####################
c_train = []
c_validation = []
c_test = []

chenoweth_train = glob.glob('data/female/chenoweth/train/*')
chenoweth_validation = glob.glob('data/female/chenoweth/validation/*')
chenoweth_test = glob.glob('data/female/chenoweth/test/*')

for i in range(0, 100):
    try:
        img = imread(chenoweth_train[i])
        img = img.reshape(-1)
        c_train.append(img)
    except Exception:
        pass

for i in range(0, 10):
    try:
        img = imread(chenoweth_validation[i])
        img = img.reshape(-1)
        c_validation.append(img)
        img = imread(chenoweth_test[i])
        img = img.reshape(-1)
        c_test.append(img)
    except Exception:
        pass

chenoweth_train = array(c_train)
chenoweth_validation = array(c_validation)
chenoweth_test = array(c_test)
chenoweth_m, chenoweth_n = asmatrix(chenoweth_train).shape


####################load male####################
####################load baldwin####################

b_train = []
b_validation = []
b_test = []

baldwin_train = glob.glob('data/male/baldwin/train/*')
baldwin_validation = glob.glob('data/male/baldwin/validation/*')
baldwin_test = glob.glob('data/male/baldwin/test/*')

for i in range(0, 100):
    try:
        img = imread(baldwin_train[i])
        img = img.reshape(-1)
        b_train.append(img)
    except Exception:
        pass

for i in range(0, 10):
    try:
        img = imread(baldwin_validation[i])
        img = img.reshape(-1)
        b_validation.append(img)
        img = imread(baldwin_test[i])
        img = img.reshape(-1)
        b_test.append(img)
    except Exception:
        pass

baldwin_train = array(b_train)
baldwin_validation = array(b_validation)
baldwin_test = array(b_test)
baldwin_m, baldwin_n = asmatrix(baldwin_train).shape

####################load hader####################

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

hader_train = array(h_train)
hader_validation = array(h_validation)
hader_test = array(h_test)
hader_m, hader_n = asmatrix(hader_train).shape

####################load carell####################

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

carell_train = array(c_train)
carell_validation = array(c_validation)
carell_test = array(c_test)
carell_m, carell_n = asmatrix(carell_train).shape


#################needed functions for linear classifier#################

#######apply linear classification algorithm###################

#Needed function for linear classifier
random.seed(0)

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
    cost = 1 / float(2 * m) * sum(square(h_theta_x - Y))
    return cost


#compute the gradient of the cost function w.r.t. each theta parameter
#apply analytical formula
#the output should be a vector of values
def computeGradient(X, theta, Y):
    m, n = X.shape
    h_theta_x = computePrediction(X, theta)
    temp = multiply(X, (h_theta_x - Y))
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
    theta = theta - learning_rate * asmatrix(gradient_val).T
    
    iter = 0
    while linalg.norm(prev_theta - theta, 2) > EPS and iter < max_iter:
        prev_theta = theta
        gradient_val = computeGradient(X, theta, Y)
        
        theta = theta - learning_rate * asmatrix(gradient_val)
        iter = iter + 1
    
    model = theta
    print "Number of iterations until convergence: " + str(iter)
    return model

###################load the validation and test data###################

####################load bracco####################

b_train = []
b_validation = []
b_test = []

bracco_train = glob.glob('data/female/bracco/train/*')
bracco_validation = glob.glob('data/female/bracco/validation/*')
bracco_test = glob.glob('data/female/bracco/test/*')

for i in range(0, 100):
    try:
        img = imread(bracco_train[i])
        img = img.reshape(-1)
        b_train.append(img)
    except Exception:
        pass

for i in range(0, 10):
    try:
        img = imread(bracco_validation[i])
        img = img.reshape(-1)
        b_validation.append(img)
        img = imread(bracco_test[i])
        img = img.reshape(-1)
        b_test.append(img)
    except Exception:
        pass

bracco_train = array(b_train)
bracco_validation = array(b_validation)
bracco_test = array(b_test)
bracco_m, bracco_n = asmatrix(bracco_train).shape

####################load gilpin####################

g_train = []
g_validation = []
g_test = []

gilpin_train = glob.glob('data/female/gilpin/train/*')
gilpin_validation = glob.glob('data/female/gilpin/validation/*')
gilpin_test = glob.glob('data/female/gilpin/test/*')

for i in range(0, 100):
    try:
        img = imread(gilpin_train[i])
        img = img.reshape(-1)
        g_train.append(img)
    except Exception:
        pass

for i in range(0, 10):
    try:
        img = imread(gilpin_validation[i])
        img = img.reshape(-1)
        g_validation.append(img)
        img = imread(gilpin_test[i])
        img = img.reshape(-1)
        g_test.append(img)
    except Exception:
        pass

gilpin_train = array(g_train)
gilpin_validation = array(g_validation)
gilpin_test = array(g_test)
gilpin_m, gilpin_n = asmatrix(gilpin_train).shape

####################load harmon####################

h_train = []
h_validation = []
h_test = []

harmon_train = glob.glob('data/female/harmon/train/*')
harmon_validation = glob.glob('data/female/harmon/validation/*')
harmon_test = glob.glob('data/female/harmon/test/*')

for i in range(0, 100):
    try:
        img = imread(harmon_train[i])
        img = img.reshape(-1)
        h_train.append(img)
    except Exception:
        pass

for i in range(0, 10):
    try:
        img = imread(harmon_validation[i])
        img = img.reshape(-1)
        h_validation.append(img)
        img = imread(harmon_test[i])
        img = img.reshape(-1)
        h_test.append(img)
    except Exception:
        pass

harmon_train = array(h_train)
harmon_validation = array(h_validation)
harmon_test = array(h_test)
harmon_m, harmon_n = asmatrix(harmon_train).shape

####################load radcliffe####################

r_train = []
r_validation = []
r_test = []

radcliffe_train = glob.glob('data/male/radcliffe/train/*')
radcliffe_validation = glob.glob('data/male/radcliffe/validation/*')
radcliffe_test = glob.glob('data/male/radcliffe/test/*')

for i in range(0, 100):
    try:
        img = imread(radcliffe_train[i])
        img = img.reshape(-1)
        r_train.append(img)
    except Exception:
        pass

for i in range(0, 10):
    try:
        img = imread(radcliffe_validation[i])
        img = img.reshape(-1)
        r_validation.append(img)
        img = imread(radcliffe_test[i])
        img = img.reshape(-1)
        r_test.append(img)
    except Exception:
        pass

radcliffe_train = array(r_train)
radcliffe_validation = array(r_validation)
radcliffe_test = array(r_test)
radcliffe_m, radcliffe_n = asmatrix(radcliffe_train).shape

####################load butler####################

b_train = []
b_validation = []
b_test = []

butler_train = glob.glob('data/male/butler/train/*')
butler_validation = glob.glob('data/male/butler/validation/*')
butler_test = glob.glob('data/male/butler/test/*')

for i in range(0, 100):
    try:
        img = imread(butler_train[i])
        img = img.reshape(-1)
        b_train.append(img)
    except Exception:
        pass

for i in range(0, 10):
    try:
        img = imread(butler_validation[i])
        img = img.reshape(-1)
        b_validation.append(img)
        img = imread(butler_test[i])
        img = img.reshape(-1)
        b_test.append(img)
    except Exception:
        pass

butler_train = array(b_train)
butler_validation = array(b_validation)
butler_test = array(b_test)
butler_m, butler_n = asmatrix(butler_train).shape

####################load vartan####################

v_train = []
v_validation = []
v_test = []

vartan_train = glob.glob('data/male/vartan/train/*')
vartan_validation = glob.glob('data/male/vartan/validation/*')
vartan_test = glob.glob('data/male/vartan/test/*')

for i in range(0, 100):
    try:
        img = imread(vartan_train[i])
        img = img.reshape(-1)
        v_train.append(img)
    except Exception:
        pass

for i in range(0, 10):
    try:
        img = imread(vartan_validation[i])
        img = img.reshape(-1)
        v_validation.append(img)
        img = imread(vartan_test[i])
        img = img.reshape(-1)
        v_test.append(img)
    except Exception:
        pass

vartan_train = array(v_train)
vartan_validation = array(v_validation)
vartan_test = array(v_test)
vartan_m, vartan_n = asmatrix(vartan_train).shape


###################data preprocessing for training the model###################
examplesperperson = range(0, 48)
for i in range(0, 48):
	examplesperperson[i] = 2 * examplesperperson[i] + 1

#construct the validation set and test set
dummy_ones = ones((60, 1))
validation_data = vstack((bracco_validation, gilpin_validation))
validation_data = vstack((validation_data, harmon_validation))
validation_data = vstack((validation_data, radcliffe_validation))
validation_data = vstack((validation_data, butler_validation))
validation_data = vstack((validation_data, vartan_validation))
validation_data = hstack((validation_data, dummy_ones))

test_data = vstack((bracco_test, gilpin_test))
test_data = vstack((test_data, harmon_test))
test_data = vstack((test_data, radcliffe_test))
test_data = vstack((test_data, butler_test))
test_data = vstack((test_data, vartan_test))
test_data = hstack((test_data, dummy_ones))

dummy_ones = ones((30, 1))
dummy_zero = zeros((30, 1))
vali_target = vstack((dummy_ones, dummy_zero))
test_target = vstack((dummy_ones, dummy_zero))

#set up the training data. 
size = 100
dummy_ones = ones((6 * size, 1))
train_data = vstack((drescher_train[:size + 1][:], ferrera_train)[:size + 1][:])
train_data = vstack((train_data, chenoweth_train[:size + 1][:]))
train_data = vstack((train_data, baldwin_train[:size + 1][:]))
train_data = vstack((train_data, hader_train[:size + 1][:]))
train_data = vstack((train_data, carell_train[:size + 1][:]))
train_data = hstack((train_data, dummy_ones))

#set female label to be 1 and male label to be 0
dummy_ones = ones((300, 1))
dummy_zero = zeros((300, 1))
target = vstack((dummy_ones, dummy_zero))

######################apply training algorithm######################
max_iter = 2000
theta = random.rand(1025, 1) / 100000.0

print "##################training starts###########################"
model = gradientDescent(computePrediction, computeCost, computeGradient, train_data, theta, target, max_iter)
print "Training data cost", computeCost(train_data, theta, target)

print "##################validati result###########################"
print "Validati data cost", computeCost(validation_data, model, vali_target)
female_correct = sum(computePrediction(validation_data, model)[0:30] >=0.5)
male_correct   =sum(computePrediction(validation_data, model)[31:60] < 0.5)
print "Female prediction correct", female_correct
print "Male prediction correct  ", male_correct
print "Total performance (correct prediction rate): ", (female_correct + male_correct) / 60.

print "##################testing result ###########################"
print "Test data cost", computeCost(test_data, model, vali_target)
female_correct = sum(computePrediction(test_data, model)[0:30] >=0.5)
male_correct   =sum(computePrediction(test_data, model)[31:60] < 0.5)
print "Female prediction correct", female_correct
print "Male prediction correct  ", male_correct
print "Total performance (correct prediction rate): ", (female_correct + male_correct) / 60.






