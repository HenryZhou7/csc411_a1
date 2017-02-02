import glob
import numpy as np
from numpy import *
from scipy.misc import imread
from scipy.misc import imsave
from scipy.misc import imresize

from random import shuffle
from random import randint
import random

#import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
###################################

from pylab import *

from matplotlib.pyplot import *
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imsave
from scipy.misc import imresize
import matplotlib.image as mpimg
import os
from scipy.ndimage import filters
import urllib

#############################################
print "===================================================================================="
print "=================================Project 1=========================================="
print "===================================================================================="
print "Start downloading data and crop the images"
print "Need to have files named subset_actors.txt and subset_actresses.txt"

male = "./cropped_male"
female = "./cropped_female"
if not os.path.exists(male):
    os.makedirs(male)

if not os.path.exists(female):
    os.makedirs(female)

np.random.seed()


act = list(set([a.split("\t")[0] for a in open("subset_actors.txt").readlines()]))

def rgb2gray(rgb):
    '''Return the grayscale version of the RGB image rgb as a 2D numpy array
    whose range is 0..1
    Arguments:
    rgb -- an RGB image, represented as a numpy array of size n x m x 3. The
    range of the values is 0..255
    '''
    
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray/255.


def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    '''From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/'''
    import threading
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return False
    else:
        return it.result

testfile = urllib.URLopener()            


#Note: you need to create the uncropped folder first in order 
#for this to work

for a in act:
    name = a.split()[1].lower()
    print a

    i = 0
    for line in open("subset_actors.txt"):
        if a in line:
            filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
            #get the crop info
            try:
                corp_info = line.split()[5]
                x1,y1,x2,y2 = map(int,(corp_info.split(",")))
            except Exception:
                pass
            #A version without timeout (uncomment in case you need to 
            #unsupress exceptions, which timeout() does)
            #testfile.retrieve(line.split()[4], "uncropped/"+filename)
            #timeout is used to stop downloading images which take too long to download
                
            timeout(testfile.retrieve, (line.split()[4], "cropped_male/"+filename), {}, 30)
            if not os.path.isfile("cropped_male/"+filename):
                continue
            
            try:
                        
                #open the file and crop it. resize and apply the grey scale to the picture
                img = imread("cropped_male/"+filename)
                img = img[y1:y2, x1:x2]
            
                img = rgb2gray(img)
                img = imresize(img, (32, 32))
                imsave("cropped_male/"+filename, img)
            except Exception:
                pass

            
            print filename
            i += 1
    
act = list(set([a.split("\t")[0] for a in open("subset_actresses.txt").readlines()])) 

testfile = urllib.URLopener()            


#Note: you need to create the uncropped folder first in order 
#for this to work

for a in act:
    name = a.split()[1].lower()
    print a

    i = 0
    for line in open("subset_actresses.txt"):
        if a in line:
            filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
            #get the crop info
            try:
                corp_info = line.split()[5]
                x1,y1,x2,y2 = map(int,(corp_info.split(",")))
            except Exception:
                pass
            #A version without timeout (uncomment in case you need to 
            #unsupress exceptions, which timeout() does)
            #testfile.retrieve(line.split()[4], "uncropped/"+filename)
            #timeout is used to stop downloading images which take too long to download
                
            timeout(testfile.retrieve, (line.split()[4], "cropped_female/"+filename), {}, 30)
            if not os.path.isfile("cropped_female/"+filename):
                continue
            
            try:
                        
                #open the file and crop it. resize and apply the grey scale to the picture
                img = imread("cropped_female/"+filename)
                img = img[y1:y2, x1:x2]
            
                img = rgb2gray(img)
                img = imresize(img, (32, 32))
                imsave("cropped_female/"+filename, img)
            except Exception:
                pass

            
            print filename
            i += 1


print "Downloading files finish"
print "Start loading all the data"
#############################################
b_train = []
b_validation = []
b_test = []

butler_data = glob.glob('cropped_male/butler*')
data_size = len(butler_data)
shuffle(butler_data)

index = 0

i = 0
while i < 10:
    try:
        img = imread(butler_data[index])
        img = img.reshape(-1)
        if img.shape != (1024,):
            index = index + 1
            continue
        index = index + 1
        i = i + 1
        b_validation.append(img)
    except Exception:
        index = index + 1
        pass

i = 0
while i < 10:
    try:
        img = imread(butler_data[index])
        img = img.reshape(-1)
        if img.shape != (1024,):
            index = index + 1
            continue
        index = index + 1
        i = i + 1
        b_test.append(img)
    except Exception:
        index = index + 1
        pass


i = 0
while i < 100:
    try:
        img = imread(butler_data[index])
        img = img.reshape(-1)
        if img.shape != (1024,):
            index = index + 1
            continue
        index = index + 1
        i = i + 1
        b_train.append(img)
    except Exception:
        index = index + 1
        pass



butler_train = array(b_train)
butler_validation = array(b_validation)
butler_test = array(b_test)
butler_m, butler_n = asmatrix(butler_train).shape



#############################################
h_train = []
h_validation = []
h_test = []

hader_data = glob.glob('cropped_male/hader*')
data_size = len(hader_data)
shuffle(hader_data)

index = 0

i = 0
while i < 10:
    try:
        img = imread(hader_data[index])
        img = img.reshape(-1)
        if img.shape != (1024,):
            index = index + 1
            continue
        index = index + 1
        i = i + 1
        h_validation.append(img)
    except Exception:
        index = index + 1
        pass

i = 0
while i < 10:
    try:
        img = imread(hader_data[index])
        img = img.reshape(-1)
        if img.shape != (1024,):
            index = index + 1
            continue
        index = index + 1
        i = i + 1
        h_test.append(img)
    except Exception:
        index = index + 1
        pass


i = 0
while i < 100:
    try:
        img = imread(hader_data[index])
        img = img.reshape(-1)
        if img.shape != (1024,):
            index = index + 1
            continue
        index = index + 1
        i = i + 1
        h_train.append(img)
    except Exception:
        index = index + 1
        pass



hader_train = array(h_train)
hader_validation = array(h_validation)
hader_test = array(h_test)
hader_m, hader_n = asmatrix(hader_train).shape



#############################################
c_train = []
c_validation = []
c_test = []

carell_data = glob.glob('cropped_male/carell*')
data_size = len(carell_data)
shuffle(carell_data)

index = 0

i = 0
while i < 10:
    try:
        img = imread(carell_data[index])
        img = img.reshape(-1)
        if img.shape != (1024,):
            index = index + 1
            continue
        index = index + 1
        i = i + 1
        c_validation.append(img)
    except Exception:
        index = index + 1
        pass

i = 0
while i < 10:
    try:
        img = imread(carell_data[index])
        img = img.reshape(-1)
        if img.shape != (1024,):
            index = index + 1
            continue
        index = index + 1
        i = i + 1
        c_test.append(img)
    except Exception:
        index = index + 1
        pass


i = 0
while i < 100:
    try:
        img = imread(carell_data[index])
        img = img.reshape(-1)
        if img.shape != (1024,):
            index = index + 1
            continue
        index = index + 1
        i = i + 1
        c_train.append(img)
    except Exception:
        index = index + 1
        pass



carell_train = array(c_train)
carell_validation = array(c_validation)
carell_test = array(c_test)
carell_m, carell_n = asmatrix(carell_train).shape


#############################################
v_train = []
v_validation = []
v_test = []

vartan_data = glob.glob('cropped_male/vartan*')
data_size = len(vartan_data)
shuffle(vartan_data)

index = 0

i = 0
while i < 10:
    try:
        img = imread(vartan_data[index])
        img = img.reshape(-1)
        if img.shape != (1024,):
            index = index + 1
            continue
        index = index + 1
        i = i + 1
        v_validation.append(img)
    except Exception:
        index = index + 1
        pass

i = 0
while i < 10:
    try:
        img = imread(vartan_data[index])
        img = img.reshape(-1)
        if img.shape != (1024,):
            index = index + 1
            continue
        index = index + 1
        i = i + 1
        v_test.append(img)
    except Exception:
        index = index + 1
        pass


i = 0
while i < 100:
    try:
        img = imread(vartan_data[index])
        img = img.reshape(-1)
        if img.shape != (1024,):
            index = index + 1
            continue
        index = index + 1
        i = i + 1
        v_train.append(img)
    except Exception:
        index = index + 1
        pass



vartan_train = array(v_train)
vartan_validation = array(v_validation)
vartan_test = array(v_test)
vartan_m, vartan_n = asmatrix(vartan_train).shape



#############################################
b_train = []
b_validation = []
b_test = []

baldwin_data = glob.glob('cropped_male/baldwin*')
data_size = len(baldwin_data)
shuffle(baldwin_data)

index = 0

i = 0
while i < 10:
    try:
        img = imread(baldwin_data[index])
        img = img.reshape(-1)
        if img.shape != (1024,):
            index = index + 1
            continue
        index = index + 1
        i = i + 1
        b_validation.append(img)
    except Exception:
        index = index + 1
        pass

i = 0
while i < 10:
    try:
        img = imread(baldwin_data[index])
        img = img.reshape(-1)
        if img.shape != (1024,):
            index = index + 1
            continue
        index = index + 1
        i = i + 1
        b_test.append(img)
    except Exception:
        index = index + 1
        pass


i = 0
while i < 100:
    try:
        img = imread(baldwin_data[index])
        img = img.reshape(-1)
        if img.shape != (1024,):
            index = index + 1
            continue
        index = index + 1
        i = i + 1
        b_train.append(img)
    except Exception:
        index = index + 1
        pass



baldwin_train = array(b_train)
baldwin_validation = array(b_validation)
baldwin_test = array(b_test)
baldwin_m, baldwin_n = asmatrix(baldwin_train).shape



#############################################
r_train = []
r_validation = []
r_test = []

radcliffe_data = glob.glob('cropped_male/radcliffe*')
data_size = len(radcliffe_data)
shuffle(radcliffe_data)

index = 0

i = 0
while i < 10:
    try:
        img = imread(radcliffe_data[index])
        img = img.reshape(-1)
        if img.shape != (1024,):
            index = index + 1
            continue
        index = index + 1
        i = i + 1
        r_validation.append(img)
    except Exception:
        index = index + 1
        pass

i = 0
while i < 10:
    try:
        img = imread(radcliffe_data[index])
        img = img.reshape(-1)
        if img.shape != (1024,):
            index = index + 1
            continue
        index = index + 1
        i = i + 1
        r_test.append(img)
    except Exception:
        index = index + 1
        pass


i = 0
while i < 100:
    try:
        img = imread(radcliffe_data[index])
        img = img.reshape(-1)
        if img.shape != (1024,):
            index = index + 1
            continue
        index = index + 1
        i = i + 1
        r_train.append(img)
    except Exception:
        index = index + 1
        pass



radcliffe_train = array(r_train)
radcliffe_validation = array(r_validation)
radcliffe_test = array(r_test)
radcliffe_m, radcliffe_n = asmatrix(radcliffe_train).shape

print "Actors all loaded"
#############################################
c_train = []
c_validation = []
c_test = []

chenoweth_data = glob.glob('cropped_female/chenoweth*')
data_size = len(chenoweth_data)
shuffle(chenoweth_data)

index = 0

i = 0
while i < 10:
    try:
        img = imread(chenoweth_data[index])
        img = img.reshape(-1)
        if img.shape != (1024,):
            index = index + 1
            continue
        index = index + 1
        i = i + 1
        c_validation.append(img)
    except Exception:
        index = index + 1
        pass

i = 0
while i < 10:
    try:
        img = imread(chenoweth_data[index])
        img = img.reshape(-1)
        if img.shape != (1024,):
            index = index + 1
            continue
        index = index + 1
        i = i + 1
        c_test.append(img)
    except Exception:
        index = index + 1
        pass


i = 0
while i < 100 and index < data_size:
    try:
        img = imread(chenoweth_data[index])
        img = img.reshape(-1)
        if img.shape != (1024,):
            index = index + 1
            continue
        index = index + 1
        i = i + 1
        c_train.append(img)
    except Exception:
        index = index + 1
        pass



chenoweth_train = array(c_train)
chenoweth_validation = array(c_validation)
chenoweth_test = array(c_test)
chenoweth_m, chenoweth_n = asmatrix(chenoweth_train).shape

#############################################
f_train = []
f_validation = []
f_test = []

ferrera_data = glob.glob('cropped_female/ferrera*')
data_size = len(ferrera_data)
shuffle(ferrera_data)

index = 0

i = 0
while i < 10:
    try:
        img = imread(ferrera_data[index])
        img = img.reshape(-1)
        if img.shape != (1024,):
            index = index + 1
            continue
        index = index + 1
        i = i + 1
        f_validation.append(img)
    except Exception:
        index = index + 1
        pass

i = 0
while i < 10:
    try:
        img = imread(ferrera_data[index])
        img = img.reshape(-1)
        if img.shape != (1024,):
            index = index + 1
            continue
        index = index + 1
        i = i + 1
        f_test.append(img)
    except Exception:
        index = index + 1
        pass


i = 0
while i < 100 and index < data_size:
    try:
        img = imread(ferrera_data[index])
        img = img.reshape(-1)
        if img.shape != (1024,):
            index = index + 1
            continue
        index = index + 1
        i = i + 1
        f_train.append(img)
    except Exception:
        index = index + 1
        pass



ferrera_train = array(f_train)
ferrera_validation = array(f_validation)
ferrera_test = array(f_test)
ferrera_m, ferrera_n = asmatrix(ferrera_train).shape

#############################################
d_train = []
d_validation = []
d_test = []

drescher_data = glob.glob('cropped_female/drescher*')
data_size = len(drescher_data)
shuffle(drescher_data)

index = 0

i = 0
while i < 10:
    try:
        img = imread(drescher_data[index])
        img = img.reshape(-1)
        if img.shape != (1024,):
            index = index + 1
            continue
        index = index + 1
        i = i + 1
        d_validation.append(img)
    except Exception:
        index = index + 1
        pass

i = 0
while i < 10:
    try:
        img = imread(drescher_data[index])
        img = img.reshape(-1)
        if img.shape != (1024,):
            index = index + 1
            continue
        index = index + 1
        i = i + 1
        d_test.append(img)
    except Exception:
        index = index + 1
        pass


i = 0
while i < 100 and index < data_size:
    try:
        img = imread(drescher_data[index])
        img = img.reshape(-1)
        if img.shape != (1024,):
            index = index + 1
            continue
        index = index + 1
        i = i + 1
        d_train.append(img)
    except Exception:
        index = index + 1
        pass



drescher_train = array(d_train)
drescher_validation = array(d_validation)
drescher_test = array(d_test)
drescher_m, drescher_n = asmatrix(drescher_train).shape


#############################################
b_train = []
b_validation = []
b_test = []

bracco_data = glob.glob('cropped_female/bracco*')
data_size = len(bracco_data)
shuffle(bracco_data)

index = 0

i = 0
while i < 10:
    try:
        img = imread(bracco_data[index])
        img = img.reshape(-1)
        if img.shape != (1024,):
            index = index + 1
            continue
        index = index + 1
        i = i + 1
        b_validation.append(img)
    except Exception:
        index = index + 1
        pass

i = 0
while i < 10:
    try:
        img = imread(bracco_data[index])
        img = img.reshape(-1)
        if img.shape != (1024,):
            index = index + 1
            continue
        index = index + 1
        i = i + 1
        b_test.append(img)
    except Exception:
        index = index + 1
        pass


i = 0
while i < 100 and index < data_size:
    try:
        img = imread(bracco_data[index])
        img = img.reshape(-1)
        if img.shape != (1024,):
            index = index + 1
            continue
        index = index + 1
        i = i + 1
        b_train.append(img)
    except Exception:
        index = index + 1
        pass



bracco_train = array(b_train)
bracco_validation = array(b_validation)
bracco_test = array(b_test)
bracco_m, bracco_n = asmatrix(bracco_train).shape

#############################################
g_train = []
g_validation = []
g_test = []

gilpin_data = glob.glob('cropped_female/gilpin*')
data_size = len(gilpin_data)
shuffle(gilpin_data)

index = 0

i = 0
while i < 10:
    try:
        img = imread(gilpin_data[index])
        img = img.reshape(-1)
        if img.shape != (1024,):
            index = index + 1
            continue
        index = index + 1
        i = i + 1
        g_validation.append(img)
    except Exception:
        index = index + 1
        pass

i = 0
while i < 10:
    try:
        img = imread(gilpin_data[index])
        img = img.reshape(-1)
        if img.shape != (1024,):
            index = index + 1
            continue
        index = index + 1
        i = i + 1
        g_test.append(img)
    except Exception:
        index = index + 1
        pass


i = 0
while i < 100 and index < data_size:
    try:
        img = imread(gilpin_data[index])
        img = img.reshape(-1)
        if img.shape != (1024,):
            index = index + 1
            continue
        index = index + 1
        i = i + 1
        g_train.append(img)
    except Exception:
        index = index + 1
        pass



gilpin_train = array(g_train)
gilpin_validation = array(g_validation)
gilpin_test = array(g_test)
gilpin_m, gilpin_n = asmatrix(gilpin_train).shape


#############################################
h_train = []
h_validation = []
h_test = []

harmon_data = glob.glob('cropped_female/harmon*')
data_size = len(harmon_data)
shuffle(harmon_data)

index = 0

i = 0
while i < 10:
    try:
        img = imread(harmon_data[index])
        img = img.reshape(-1)
        if img.shape != (1024,):
            index = index + 1
            continue
        index = index + 1
        i = i + 1
        h_validation.append(img)
    except Exception:
        index = index + 1
        pass

i = 0
while i < 10:
    try:
        img = imread(harmon_data[index])
        img = img.reshape(-1)
        if img.shape != (1024,):
            index = index + 1
            continue
        index = index + 1
        i = i + 1
        h_test.append(img)
    except Exception:
        index = index + 1
        pass


i = 0
while i < 100 and index < data_size:
    try:
        img = imread(harmon_data[index])
        img = img.reshape(-1)
        if img.shape != (1024,):
            index = index + 1
            continue
        index = index + 1
        i = i + 1
        h_train.append(img)
    except Exception:
        index = index + 1
        pass



harmon_train = array(h_train)
harmon_validation = array(h_validation)
harmon_test = array(h_test)
harmon_m, harmon_n = asmatrix(harmon_train).shape



print "Download data and load data finished"
#Needed function for linear classifier
print "=============Start Part 3============="

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
    learning_rate = 1e-7
    EPS = 1e-7
    
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


#deciding the output based on the prediction of the model
def output_on_prediction(prediction):
    row, col = prediction
    for i in range(0, row):
        if prediction[row] >= 0.5:
            return "Hader"
        else:
            return "Carell"


#############Train the data##############
max_iter = 5000
#############set up the data#############
#add a bias term in the training data
dummy_ones = ones((hader_m + carell_m, 1))
train_data = vstack((hader_train, carell_train))
train_data = hstack((train_data, dummy_ones))

#set up the target value. assume hader is 1 and carell output 0
dummy_ones = ones((hader_m, 1))
dummy_zero = zeros((carell_m, 1))
target = vstack((dummy_ones, dummy_zero))

#set up the theta parameter
assert(hader_n == carell_n), "Two sets of data's features do not align"
theta = np.random.rand(hader_n + 1, 1) / 1e5

#########apply the learning algorithm#########
model = gradientDescent(computePrediction, computeCost, computeGradient, train_data, theta, target, max_iter)
print "#############################################"
print "Training data cost", computeCost(train_data, theta, target)


##########test for validation and test set data############
print "#############################################"
dummy_ones = ones((20, 1))
validation_data = vstack((hader_validation, carell_validation))
test_data = vstack((hader_test, carell_test))
validation_data = hstack((validation_data, dummy_ones))
test_data = hstack((test_data, dummy_ones))
print "The result w.r.t. the validation set"
#print computePrediction(validation_data, model)

dummy_ones = ones((10, 1))
dummy_zero = zeros((10, 1))
vali_target = vstack((dummy_ones, dummy_zero))
print "Validation cost", computeCost(validation_data, theta, vali_target)

print "#############################################"
print "Test data Result:"
hader_correct = sum(computePrediction(test_data, model)[0:10] >=0.5)
carell_correct = sum(computePrediction(test_data, model)[11:20] < 0.5)
print "Hader Correct ", hader_correct
print "Carell Correct", carell_correct
print "Correctness Rate: ", (hader_correct + carell_correct) / 20.

##########################################################################################
########################################### Part 4 #######################################
##########################################################################################

print "#############start of part 4#############"
theta = np.random.rand(1024, 1) / 100000.0
dummy_ones = ones((2, 1))
dummy_zero = zeros((2, 1))
target = vstack((dummy_ones, dummy_zero))
twoimage_data = hader_train[4:6][:]
twoimage_data = vstack((twoimage_data, carell_train[9:11][:]))
twoimage_model = gradientDescent(computePrediction, computeCost, computeGradient, twoimage_data, theta, target, max_iter)
twoimage_model = reshape(twoimage_model, (32, 32))



model = model[:][0:1024]
model = reshape(model, (32, 32))




fig = figure()
a = fig.add_subplot(1,2,1)
filename = "full_data_trained.jpg"
imsave(filename, model)
imgplot = imshow(model, cmap=get_cmap('gray'))
a.set_title("Trained full model")
a = fig.add_subplot(1,2,2)
filename = "two_image_per_person_model.jpg"
imsave(filename, twoimage_model)
imgplot = imshow(twoimage_model, cmap=get_cmap('gray'))
a.set_title("2 images from each person")
plt.show()

print "#############start of part 5#############"
print "male/female classification"



#################needed functions for linear classifier#################

#######apply linear classification algorithm###################

#Needed function for linear classifier

#output the prediction
#take in data and parameter
#output a vector of prediction for each data example
def computePrediction_p5(X, theta):
    m, n = X.shape
    theta_feature, output_col = theta.shape
    assert (theta_feature == n), "theta feature size and X size does not align"
    
    prediction = X.dot(theta)
    return prediction
 
 
#compute the cost function
#cost function J = 1 / (2m) * sum(h_theta(x) - y) ** 2
def computeCost_p5(X, theta, Y):
    m, n = X.shape
    h_theta_x = computePrediction_p5(X, theta)
    cost = 1 / float(2 * m) * sum(square(h_theta_x - Y))
    return cost


#compute the gradient of the cost function w.r.t. each theta parameter
#apply analytical formula
#the output should be a vector of values
def computeGradient_p5(X, theta, Y):
    m, n = X.shape
    h_theta_x = computePrediction_p5(X, theta)
    temp = multiply(X, (h_theta_x - Y))
    gradient = sum(temp, 0) / float(m)
    return gradient.T
    

#gradient descent algorithm
#the output of the function should give a trained model: a vector of parameters
def gradientDescent_p5(computePrediction_p5, computeCost_p5, computeGradient_p5, X, theta, Y, max_iter):
    print "Starting Gradient Descent..."
    learning_rate = 1e-7
    EPS = 1e-7
    
    prev_theta = theta - 10 * EPS
    gradient_val = computeGradient_p5(X, theta, Y)
    theta = theta - learning_rate * asmatrix(gradient_val).T
    
    iter = 0
    while linalg.norm(prev_theta - theta, 2) > EPS and iter < max_iter:
        prev_theta = theta
        gradient_val = computeGradient_p5(X, theta, Y)
        
        theta = theta - learning_rate * asmatrix(gradient_val)
        iter = iter + 1
    
    model = theta
    print "Number of iterations until convergence: " + str(iter)
    return model


###################data preprocessing for training the model###################
size = range(0, 10)
for i in range(0, 10):
    size[i] = 10 * size[i] + 1

#construct the validation set and test set
dummy_ones = ones((60, 1))
validation_data = vstack((drescher_validation, ferrera_validation))
validation_data = vstack((validation_data, chenoweth_validation))
validation_data = vstack((validation_data, baldwin_validation))
validation_data = vstack((validation_data, hader_validation))
validation_data = vstack((validation_data, carell_validation))
validation_data = hstack((validation_data, dummy_ones))

dummy_ones = ones((bracco_m + gilpin_m + harmon_m + radcliffe_m + butler_m + vartan_m , 1))
test_data = vstack((bracco_train, gilpin_train))
test_data = vstack((test_data, harmon_train))
test_data = vstack((test_data, radcliffe_train))
test_data = vstack((test_data, butler_train))
test_data = vstack((test_data, vartan_train))
test_data = hstack((test_data, dummy_ones))

dummy_ones = ones((bracco_m + gilpin_m + harmon_m, 1))
dummy_zero = zeros((radcliffe_m + butler_m + vartan_m, 1))
test_target = vstack((dummy_ones, dummy_zero))


dummy_ones = ones((30, 1))
dummy_zero = zeros((30, 1))
vali_target = vstack((dummy_ones, dummy_zero))
#test_target = vstack((dummy_ones, dummy_zero))

training_cost = []
validation_cost = []
test_cost = []
test_performance = []

#set up the training data with different sizes



for i in range(0, 10):
    
    dummy_ones = ones((6 * size[i], 1))
    train_data = vstack((drescher_train[:size[i]][:], ferrera_train[:size[i]][:]))
    train_data = vstack((train_data, chenoweth_train[:size[i]][:]))
    train_data = vstack((train_data, baldwin_train[:size[i]][:]))
    train_data = vstack((train_data, hader_train[:size[i]][:]))
    train_data = vstack((train_data, carell_train[:size[i]][:]))
    train_data = hstack((train_data, dummy_ones))

    dummy_ones = ones((3 * size[i], 1))
    dummy_zero = zeros((3 * size[i], 1))
    target = vstack((dummy_ones, dummy_zero))

    max_iter = 3000
    theta = np.random.rand(1025, 1) / 1e5

    model = gradientDescent_p5(computePrediction_p5, computeCost_p5, computeGradient_p5, train_data, theta, target, max_iter)
    t_cost = computeCost_p5(train_data, theta, target)
    v_cost = computeCost_p5(validation_data, model, vali_target)
    te_cost = computeCost_p5(test_data, model, test_target)
    training_cost.append(t_cost)
    validation_cost.append(v_cost)
    test_cost.append(te_cost)

    female_correct = sum(computePrediction_p5(test_data, model)[0:100 + 97 + 100] >=0.5)
    male_correct   =sum(computePrediction_p5(test_data, model)[100 + 97 + 100 + 1:] < 0.5)

    test_performance.append((female_correct + male_correct) / 597.)
    print "==========Training for", train_data.shape, "train set is finished.=========="



size = asarray(size)
training_cost = asarray(training_cost)
validation_cost = asarray(validation_cost)
test_cost = asarray(test_cost)
test_performance = asarray(test_performance)


fig = figure()

axes = fig.gca()
axes.set_xlim([0, 100])
axes.set_ylim([0, 0.2])

plot(size, training_cost, label = "Train")
plot(size, validation_cost, label = "Validation")
#plot(size, test_cost, label = "Test")
xlabel("Images per person in training set")
ylabel("Loss on Train/Validation Data")
legend(loc = 'center right', framealpha = 0.5)
title("Loss on Training and Validation Set")
show()

plot(size, test_performance, label = "Performance on testing set")
xlabel("Images per person in training set")
ylabel("Performance")
legend(loc = 'lower right', framealpha = 0.5)
title("Performance on Different Test Set")
show()


print "===============Start Part 6 7 and 8==============="

######data preprocessing

drescher_train = drescher_train.T
ferrera_train = ferrera_train.T
chenoweth_train = chenoweth_train.T
baldwin_train = baldwin_train.T
hader_train = hader_train.T
carell_train = carell_train.T


drescher_validation = drescher_validation.T
ferrera_validation = ferrera_validation.T
chenoweth_validation = chenoweth_validation.T
baldwin_validation = baldwin_validation.T
hader_validation = hader_validation.T
carell_validation = carell_validation.T


drescher_test = drescher_test.T
ferrera_test = ferrera_test.T
chenoweth_test = chenoweth_test.T
baldwin_test = baldwin_test.T
hader_test = hader_test.T
carell_test = carell_test.T

#Copy the related part 6 functions

#prediction
#input X: [n x m] model: [k x n]
#output for each case, predict the score for each class
def predict_p6(X, model):
    temp = model.T.dot(X)
    return temp

#loss function
#input X: [n x m] Y: [k, m] Theta: [k x n]
#output one value, loss
def loss_p6(X, Y, Theta):
    temp = square((Theta.T).dot(X) - Y)
    temp = asmatrix(temp)
    return sum(temp)

#compute gradient
#input X: [n x m] Y: [k, m] Theta: [k x n]
#output a matrix corresponding to the theta
def gradient_p6(X, Y, Theta):
    temp = 2 * X.dot((Theta.T.dot(X) - Y).T)
    return temp

#gradient using small differences
#input X, Y, Theta
#compute the gradient using small differences with respect to index p, q
#output is the same size as Theta
def gradient_small_diff_p6(X, Y, Theta, p, q):
    h = 1e-15

    Theta_prime = Theta.copy()
    value = Theta[p, q] + h
    Theta_prime[p, q] = value
      
    gradient_s = (loss_p6(X, Y, Theta_prime) - loss_p6(X, Y, Theta)) / float(h)
    
    return gradient_s

#gradient descent function
#apply gradient descent algorithm
def gradientDescent_p6(X, Theta, Y):
    alpha = 1e-11
    threshold = 120
    max_itr = 5000

    i = 0
    while loss_p6(X, Y, Theta) > threshold and i < max_itr:
        #verify small difference method yields the same result

        Theta = Theta - alpha * gradient_p6(X, Y, Theta)
        i = i + 1

    model = Theta
    print "Learning complete at iteration: ", i
    return model

#given the input and the target, test the correct prediction rate
def result_p6(X, model, Y):
    prediction = predict_p6(X, model)
    decision = argmax(prediction, 0)
    decision = decision + 1
    correct = 0
    correct = sum(decision[0:10] == 1) + sum(decision[11:20] == 2) + sum(decision[21:30] == 3)
    correct = correct + sum(decision[31:40] == 4) + sum(decision[41:50] == 5) + sum(decision[51:60] == 6)
    return correct / 60.


######interested actors/actresses
######################################################################################################
#'Fran Drescher', 'America Ferrera', 'Kristin Chenoweth', 'Alec Baldwin', 'Bill Hader', 'Steve Carell'
######################################################################################################

dummy_ones = ones((1, 600)) #bias term
train_data = hstack((drescher_train, ferrera_train, chenoweth_train, baldwin_train, hader_train, carell_train))
train_data = vstack((dummy_ones, train_data))

dummy_ones = ones((1, 60)) #bias term
validation_data = hstack((drescher_validation, ferrera_validation, chenoweth_validation, baldwin_validation, hader_validation, carell_validation))
validation_data = vstack((dummy_ones, validation_data))

dummy_ones = ones((1, 60)) #bias term
test_data = hstack((drescher_test, ferrera_test, chenoweth_test, baldwin_test, hader_test, carell_test))
test_data = vstack((dummy_ones, test_data))


#build the label for the training set
dummy_ones = ones((1, 100))
dummy_zero = zeros((1, 100))
a1 = vstack((dummy_ones, dummy_zero, dummy_zero, dummy_zero, dummy_zero, dummy_zero))
a2 = vstack((dummy_zero, dummy_ones, dummy_zero, dummy_zero, dummy_zero, dummy_zero))
a3 = vstack((dummy_zero, dummy_zero, dummy_ones, dummy_zero, dummy_zero, dummy_zero))
a4 = vstack((dummy_zero, dummy_zero, dummy_zero, dummy_ones, dummy_zero, dummy_zero))
a5 = vstack((dummy_zero, dummy_zero, dummy_zero, dummy_zero, dummy_ones, dummy_zero))
a6 = vstack((dummy_zero, dummy_zero, dummy_zero, dummy_zero, dummy_zero, dummy_ones))

train_target = hstack((a1, a2, a3, a4, a5, a6))

dummy_ones = ones((1, 10))
dummy_zero = zeros((1, 10))
a1 = vstack((dummy_ones, dummy_zero, dummy_zero, dummy_zero, dummy_zero, dummy_zero))
a2 = vstack((dummy_zero, dummy_ones, dummy_zero, dummy_zero, dummy_zero, dummy_zero))
a3 = vstack((dummy_zero, dummy_zero, dummy_ones, dummy_zero, dummy_zero, dummy_zero))
a4 = vstack((dummy_zero, dummy_zero, dummy_zero, dummy_ones, dummy_zero, dummy_zero))
a5 = vstack((dummy_zero, dummy_zero, dummy_zero, dummy_zero, dummy_ones, dummy_zero))
a6 = vstack((dummy_zero, dummy_zero, dummy_zero, dummy_zero, dummy_zero, dummy_ones))
validation_target = hstack((a1, a2, a3, a4, a5, a6))
test_target = hstack((a1, a2, a3, a4, a5, a6))


theta = np.random.rand(1025, 6) / 1e5


print "=================Small Diff Gradient================="
init_param = np.random.rand(1025, 6) / 1e6

param = gradient_p6(train_data, train_target, init_param)


for i in range(0, 4):
    print "Example No.", i + 1
    p = randint(0, 1025)
    q = randint(0, 5)
    small_val = gradient_small_diff_p6(train_data, train_target, init_param, p, q)
    print "Small_diff val :", small_val
    print "Gradient val   :", param[p, q]
    print "Difference rate:", (small_val - param[p, q]) / param[p, q]





print "=================Start Learning==============="
model = gradientDescent_p6(train_data, theta, train_target)
print "Training Loss: ", loss_p6(train_data, train_target, model)

print "================Validation data==============="
print "Validation Loss: ", loss_p6(validation_data, validation_target, model)

print "===============Test Performance==============="
print "Test data Loss: ", loss_p6(test_data, test_target, model)
print "Test data correction: ", result_p6(test_data, model, test_target)


model = asmatrix(model)
p1 = model[1: , 0]
p1 = reshape(p1, (32, 32))
p2 = model[1: , 1]
p2 = reshape(p2, (32, 32))
p3 = model[1: , 2]
p3 = reshape(p3, (32, 32))
p4 = model[1: , 3]
p4 = reshape(p4, (32, 32))
p5 = model[1: , 4]
p5 = reshape(p5, (32, 32))
p6 = model[1: , 5]
p6 = reshape(p6, (32, 32))

fig = figure()
a = fig.add_subplot(2,3,1)
imgplot = imshow(p1, cmap=get_cmap('gray'))
title("Drescher")
a = fig.add_subplot(2,3,2)
imgplot = imshow(p2, cmap=get_cmap('gray'))
title("Ferrera")
a = fig.add_subplot(2,3,3)
imgplot = imshow(p3, cmap=get_cmap('gray'))
title("Chenoweth")
a = fig.add_subplot(2,3,4)
imgplot = imshow(p4, cmap=get_cmap('gray'))
title("Baldwin")
a = fig.add_subplot(2,3,5)
imgplot = imshow(p5, cmap=get_cmap('gray'))
title("Hader")
a = fig.add_subplot(2,3,6)
imgplot = imshow(p6, cmap=get_cmap('gray'))
title("Carell")
show()



print "=====================end of the project====================="

























