########this file includes codes from part 6 onwards################

import glob
from numpy import *
from scipy.misc import imread
from scipy.misc import imsave
from scipy.misc import imresize

from matplotlib.pyplot import *

####################load the data####################
######interested actors/actresses
#'Fran Drescher', 'America Ferrera', 'Kristin Chenoweth', 'Alec Baldwin', 'Bill Hader', 'Steve Carell'
######
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

drescher_train = array(d_train).T
drescher_validation = array(d_validation).T
drescher_test = array(d_test).T
drescher_n, drescher_m = asmatrix(drescher_train).shape

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

ferrera_train = array(f_train).T
ferrera_validation = array(f_validation).T
ferrera_test = array(f_test).T
ferrera_n, ferrera_m = asmatrix(ferrera_train).shape

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

chenoweth_train = array(c_train).T
chenoweth_validation = array(c_validation).T
chenoweth_test = array(c_test).T
chenoweth_n, chenoweth_m = asmatrix(chenoweth_train).shape


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

baldwin_train = array(b_train).T
baldwin_validation = array(b_validation).T
baldwin_test = array(b_test).T
baldwin_n, baldwin_m = asmatrix(baldwin_train).shape

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

hader_train = array(h_train).T
hader_validation = array(h_validation).T
hader_test = array(h_test).T
hader_n, hader_m = asmatrix(hader_train).shape

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

carell_train = array(c_train).T
carell_validation = array(c_validation).T
carell_test = array(c_test).T
carell_n, carell_m = asmatrix(carell_train).shape

#prediction
#input X: [n x m] model: [k x n]
#output for each case, predict the score for each class
def predict(X, model):
	temp = model.T.dot(X)
	return temp

#loss function
#input X: [n x m] Y: [k, m] Theta: [k x n]
#output one value, loss
def loss(X, Y, Theta):
	temp = square(Theta.T.dot(X) - Y)
	temp = asmatrix(temp)
	return sum(temp)

#compute gradient
#input X: [n x m] Y: [k, m] Theta: [k x n]
#output a matrix corresponding to the theta
def gradient(X, Y, Theta):
	temp = 2 * X.dot((Theta.T.dot(X) - Y).T)
	return temp

#gradient descent function
#apply gradient descent algorithm
def gradientDescent(X, Theta, Y):
	alpha = 1e-11
	threshold = 120
	max_itr = 5000

	i = 0
	while loss(X, Y, Theta) > threshold and i < max_itr:
		Theta = Theta - alpha * gradient(X, Y, Theta)
		i = i + 1

	model = Theta
	print "Learning complete at iteration: ", i
	return model

#given the input and the target, test the correct prediction rate
def result(X, model, Y):
	prediction = predict(X, model)
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

np.random.seed(0)
theta = np.random.rand(1025, 6) / 1e5

print "=================Start Learning==============="
model = gradientDescent(train_data, theta, train_target)
print "Training Loss: ", loss(train_data, train_target, model)

print "================Validation data==============="
print "Validation Loss: ", loss(validation_data, validation_target, model)

print "===============Test Performance==============="
print "Test data Loss: ", loss(test_data, test_target, model)
print "Test data correction: ", result(test_data, model, test_target)


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
######interested actors/actresses
#'Fran Drescher', 'America Ferrera', 'Kristin Chenoweth', 'Alec Baldwin', 'Bill Hader', 'Steve Carell'
######








