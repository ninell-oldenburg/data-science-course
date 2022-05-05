#!/usr/bin/env python
# coding: utf-8

# # Exercise 2 (Using Linear Regression). 
# Your task is to predict the quality score of red wine from its physicochemical properties using linear regression. You are going to learn a linear model t = f(x, w) = w0 + w1x1 + w2x2 + . . . + wDxD = wT x, where t is the predicted output variable (quality score), x = (1, x1, x2, . . . , xD) T is the vector-valued
# input variable (physicochemical properties), and w = (w0, w1, . . . , wD) are the free parameters. The
# parameters wi define the regression model, and once they have been estimated, the model can be used
# to predict outputs for new input values x0.
# a) Implement linear regression, call your function multivarlinreg(), a template is provided in
# the multivarlinreg.py file. Your code should load the data matrix X containing the input
# variables, as well as the output vector t, and output an estimate of the free parameters in the
# model, that is the wi
# in the form of the vector w. Remember the offset parameter w0. You
# should not use a built-in function for regression.
# b) Run your regression function on a version of the training set that only contains the first feature
# (fixed acidity). Please report your weights. What can you learn from them?
# c) Run your regression function on all the features in the training set and report your estimated
# parameters wi
# . What do they tell you about how the different physiochemical properties relate
# to the wine quality?

# In[1]:


# import packages, data, & set figure sizes
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10,6)

dataWineTrain = np.loadtxt('redwine_training.txt')
dataWineTest = np.loadtxt('redwine_testing.txt')

XWineTrain = dataWineTrain[:,:-1]
YWineTrain =dataWineTrain[:,-1]

XWineTest = dataWineTest[:,:-1]
YWineTest =dataWineTest[:,-1]

minstX = np.loadtxt('MNIST_179_digits.txt')
minstY = np.loadtxt('MNIST_179_labels.txt')

minstXTrain = minstX[:900]
minstYTrain = minstY[:900]

minstXTest = minstX[900:]
minstYTest = minstY[900:]


# a) Implement linear regression, call your function multivarlinreg(), a template is provided in
# the multivarlinreg.py file. Your code should load the data matrix X containing the input
# variables, as well as the output vector t, and output an estimate of the free parameters in the
# model, that is the wi
# in the form of the vector w. Remember the offset parameter w0. You
# should not use a built-in function for regression.

# In[2]:


# input: 1) X: the independent variables (data matrix), an (N x D)-dimensional matrix, as a numpy array
#        2) y: the dependent variable, an N-dimensional vector, as a numpy array
#
# output: 1) the regression coefficients, a (D+1)-dimensional vector, as a numpy array
#
# note: remember to either expect an initial column of 1's in the input X, or to append this within your code
def multivarlinreg(X, y):
    X = np.array(X)
    w = np.array((X.shape[0]))
        
    # add preceding 1 column if not given
    if not np.all(X[0,:] == 1):
        preced = np.ones((X.shape[0], 1))
        X = np.hstack((preced, X))
      
    # compute analytical solution for multivariat lin reg
    w = np.dot((np.linalg.inv(np.dot(X.T,X))),(np.dot(X.T,y)))
    return w


# b) Run your regression function on a version of the training set that only contains the first feature (fixed acidity). Please report your weights. What can you learn from them?

# In[3]:


# run on only first column (fixed acidity)
XWineTrainReduced = XWineTrain[:,:1]
print(f'Parameters w_i: {multivarlinreg(XWineTrainReduced, YWineTrain)}')

# TODO: what can you learn from them?


# c) Run your regression function on all the features in the training set and report your estimated
# parameters wi
# . What do they tell you about how the different physiochemical properties relate
# to the wine quality?

# In[4]:


# run on all the data set
print(f'Parameters w_i: {multivarlinreg(XWineTrain, YWineTrain)}')

# TODO: What do they tell you about the different properties & how they relate to the wine quality?


# # Exercise 3 (Evaluating Linear Regression)
# a) Implement the root means square error as function rmse() for the linear model. A template is provided in the rmse.py file . For a set of known input-output values (x1, y1), . . . ,(xN , yN ), where yi is the recorded output value associated to the i th data point xi in the dataset, the parameter w allows to compute f(xi, w), the output value associated to the input xi as predicted by your regression model. Your code should take as input the predicted values f(xi, w) and ground truth output values yi. You should not use a built-in function for RMSE.

# In[5]:


# input: 1) f: the predicted values of the dependent output variable, an N-dimensional vector, as a numpy array
#        2) t: the ground truth values of dependent output variable, an N-dimensional vector, as a numpy array
#
# output: 1) the root mean square error (rmse) as a 1 x 1 numpy array

def rmse(f, t):
    error_sum = 0
    for i in range(len(f)):
        error_sum += (t[i] - f[i])**2
    return np.sqrt(error_sum/len(f))


# In[6]:


def predict(test, weight):
    return weight[0] + np.dot(weight[1:], test.T)


# b) Build the regression model for 1-dimensional input variables using the training set as in 1b).
# Use the first feature for the test set to compute the RMSE of your model.

# In[7]:


# build regression model
weight_first_col = multivarlinreg(XWineTrainReduced, YWineTrain)
# get first feature of test set
XWineTestReduced = XWineTest[:,:1]
predict_first_col = predict(XWineTestReduced, weight_first_col)
# compute RMSE
print(f'RMSE of first parameter "fixed acidity": {round(rmse(predict_first_col, YWineTest), 2)}')


# c) Build the regression model for the full 11-dimensional input variables using the training set as in
# 1c). Use the full-dimensional test set to compute the RMSE of this model. How does it compare
# to your answer from b)?

# In[8]:


# build regression model
weights_full = multivarlinreg(XWineTrain, YWineTrain)
predict_full = predict(XWineTest, weights_full)
# compute RMSE
print(f'RMSE of all parameters: {round(rmse(predict_full, YWineTest), 2)}')


# # Exercise 5 (Applying random forrest ). 
# Train a random forest with 50 trees on IDSWeedCropTrain.csv from the previous assignment(s). Test the resulting classifier using the data in IDSWeedCropTest.csv.

# In[9]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# read in the data
IDSTrain = np.loadtxt('IDSWeedCropTrain.csv', delimiter=',')
IDSTest = np.loadtxt('IDSWeedCropTest.csv', delimiter=',')

# split input variable & labels
XIDSTrain = IDSTrain[:,:-1]
YIDSTrain = IDSTrain[:,-1]
XIDSTest = IDSTest[:,:-1]
YIDSTest = IDSTest[:,-1]

clf = RandomForestClassifier(n_estimators=50, random_state=0)


# In[10]:


# fit model, make prediction, print accuracy
clf.fit(XIDSTrain, YIDSTrain)
predictions = clf.predict(XIDSTest)
print(f'Accuracy of sklearn.ensemble.RandomForestClassifier on IDS Weed Crop data set: {round(accuracy_score(YIDSTest, predictions), 3)}')


# # Exercise 6 (Gradient descent & learning rates).
# Apply gradient descent to find the minimum of the
# function f(x) = e^(−x/2) + 10x^2
# 
# Detailed instructions:
# 1. Compute the derivative of the function f.
# 2. Apply gradient descent with learning rates η = 0.1, 0.01, 0.001, 0.0001.3
# 3. For each of the learning rates do the following:
#     
# (a) Take x = 1 as a starting point.
# 
# (b) Visualize the tangent lines and gradient descent steps for the first three iterations (produce
# 4 plots for your report corresponding to gradient descent with the four learning rates). The
# first tangent line should be at the initial point x = 1.
# 
# (c) Visualize gradient descent steps for the first 10 iterations (another 4 plots; no visualization
# of tangent lines in this case).
# 
# (d) Run the algorithm until the magnitude of the gradient falls below 10−10 or the algorithm
# has exceeded 10,000 iterations. Report the number of iterations it took the algorithm to
# converge and the function value at the final iteration (4 × 2 values corresponding to the
# four learning rates).

# In[11]:


def graddesc(rate, f, init=1, max_iters=10000):
    curr_min = init
    old_min = 0
    iters = 0
    precision = 0.000001
    x_coor = np.array(())
        
    while abs(curr_min - old_min) > precision and iters < max_iters:
        
        x_coor = np.append(x_coor, curr_min)
            
        # TODO: what does the gradient thing do?
        # TODO: what does the learning rate do?
            
        old_min = curr_min
        curr_min = old_min - (rate * f(old_min))
        iters += 1
    
    return x_coor, iters


# In[12]:


def make_tangent(x_rang, x_val, func, deriv):
    y = np.array(len(x_rang))
    a = deriv(x_val)
    x = x_range-x_val
    b = func(x_val)
    y = a * x + b
    return y


# In[13]:


x_range = np.arange(-1.5, 1.5, 0.1)

fig1, axes1 = plt.subplots(nrows=2, ncols=2)
fig1.set_size_inches(10, 10)

fig2, axes2 = plt.subplots(nrows=2, ncols=2)
fig2.set_size_inches(10, 10)

learn_rates = [0.1, 0.01, 0.001, 0.0001]
f = lambda x: np.exp(-x/2) + 10*x**2
g = lambda x: 20*x - np.exp(-x/2)/2

for i, rate in enumerate(learn_rates):
    ax1 = axes1[i//2, i%2]
    ax2 = axes2[i//2, i%2]
    
    # plot original function f
    ax1.plot(x_range, f(x_range), label='f(x)')
    ax2.plot(x_range, f(x_range), label='f(x)')
    x_coor, num_iters = graddesc(rate, g)
    print(f'Function value for learning rate {rate} = ({round(f(x_coor[-1]), 10)}) after {num_iters} iterations')
    
    # fill gradient descent steps for both plots
    ax1.scatter(x_coor[:3], f(x_coor[:3]), c='r', label='Gradient steps')
    ax2.scatter(x_coor[:10], f(x_coor[:10]), c='r', label='Gradient steps')
    
    # plot tangent lines to first plot
    for i in range(3):
        ax1.plot(x_range, make_tangent(x_range, x_coor[i], f, g), c='g', label=f'Tangent line {i+1}')
    
    # set y lims & grid
    ax1.grid()
    ax1.legend()
    ax1.set_ylim(-1,20)
    ax2.grid()
    ax2.legend()
    ax2.set_ylim(-1,20)
    
    # set axes labels & titel
    fig1.suptitle('Visualization of first three gradient descent steps\n+ respective tangent lines for 4 different learning rates')
    ax1.set_xlabel('x') 
    ax1.set_ylabel('y')
    ax1.set_title(f'Learning rate {rate}')
    
    fig2.suptitle('Visualization of first 10 gradient descent steps for 4 different learning rates')
    ax2.set_xlabel('x') 
    ax2.set_ylabel('y')
    ax2.set_title(f'Learning rate {rate}')
    
plt.show()


# # Logistic Regression

# # Exercise 7 - Logistic regression implementation
# In this exercise you will implement, run and test logistic regression on two simple data sets, made of
# two classes each. They come from Fisher’s Iris dataset and are described below.

# 1. Make a scatter plot of each dataset, with point color depending on classes. What do you observe?

# In[14]:


# import cmap
# TODO INDEX DESCRIPTION VALUES
blues = plt.get_cmap('Blues')

fig, ax = plt.subplots(nrows=1, ncols=2)
fig.set_size_inches(20, 10)

# import data
iris2D2Train = np.loadtxt('Iris2D2_train.txt')
iris2D2Test = np.loadtxt('Iris2D2_test.txt')

iris2D1Train = np.loadtxt('Iris2D1_train.txt')
iris2D1Test = np.loadtxt('Iris2D1_test.txt')

cumulative2D2 = np.vstack((iris2D2Train, iris2D2Test))
cumulative2D1 = np.vstack((iris2D1Train, iris2D1Test))

Ycumulative2D2 = cumulative2D2[:,-1]
Ycumulative2D1 = cumulative2D1[:,-1]

idx0 = np.where(Ycumulative2D2==0)
idx1 = np.where(Ycumulative2D2==1)

idx2 = np.where(Ycumulative2D1==0)
idx3 = np.where(Ycumulative2D1==1)

ax[0].scatter(cumulative2D1[idx2,0], cumulative2D1[idx2,1], color=blues(0.3), label='Iris virginica')
ax[0].scatter(cumulative2D1[idx3,0], cumulative2D1[idx3,1], color=blues(0.8), label='Iris setosa')
ax[0].set_title('Iris2D1');
ax[0].set_xlabel('length of sepals') 
ax[0].set_ylabel('width of sepals')
ax[0].grid(True)
ax[0].legend()

ax[1].scatter(cumulative2D2[idx0,0], cumulative2D2[idx0,1], color=blues(0.3), label='Iris virginica')
ax[1].scatter(cumulative2D2[idx1,0], cumulative2D2[idx1,1], color=blues(0.8), label='Iris versicolor')
ax[1].set_title('Iris2D2');
ax[1].set_xlabel('length of sepals') 
ax[1].set_ylabel('length of petals')
ax[1].grid(True)
ax[1].legend()

fig.suptitle('Data points of cumulative train & test data sets for Iris data');
plt.show()


# 2. Implement logistic regression as presented in the lectures. Your function should take as input
# the training set data matrix, the training set label vector, and the test set data matrix.

# ### $$ \nabla_w E_{in} = -\frac{1}{N}\sum_{n=1}^N \frac{y_n e^{-y_n w^\top x_n}}{1 + e^{-y_n w^\top x_n}}x_n $$

# In[15]:


def log1px(x):
    return (x >= 0)*x + (np.log(1 + np.exp(-np.abs(x))))

def logistic(x):
    xp = (x >  0) * 1.0
    xm = (x <= 0) * 1.0
    return xm * (np.exp(x * xm))/(1 + np.exp(x * xm)) + xp * (1/(1 + np.exp(-xp * x)))


# In[16]:


def logistic_insample(X, y, w):
    N = X.shape[0]
    return np.log(1 + np.exp(-y * (X @ w))).sum()/N 


# In[17]:


def logistic_gradient(X, y, w):
    N, _ = X.shape
    
    curr = y * np.exp(-y*(np.dot(X, w))) / (1 + (np.exp(-y*(np.dot(X, w)))))
    curr = curr.reshape(N, 1)
    
    return -np.mean(curr*X, axis=0)


# In[18]:


def log_reg(Xorig, y, max_iter, tol=1e-5, step=1e-2):   
    
    # X is a d by N data matrix of input values
    N, d = Xorig.shape
    X = np.hstack((np.ones((N, 1)), Xorig))
        
    # y is a N by 1 matrix of target values -1 and 1
    y = np.array((y-.5) * 2)
        
    # Initialize weights at time step 0    
    w = 0.1 * np.random.randn(d + 1)
    
    # Compute value of logistic log likelihood
    value = logistic_insample(X, y, w)
    
    # Keep track of function values
    E_in = [value]
    G_norm= []
    converged = False
    for num_iter in range(max_iter):
        # Compute gradient at current w, and take 
        # its opposite as descent direction
        p = logistic_gradient(X, y, w)
                     
        # Update weights
        w_new = w-step*p
       
        # Determine whether we have converged: Is gradient norm below
        # tolerance?
               
        g_norm = np.linalg.norm(p)
        G_norm.append(g_norm)
        if g_norm < tol:
            # converged!
            converged = True
            break
            
        w = w_new
        value = logistic_insample(X, y, w)
        E_in.append(value)
        
    if not converged:
        # We ran all the iterations, not reching the tolerance
        print(f"The descent procedure did not converge after {max_iter} iterations, last gradient norm was {g_norm}.")
    else:
        # we did actually converge!
        print(f'The descent converged after {num_iter} iterations with a gradient magnitude {g_norm}.')
    return w, E_in, G_norm


# In[19]:


def log_pred(Xorig, w):
    N, d = Xorig.shape
    # add a first column with ones
    X = np.hstack((np.ones((N, 1)), Xorig))
    p = logistic(X @ w)
    pred = -(p < 0.5).astype(int) + (p >= 0.5).astype(int)
    return p, pred


# In[20]:


def get_acc(x_train, x_test, y_train, y_test, name):
    N_train = x_train.shape[0]
    N_test = x_test.shape[0]
    
    w, E, G = log_reg(x_train, y_train, 100000, tol=0.5e-5)
    
    p_val, train_pred_cl = log_pred(x_train, w)
    y_train = (y_train-.5) * 2
    train_err_sum = (np.abs(train_pred_cl-y_train)/2).sum()
    train_err_rate = train_err_sum/N_train
    
    p_val, test_pred_cl = log_pred(x_test, w)
    y_test = (y_test-.5) * 2
    test_err_sum = (np.abs(test_pred_cl-y_test)/2).sum()
    test_err_rate = test_err_sum/N_test
    
    print(f'Logistic regression for {name}:\nTrain error rate: {round(train_err_rate, 3)}\nTest error rate: {round(test_err_rate, 3)}\nParameters: {w}\n')
    


# In[21]:


iris2D2Train_X = iris2D2Train[:,:-1]
iris2D2Train_Y = iris2D2Train[:,-1]
iris2D2Test_X = iris2D2Test[:,:-1]
iris2D2Test_Y = iris2D2Test[:,-1]

iris2D1Train_X = iris2D1Train[:,:-1]
iris2D1Train_Y = iris2D1Train[:,-1]
iris2D1Test_X = iris2D1Test[:,:-1]
iris2D1Test_Y = iris2D1Test[:,-1]


# In[22]:


get_acc(iris2D2Train_X, iris2D2Test_X, iris2D2Train_Y, iris2D2Test_Y, 'Iris 2D2')
get_acc(iris2D1Train_X, iris2D1Test_X, iris2D1Train_Y, iris2D1Test_Y, 'Iris 2D1')


# # Exercise 9 - Clustering and classification I
# In this exercise, you will use some of the algorithms you
# used in previous assignments for clustering and classifying the dataset.
# 
# 
# a) Run the k-Means algorithm with k = 3 to cluster the dataset. Are the results meaningful? As
# the k-means algorithm will return labels 0, 1 and 2, you cannot compare them directly with the
# 1, 7 and 9 from the label file. Count instead the proportion of 1s, 7s and 9s in each cluster. For
# that you may want to prepend the label data to the image data, but not use it in the k-means
# algorithm, e.g., by fitting all but the first column.
# 

# In[23]:


from sklearn.cluster import KMeans
from collections import Counter

kmeans = KMeans (n_clusters=3, random_state=42, algorithm='full', n_init=1)
kmeans.fit(minstXTrain)
real_proportion = Counter(minstYTrain)
true_labels = np.vstack((minstYTrain, kmeans.labels_))
predict_proportion = Counter(zip(true_labels[0], true_labels[1]))
cluster_proportion = Counter(kmeans.labels_)
centers = kmeans.cluster_centers_
print(f'predicted proportion: {cluster_proportion}\nreal proportion: {real_proportion}\nproportion per true label: {predict_proportion}')

fig, ax = plt.subplots(nrows=1, ncols=3)
for i in range(3):
    this_digit = centers[i].reshape(28, 28)
    ax[i].imshow(this_digit,cmap='Greys_r',interpolation='none')
    ax[i].set_title(f"Cluster center {i+1}")
    ax[i].set_xlabel('num of pixels')
    ax[i].set_ylabel('num of pixels')
fig.suptitle('Cluster Center for K-Means clustering of MNIST_179_digits data set');
plt.show()


# b) Classification. Train a k-NN classifier on the dataset, using n−fold validation to obtain the
# optimal k. Report test accuracy. Note that if you use KNeighborsClassifier, you can pass it
# the true labels as labels.

# In[24]:


from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier

def get_kbest_and_acc(ks, x_train, x_test, y_train, y_test, split=5):
    # create indices for CV
    cv = KFold(n_splits=split)
    # loop over CV folds
    acc_score = np.zeros((len(ks)))
    for i, k in enumerate(ks):
        # set up classifier with given k
        neigh = KNeighborsClassifier(n_neighbors=k)
                
        # split data and evaluate accuracy
        for train, test in cv.split(x_train):
            XTrainCV, XTestCV, YTrainCV, YTestCV = x_train[train], x_train[test], y_train[train], y_train[test]
            neigh.fit(XTrainCV, YTrainCV)
            acc_score[i] += 1 - (accuracy_score(YTestCV, neigh.predict(XTestCV)))
        acc_score[i] = acc_score[i] / split
    
    # find minimum acc & get value for kbest respectively
    idx_min = np.argmin(acc_score)
    kbest = ks[idx_min]
    
    # create new KNN classifier with freshly found kbest value
    neigh_test = KNeighborsClassifier(n_neighbors=kbest)
    neigh_test.fit(x_train, y_train)
    accTest = accuracy_score(y_test, neigh_test.predict(x_test))
    
    return kbest, accTest

k_list = [1, 3, 5, 7, 9, 11]
kb, acc = get_kbest_and_acc(k_list, minstXTrain, minstXTest, minstYTrain, minstYTest)
print(f'Accuracy for MNIST_179_digits data set: {round(acc,3)} with k_best={kb}')


# # Exercise 10 - Clustering and classification after dimensionality reduction.
# In the second part, you will use dimensionality reduction (i.e. PCA here) to first reduce the dimensionality, and then perform clustering and classification.

# a) Compute the PCA of the training data. Plot the cumulative variance (in %) w.r.t. the principal
# components.

# In[25]:


# function from assignment 3
def pca(data):
    # subtract off the mean for each dimension
    data_mean = data - np.mean(data, 0)
    
    # transpose data & calculate covariance matrix
    data_covar = np.cov(data_mean.T)
    
    # eigenvalues and eigenvectors
    data_eigenval, data_eigenvec = np.linalg.eigh(data_covar)
    
    # reverse with same indexing
    idx = data_eigenval.argsort()[::-1]   
    data_eigenval = data_eigenval[idx]
    data_eigenvec = data_eigenvec[:,idx]
        
    return data_eigenval, data_eigenvec, np.mean(data, axis=0)


# In[26]:


# multidimensional scaling for input data & d number of dimensions
# taken from assignment 3
def mds(data, d):
    # get PCA components
    variance, components, mean = pca(data)
    # slice number of PCs
    components = components[:,:d]
    # compute dot product to get projection
    matrix = np.dot(data, components)
    # compute cumulative variance
    cum_var = round(variance[:d].sum() * 100 / variance.sum(), 2)
    
    return matrix, cum_var


# b) Run the k-Means algorithm, again with k = 3 to cluster the data projected on the first principal
# components. Experiment with 20 and 200 components. Count the proportion of 1s, 7s and
# 9s in each cluster. Again, you should prepend the labels to the projected data on the principal
# components as first column, but take care of not using it in k-means. Compare with the non-PCA
# clustering result in the previous exercise.

# In[27]:


# perform pca & calculate squareroots
eigval, eigvec, m = pca(minstXTrain)
sigma = np.sqrt(eigval[i])

def print_pca_cluster(num_comp):
    mds_minst, var = mds(minstXTrain, num_comp)
    kmeans_2 = KMeans(n_clusters=3, random_state=42, algorithm='full', n_init=1)
    kmeans_2.fit(mds_minst)
    true_labels = np.vstack((minstYTrain, kmeans_2.labels_))
    predict_proportion = Counter(zip(true_labels[0], true_labels[1]))
    centers = kmeans_2.cluster_centers_
    center_projection = np.dot(centers, eigvec[:,:num_comp].T)
    #center_projection += m
    print(f'predicted proportion: {predict_proportion}')

    fig, ax = plt.subplots(nrows=1, ncols=3)
    
    for i in range(3):
        this_digit = center_projection[i].reshape(28,28)
        ax[i].imshow(this_digit,cmap='Greys_r',interpolation='none')
        ax[i].set_title(f"Cluster center {i+1}", size=14)
        ax[i].set_xlabel('num of pixels')
        ax[i].set_ylabel('num of pixels')
    
    fig.suptitle(f'Cluster Center for K-Means clustering of first {num_comp} PCs\non MNIST_179_digits data set', size=16);
    plt.show()
    
    print(f'Cumulative Variance of first {num_comp} PCs: {var}%')


# In[28]:


x = np.arange(0, len(eigval), 1)
c_val = np.cumsum(eigval/np.sum(eigval))
plt.plot(x, c_val*100);
plt.grid()
plt.xlabel('Number of PCs')
plt.ylabel('Cumulative percentage in %')
plt.title('Cumulative percentage over number of PCs for MNIST_179_digits data set');
plt.show()


# In[29]:


print_pca_cluster(2)


# In[30]:


print_pca_cluster(20)


# In[31]:


print_pca_cluster(200)


# c) Classification. Train a k-NN classifier on the dataset:
# 
# – Do it using the first 20 principal components. Use n-fold validation to obtain the optimal k. Report test accuracy.
# 
# – Do it again with the first 200 principal components. Use n-fold validation to obtain the optimal k. Report test accuracy.

# In[32]:


def print_pca_classifier(num_comp):
    XTrain_minst_pca, _ = mds(minstXTrain, num_comp)
    XTest_minst_pca, _ = mds(minstXTest, num_comp)
    kb_pca, acc_pca = get_kbest_and_acc(k_list, XTrain_minst_pca, XTest_minst_pca, minstYTrain, minstYTest)
    print(f'Accuracy for MNIST_179_digits data set\nwith train data for the first {num_comp} PCs: {round(acc_pca,4)} with k_best={kb}')


# In[33]:


print_pca_classifier(20)


# In[34]:


print_pca_classifier(200)


# In[ ]:




