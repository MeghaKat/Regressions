# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 16:25:18 2019

@author: Ayesha
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

import time



def loadData():
    with np.load('notMNIST.npz') as data :
        Data, Target = data ['images'], data['labels']
        posClass = 2
        negClass = 9
        dataIndx = (Target==posClass) + (Target==negClass)
        Data = Data[dataIndx]/255.
        Target = Target[dataIndx].reshape(-1, 1)
        Target[Target==posClass] = 1
        Target[Target==negClass] = 0
        np.random.seed(421)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data, Target = Data[randIndx], Target[randIndx]
        trainData, trainTarget = Data[:3500], Target[:3500]
        validData, validTarget = Data[3500:3600], Target[3500:3600]
        testData, testTarget = Data[3600:], Target[3600:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget


#######################################1.Linear Regression Solutions #####################################################
    
################################ Linear Regression Solution 1.1################################################
#Loss Function and Gradient
#Reutrns MSE(mean Squared error)
def MSE(W, b, x, y, reg):
    # Your implementation here
    y_pred = np.matmul(x,W).reshape(x.shape[0],1)
    Ld_norm = LA.norm(y_pred - y + b)
    Ld = np.square(Ld_norm)/(2.0*train_N)
    Lw = (reg/2.0)*np.square(LA.norm(W))
    
    return Ld + Lw

#Reutrns gradient of MSE(mean Squared error) w.r.to W and b
def gradMSE(W, b, x, y, reg):
    # Your implementation here
    N = x.shape[0]
    l = (np.matmul(x,W) + b - y)
    
    gradient_wrt_W = np.matmul(x.T,l)/N + reg*W
    gradient_wrt_b = np.sum(l)/N
    
    return gradient_wrt_W,gradient_wrt_b
    
################################ Linear Regression Solution 1.2 ################################################
#Gradient Descent Implementation
#Implements the gradient descent algorithm and Returns trained weights and biases
def grad_descent(W, b, x, y, alpha, epochs, reg, error_tol, lossType="None"):
#Your implementation here#
    result_set = {
            'iterations':[],
            'train_error':[],'train_accuracy':[],
            'test_error':[],'test_accuracy':[],
            'validation_error':[],'validation_accuracy':[]
            }
    
    W_updated = W.copy()
    b_updated = np.array([b])

    #start timing the algorithm
    start_time = time.clock()
    
    for i in range(0,epochs):

        W_old = W_updated
        b_old = b_updated
        
        if(lossType == "MSE"):
            grad_wrt_W, grad_wrt_b = gradMSE(W_old,b_old,x,y,reg)
        elif (lossType == "CE"):
            grad_wrt_W, grad_wrt_b = gradCE(W_old,b_old,x,y,reg)
        
        W_updated = W_old - (alpha * grad_wrt_W)
        b_updated = b_old - (alpha * grad_wrt_b)
        
        result_set['iterations'].append(i+1)
        
        #recording the errors for training, validation and test dataset
        result_set['train_error'].append(calculate_loss(W_updated,b_updated,x,y,reg,lossType))
        result_set['test_error'].append(calculate_loss(W_updated,b_updated,testData,testTarget,reg,lossType))
        result_set['validation_error'].append(calculate_loss(W_updated,b_updated,validData,validTarget,reg,lossType))
                
        #recording the accuracy for training, validation and test dataset
        result_set['train_accuracy'].append(accuracy(W_updated,b_updated,x,y,lossType))
        result_set['test_accuracy'].append(accuracy(W_updated,b_updated,testData,testTarget,lossType))
        result_set['validation_accuracy'].append(accuracy(W_updated,b_updated,validData,validTarget,lossType))
        
        if ((i+1)%500 == 0):
            print("------------------iteration",i+1)
            print("difference in weights: ",LA.norm(W_old - W_updated))

        if LA.norm(np.concatenate([[b_old],W_old]) - np.concatenate([[b_updated],W_updated])) <= error_tol:
            print("converged")
            break
        
        
    #end for
    elapsed_time = time.clock() - start_time
    print("elapsed time",elapsed_time)
    
    print("optimized weights","\noptimized bias",b_updated)
    
    print("----- FOR  alpha, epochs, reg, error_tol\n", alpha, epochs, reg, error_tol)
    
    
    print("train error\t",result_set['train_error'][-1])
    print("validation error\t",result_set['validation_error'][-1])
    print("test error\t",result_set['test_error'][-1])
    
    print("training_accuracy\t",result_set['train_accuracy'][-1])
    print("validation_accuracy\t",result_set['validation_accuracy'][-1])
    print("test_accuracy\t",result_set['test_accuracy'][-1])
    
    
    return {"W_optimized": W_updated,
            "bias_optimized": b_updated,
            "elapsed_time":elapsed_time,
            "result_set":result_set,
            "final_accuracy":result_set['test_accuracy'][-1],
            "train_mse":result_set['train_error'][-1]}
        
########################################################   End of Linear Regression Solutions   ###########################################################        
    

########################################################    2.1. Logistic Regression Solutions   ###########################################################
    
################################ Logistic Regression Solution 2.1.1################################################
#Loss Function and Gradient 
 #returns the cross entropy loss
def crossEntropyLoss(W, b, x, y, reg):
     # Your implementation here
    Z= np.multiply(W.T, x) + b
    A = sigmoid(Z)
    N = x.shape[0]
   
    cost =  np.mean(np.multiply(y, np.log(A)) + np.multiply(1.0-y, np.log(1.0 - A)))+((reg/2.0)*np.square(LA.norm(W)))
    
    return cost
       
#returns the gradient of cross entropy loss w.r.t W and b
def gradCE(W, b, x, y, reg):
     # Your implementation here

    N = x.shape[0]

    Z= np.matmul(x,W) + b
    A = sigmoid(Z)
    
    l1 = np.sum(np.multiply(x,A),axis=0).reshape(784,1)
    l2 = np.matmul(x.T,y)
        
    dw= (l1-l2)*(1.0/N)+reg*W
    db = np.mean((A - y))
   
    return dw,db
################################ Logistic Regression Functions################################################

#Calculates the loss based on lossType
def calculate_loss(W, b, x, y, reg, lossType):
    if lossType == "MSE":
       return MSE(W,b,x,y,reg)
    elif lossType == "CE":
       return crossEntropyLoss(W,b,x,y,reg)

    
def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
    """
    
    s = 1.0/(1.0 + np.exp(-1.0 * z))
    
    return s


def accuracy(W,b,x,y,lossType): 
 
    y_interim = np.matmul(x,W).reshape(x.shape[0],1) +b
    if lossType == "MSE":
        output = np.sign(y_interim)
        y_pred = np.where(output>0,1,0)
    elif lossType == "CE":
        output = sigmoid(y_interim)
        y_pred = np.where(output>=0.5,1,0)

    correct_predictions = np.sum(np.where(y==y_pred,1,0))
    return correct_predictions*100/y.shape[0]

def closed_form_linear_regression(x,y):
    #calculating the optimal weight
    start_time = time.clock()
    
    #append column of ones to convert x to d+1 dimensions
    col_of_ones = np.ones((3500,1))
    x = np.append(col_of_ones,x,axis=1)
    intermediate = np.matmul(LA.inv(np.matmul(x.T,x)),x.T)
    optimized_weights = np.matmul(intermediate,y)
    W_optimized = optimized_weights[1:]
    b_optimized = optimized_weights[0]
    
    elapsed_time = time.clock() - start_time
    
    return W_optimized,b_optimized,elapsed_time
        
################################ End of Logistic Regression Functions################################################    

########################## LOADING THE DATA #####################################################


#loading the data 
trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
print("trainData",trainData.shape)
print("trainTarget",trainTarget.shape)

train_N = trainData.shape[0]
valid_N = validData.shape[0]
test_N = testData.shape[0]
print("train_N",train_N, "valid_N",valid_N,"test_N",test_N)

#reshaping the data from 3d to 2d matrix
trainData = trainData.reshape(train_N,784)
validData = validData.reshape(valid_N,784)
testData = testData.reshape(test_N,784)

graph_directory = "C:\\Users\\Ayesha\\Documents\\UofT\\Winter 2019\\ECE 421 - Intro to ML\\Assignments\\Graphs Final\\Q1\\P3\\"

########################################################  1.Linear Regression Solutions   ###########################################################
################################ Linear Regression Solution 1.3 ################################################
#Tuning the learning rate
#initialize model parameters
bias = 0
W = np.ones((784,1))
no_of_epochs = 5000
reg = 0 #lambda
error_tolerance = 1e-7
alphas = [0.005,0.001,0.0001]

results_for_alphas = []
model_accuracies = []
training_times = []
color_array = ['r','b','g']

for alpha in alphas:
    
    output = grad_descent(W, bias, trainData, trainTarget, alpha, no_of_epochs, reg, error_tolerance,"MSE")
    
    result_set = output['result_set']
    results_for_alphas.append(result_set)
    training_times.append(output['elapsed_time'])
    
    #plot for each alpha
    plt.plot(result_set['iterations'],result_set['train_error'],c='r',label='Training Error')
    plt.plot(result_set['iterations'],result_set['validation_error'],c='b',label='Validation Error')
    plt.plot(result_set['iterations'],result_set['test_error'],c='g',label='Test Error')
    plt.legend()
    plt.title("Error Curves: Lambda = {}, Epochs = {}, Learning Rate={}".format(reg,no_of_epochs,alpha))
    plt.show()
    #plot for each alpha
    plt.plot(result_set['iterations'],result_set['train_accuracy'],c='r',label='Training Accuracy')
    plt.plot(result_set['iterations'],result_set['validation_accuracy'],c='b',label='Validation Accuracy')
    plt.plot(result_set['iterations'],result_set['test_accuracy'],c='g',label='Test Accuracy')
    plt.legend()
    plt.title("Accuracy Curves: Lambda = {}, Epochs = {}, Learning Rate={}".format(reg,no_of_epochs,alpha))
    plt.show()

#Alpha Comparisons
#Train Error
for i in range(len(alphas)):

    plt.plot( results_for_alphas[i]['iterations'],results_for_alphas[i]['train_error'],c=color_array[i%3],label='Alpha: {}'.format(alphas[i]))
   

plt.xlabel('iterations')
plt.ylabel('training error')
plt.legend()
plt.savefig(graph_directory+"Alpha_Loss_Comparisions.png")
plt.close()

#Test Accuracy
for i in range(len(alphas)):
    plt.plot( results_for_alphas[i]['iterations'],results_for_alphas[i]['test_accuracy'],c=color_array[i%3],label='Alpha: {}'.format(alphas[i]))

plt.xlabel('iterations')
plt.ylabel('test accuracy')
plt.legend()
plt.savefig(graph_directory+"Alphas_Accuracy_Comparisions.png")
plt.close()

#Training Time
plt.plot( alphas,training_times,c=color_array[i%3],marker='x')

plt.xlabel('Learning Rate')
plt.ylabel('Training Time (s)')
plt.show()

    

################################# Linear Regression Solution 1.4 ################################################
#Generalization
bias = 0
W = np.zeros((784,1))

alpha = 0.005
reg_params = [0.001,0.1,0.5]
no_of_epochs = 5000

result_for_regs = []
metrics=['train_error','validation_error','test_error','train_accuracy','test_accuracy','validation_accuracy']

for reg in reg_params:
    output = grad_descent(W, bias, trainData, trainTarget, alpha, no_of_epochs, reg, 1e-7,"MSE")
    result_for_regs.append(output['result_set'])
    
    
    print("--- regularization param:{} \t accuracy:{}".format(reg,output['final_accuracy']))
  
for metric in metrics:
    for i in range(len(reg_params)):
        plt.plot(result_for_regs[i]['iterations'],result_for_regs[i][metric],c=color_array[i%3],label='lambda = {}'.format(reg_params[i]))
        plt.xlabel('Iterations')
        plt.ylabel(metric)
        plt.legend()
    plt.show()




################################# Linear Regression Solution 1.5 ################################################
#Computing Batch GD with normal equation
bias = 1
W = np.ones((784,1))

####fine tuned hyperparameters
no_of_epochs = 5000
alpha = 0.005
reg = 0
error_tolerance = 1e-7

#batch gradient descent 
final_output = grad_descent(W,bias,trainData,trainTarget,alpha,no_of_epochs,reg,error_tolerance,"MSE")
print("--------------gradient descent:\n","elapsed time:\t",final_output["elapsed_time"],"\naccuracy:\t",final_output["final_accuracy"],"\ntraining error:\t",final_output["train_mse"])

#closed form linear regression
W_optimized, b_optimized, cf_elapsed_time = closed_form_linear_regression(trainData,trainTarget)
cf_train_mse = MSE(W_optimized,b_optimized,trainData,trainTarget,reg)
cf_accuracy = accuracy(W_optimized, b_optimized,testData,testTarget,"MSE")
cf_train_accuracy = accuracy(W_optimized, b_optimized,trainData,trainTarget,"MSE")

print("-------------closed form:\n","elapsed time:\t",cf_elapsed_time,"\naccuracy:\t",cf_accuracy,"\ntraining error:\t",cf_train_mse,"\ntrain_accuracy",cf_train_accuracy)


########################################################   End of Linear Regression Solutions    ###########################################################



########################################################    2.1.Logistic Regression Solutions   ###########################################################
################################ Logistic Regression Solution 2.1.2################################################
#Learning
# Report on the performance of Logistic Regression Model by setting lambda =0.1 and 500 epochs
bias = 1
W = np.ones((784,1))

reg =0.1
alpha = 0.001

no_of_epochs = 5000
error_tolerance =  1e-7

color_array = ['r','b','g']


output = grad_descent(W, bias, trainData, trainTarget, alpha, no_of_epochs, reg,error_tolerance,"CE")
result_set = output['result_set']

#plot the loss curves

plt.title("CE Loss Curves: Lambda = {}, Epochs = {}, Learning Rate={}".format(reg,no_of_epochs,alpha))

metrics = ['train_error','validation_error','test_error']
for i,metric in enumerate(metrics):
    plt.plot(result_set['iterations'],result_set[metric],c= color_array[i%3],label=metric)
plt.xlabel("iterations")
plt.ylabel("loss")
plt.legend()
plt.show()

#plot the accuracy curves
plt.title("CE Accuracy Curves: Lambda = {}, Epochs = {}, Learning Rate={}".format(reg,no_of_epochs,alpha))

metrics = ['train_accuracy','validation_accuracy','test_accuracy']
for i,metric in enumerate(metrics):
    plt.plot(result_set['iterations'],result_set[metric],c= color_array[i%3],label=metric)
plt.xlabel("iterations")
plt.ylabel("accuracy %")
plt.legend()
plt.show()
################################ Logistic Regression Solution 2.1.3################################################
#Comparision to Linear Regression
bias = 1
W = np.ones((784,1))

reg =0 #no regularization
alpha = 0.005

no_of_epochs = 5000
error_tolerance =  1e-7

MSE_output = grad_descent(W, bias, trainData, trainTarget, alpha, no_of_epochs, reg,error_tolerance,"MSE")
MSE_noreg_result_set = MSE_output['result_set']

CE_output = grad_descent(W, bias, trainData, trainTarget, alpha, no_of_epochs, reg,error_tolerance,"CE")
CE_noreg_result_set = CE_output['result_set']

####### Plot the loss curves for both MSE and CE

plt.plot(MSE_noreg_result_set['iterations'],MSE_noreg_result_set['train_error'],c='r',label='Training Error')
plt.plot(MSE_noreg_result_set['iterations'],MSE_noreg_result_set['validation_error'],c='b',label='Validation Error')
plt.plot(MSE_noreg_result_set['iterations'],MSE_noreg_result_set['test_error'],c='g',label='Test Error')
plt.legend()
plt.title("MSE Loss Curves: Lambda = {}, Epochs = {}, Learning Rate={}".format(reg,no_of_epochs,alpha))
plt.xlabel("iterations")
plt.ylabel("loss")
plt.show()
#plt.savefig()

plt.plot(CE_noreg_result_set['iterations'],CE_noreg_result_set['train_error'],c='r',label='Training Error')
plt.plot(CE_noreg_result_set['iterations'],CE_noreg_result_set['validation_error'],c='b',label='Validation Error')
plt.plot(CE_noreg_result_set['iterations'],CE_noreg_result_set['test_error'],c='g',label='Test Error')
plt.legend()
plt.title("CE Loss Curves: Lambda = {}, Epochs = {}, Learning Rate={}".format(reg,no_of_epochs,alpha))
plt.xlabel("iterations")
plt.ylabel("loss")
plt.show()

plt.plot(MSE_noreg_result_set['iterations'],MSE_noreg_result_set['train_error'],c='r',label='MSE')
plt.plot(CE_noreg_result_set['iterations'],CE_noreg_result_set['train_error'],c='b',label='CE')
plt.legend()
plt.title("Training Loss Curves: Lambda = {}, Epochs = {}, Learning Rate={}".format(reg,no_of_epochs,alpha))
plt.xlabel("iterations")
plt.ylabel("loss")
plt.show()
####### Plot the accuracy curves for both MSE and CE

plt.plot(MSE_noreg_result_set['iterations'],MSE_noreg_result_set['train_accuracy'],c='r',label='Training Accuracy')
plt.plot(MSE_noreg_result_set['iterations'],MSE_noreg_result_set['validation_accuracy'],c='b',label='Validation Accuracy')
plt.plot(MSE_noreg_result_set['iterations'],MSE_noreg_result_set['test_accuracy'],c='g',label='Test Accuracy')
plt.legend()
plt.title("MSE Accuracy Curves: Lambda = {}, Epochs = {}, Learning Rate={}".format(reg,no_of_epochs,alpha))
plt.xlabel("iterations")
plt.ylabel("accuracy")
plt.show()
    

plt.plot(CE_noreg_result_set['iterations'],CE_noreg_result_set['train_accuracy'],c='r',label='Training Accuracy')
plt.plot(CE_noreg_result_set['iterations'],CE_noreg_result_set['validation_accuracy'],c='b',label='Validation Accuracy')
plt.plot(CE_noreg_result_set['iterations'],CE_noreg_result_set['test_accuracy'],c='g',label='Test Accuracy')
plt.legend()
plt.title("CE Accuracy Curves: Lambda = {}, Epochs = {}, Learning Rate={}".format(reg,no_of_epochs,alpha))
plt.xlabel("iterations")
plt.ylabel("accuracy")
plt.show()
#plt.savefig()

plt.plot(MSE_noreg_result_set['iterations'],MSE_noreg_result_set['test_accuracy'],c='r',label='MSE')
plt.plot(CE_noreg_result_set['iterations'],CE_noreg_result_set['test_accuracy'],c='b',label='CE')
plt.legend()
plt.title("Test Accuracy Curves: Lambda = {}, Epochs = {}, Learning Rate={}".format(reg,no_of_epochs,alpha))
plt.xlabel("iterations")
plt.ylabel("accuracy")
plt.show()
########################################################   End of Logistic Regression Solutions   ###########################################################



