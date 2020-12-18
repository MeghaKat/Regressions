# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 13:34:19 2019

@author: megha
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

graph_directory = "C:\\Users\\Ayesha\\Documents\\UofT\\Winter 2019\\ECE 421 - Intro to ML\\Assignments\\A1 Q3 Graphs\\"

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
    return Data, Target

########################################################  3.1. Batch Gradient Descent vs SGD and Adam   ###########################################################
########################################################  3.1.1 Solution Function   ###########################################################
#Building the Computational Graph

def buildGraph(beta1=None, beta2=None, epsilon=None, lossType=None, learning_rate=None):

    tf.set_random_seed(421)

    # weight and bias initialization
    #variables because we want to update and optimize
    W= tf.Variable(tf.truncated_normal([784,1]),trainable=True,dtype=tf.float32)
    b= tf.Variable(tf.truncated_normal([1]),trainable=True,dtype=tf.float32)

    #initialize placeholders for data, labels and regularization param
    x=tf.placeholder(dtype=tf.float32,shape=[None,784],name="x")
    y=tf.placeholder(dtype=tf.float32,shape=[None,1],name="y")
    y_pred = tf.placeholder(dtype=tf.float32,shape=[None,1],name='y_pred')


    reg=tf.placeholder(dtype=tf.float32,name="reg")


    y_intermediate = tf.matmul(x, W) + b
    #initialize placeholders for loss tensor
    if lossType=="MSE":

        y_pred = y_intermediate
        error=tf.losses.mean_squared_error(y,y_pred) + reg*tf.nn.l2_loss(W)

    elif lossType=="CE":
        y_pred = tf.sigmoid(y_intermediate)
        error=tf.reduce_mean(tf.losses.sigmoid_cross_entropy(y,y_pred)) + reg*tf.nn.l2_loss(W)


    if(beta1!=None and beta2==None and epsilon==None):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=beta1).minimize(error) #beta2 and epsilon paramters kept as default
    if(beta1==None and beta2!=None and epsilon==None):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,beta2=beta2).minimize(error) #beta1 and epsilon paramters kept as default
    if(beta1==None and beta2==None and epsilon!=None):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,epsilon=epsilon).minimize(error)#beta1 and beta2 paramters kept as default

    else:
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(error)


    if lossType == "MSE":
        y_pred_adjusted = tf.cast(tf.sign(y_pred) >= 0, y_intermediate.dtype)
        correct_predictions =tf.cast( tf.equal(y_pred_adjusted,y),tf.float32)
        accuracy = tf.reduce_mean(correct_predictions)*100
    elif lossType == "CE":
        y_pred_adjusted = tf.cast(y_pred >= 0.5, y_intermediate.dtype)
        correct_predictions =tf.cast( tf.equal(y_pred_adjusted,y),tf.float32)
        accuracy = tf.reduce_mean(correct_predictions)*100


    return W,b, x, y, y_pred, optimizer, reg, error, accuracy


########################################################  3.1.2 Solution Function   ###########################################################
    #Implementing Stochastic Gradient Descent

def SGD(epochs,minibatch_size = 500, lossType = "MSE"):
    Data, Target = loadData()
    Data = Data.reshape(Data.shape[0],784)

    train_N = 3500

    no_of_batches = int(train_N/minibatch_size)
    alpha = 0.001
    regulization_coeff = 0

    result_set = {'iterations':[],
                  'train_error':[],'train_accuracy':[],
                  'test_error':[],'test_accuracy':[],
                  'validation_error':[],'validation_accuracy':[]
                  }

    np.random.seed(421)
    sess = tf.InteractiveSession()
    #operations: y_pred, train_opt, accuracy and error
    #placeholders: x,y,reg
    #variables: W, b
    W,b, x, y, y_pred, train_opt, reg, error, accuracy = buildGraph(lossType=lossType,learning_rate=alpha)

    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    for epoch_no in range(0,epochs):

        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        trainData, trainTarget = Data[:3500], Target[:3500]
        validData, validTarget = Data[3500:3600], Target[3500:3600]
        testData, testTarget = Data[3600:], Target[3600:]

        for batch_no in range(0,no_of_batches):
            startIndex = minibatch_size*(batch_no)
            endIndex = minibatch_size*(batch_no+1)
            tData_batch = trainData[startIndex:endIndex]
            tTarget_batch = trainTarget[startIndex:endIndex]


            optimizer,trainPred_batch, trainBatch_error,trainBatch_accuracy = sess.run([train_opt,y_pred,error,accuracy],feed_dict={x:tData_batch,y:tTarget_batch, reg:regulization_coeff})
#            optimizer, trainBatch_accuracy= sess.run([train_opt,accuracy],feed_dict={x:tData_batch,y:tTarget_batch, reg:regulization_coeff})

#            trainPred_batch = sess.run(y_pred,feed_dict={x:tData_batch,y:tTarget_batch, reg:regulization_coeff})
#            trainBatch_accuracy = sess.run(accuracy(trainPred_batch,tTarget_batch))
#            trainBatch_accuracy = sess.run([accuracy],feed_dict={y_pred:trainPred_batch,x:tData_batch,y:tTarget_batch, reg:regulization_coeff})
#            optimizer, trainBatch_error,trainBatch_pred, trainBatch_accuracy = sess.run([train_opt,error,y_pred,accuracy],feed_dict={x:tData_batch,y:tTarget_batch, reg:regulization_coeff})

            test_error, test_accuracy = sess.run([error, accuracy], feed_dict={x:testData,y:testTarget, reg:regulization_coeff})
            validation_error, validation_accuracy = sess.run([error, accuracy], feed_dict={x:validData,y:validTarget, reg:regulization_coeff})

        if (epoch_no % 100 == 0):
            print("epoch ",epoch_no, " batch_no ",batch_no,"\nerror ",trainBatch_error)
        result_set['iterations'].append((epoch_no+1))
        result_set['train_error'].append(trainBatch_error)
        result_set['train_accuracy'].append(trainBatch_accuracy)
        result_set['test_error'].append(test_error)
        result_set['test_accuracy'].append(test_accuracy)
        result_set['validation_error'].append(validation_error)
        result_set['validation_accuracy'].append(validation_accuracy)

    sess.close()


    #plot epochs vs errors
    print("Batchsize:",minibatch_size)
    print("Final_train_accuracy:",result_set['train_accuracy'][-1])
    print("Final_test_accuracy:",result_set['test_accuracy'][-1])
    print("Final_validation_accuracy:",result_set['validation_accuracy'][-1])


    plt.plot(result_set['iterations'],result_set['train_error'],c='r',label='Training Error')
    plt.plot(result_set['iterations'],result_set['test_error'],c='b',label='Test Error')
    plt.plot(result_set['iterations'],result_set['validation_error'],c='g', label='Validation Error')

    plt.xlabel('Epochs')
    plt.ylabel('Errors')
    plt.legend()
    plt.title('Loss Curves: Batch size: {}'.format(minibatch_size))

#    plt.show()
    plt.savefig(graph_directory+"loss_curve_bs_{}.png".format(minibatch_size))
    plt.close()


    #plot epochs vs accuracies
    plt.plot(result_set['iterations'],result_set['train_accuracy'],c='r',label='Training Accuracy')
    plt.plot(result_set['iterations'],result_set['test_accuracy'],c='b',label='Test Accuracy')
    plt.plot(result_set['iterations'],result_set['validation_accuracy'],c='g', label='Validation Accuracy')

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curves: Batch size: {}'.format(minibatch_size))

#    plt.show()
    plt.savefig(graph_directory+"accuracy_curve_bs_{}.png".format(minibatch_size))
    plt.close()
    return result_set

     ########################################################  Functions   ###########################################################

def SGD_with_hyperparamters(epochs,minibatch_size = 500,beta1=None, beta2=None, epsilon=None, lossType=None, learning_rate=None):
    Data, Target = loadData()
    Data = Data.reshape(Data.shape[0],784)

    train_N = 3500

    no_of_batches = int(train_N/minibatch_size)
    regulization_coeff = 0

    result_set = {'iterations':[],
                  'train_error':[],'train_accuracy':[],
                  'test_error':[],'test_accuracy':[],
                  'validation_error':[],'validation_accuracy':[]
                  }

    hyperparamters_setting = ''
    if (beta1!=None):
        hyperparamters_setting += 'Beta1 {} '.format(beta1)
    if (beta2!=None):
        hyperparamters_setting += 'Beta2 {} '.format(beta2)
    if (epsilon!=None):
        hyperparamters_setting += 'Epsilon {} '.format(epsilon)
    hyperparamters_setting+=' BatchSize {}, Epochs {}, LossType {}, Learning Rate {}'.format(minibatch_size,epochs,lossType,learning_rate)

    np.random.seed(421)

    sess = tf.InteractiveSession()


    #operations: y_pred, train_opt, accuracy and error
    #placeholders: x,y,reg
    #variables: W, b
    W,b, x, y, y_pred, train_opt, reg, error, accuracy = buildGraph(beta1, beta2, epsilon, lossType, learning_rate)


    init_op = tf.global_variables_initializer()
    sess.run(init_op)


    for epoch_no in range(0,epochs):

        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        trainData, trainTarget = Data[:3500], Target[:3500]
        validData, validTarget = Data[3500:3600], Target[3500:3600]
        testData, testTarget = Data[3600:], Target[3600:]

        for batch_no in range(0,no_of_batches):
            startIndex = minibatch_size*(batch_no)
            endIndex = minibatch_size*(batch_no+1)
            tData_batch = trainData[startIndex:endIndex]
            tTarget_batch = trainTarget[startIndex:endIndex]


            optimizer,trainPred_batch, trainBatch_error,trainBatch_accuracy = sess.run([train_opt,y_pred,error,accuracy],feed_dict={x:tData_batch,y:tTarget_batch, reg:regulization_coeff})
#            optimizer, trainBatch_accuracy= sess.run([train_opt,accuracy],feed_dict={x:tData_batch,y:tTarget_batch, reg:regulization_coeff})

#            trainPred_batch = sess.run(y_pred,feed_dict={x:tData_batch,y:tTarget_batch, reg:regulization_coeff})
#            trainBatch_accuracy = sess.run(accuracy(trainPred_batch,tTarget_batch))
#            trainBatch_accuracy = sess.run([accuracy],feed_dict={y_pred:trainPred_batch,x:tData_batch,y:tTarget_batch, reg:regulization_coeff})



#            optimizer, trainBatch_error,trainBatch_pred, trainBatch_accuracy = sess.run([train_opt,error,y_pred,accuracy],feed_dict={x:tData_batch,y:tTarget_batch, reg:regulization_coeff})
#
            test_error, test_accuracy = sess.run([error, accuracy], feed_dict={x:testData,y:testTarget, reg:regulization_coeff})
#
            validation_error, validation_accuracy = sess.run([error, accuracy], feed_dict={x:validData,y:validTarget, reg:regulization_coeff})


        if (epoch_no % 100 == 0):
            print("epoch ",epoch_no, " batch_no ",batch_no,"\nerror ",trainBatch_error)
#            print("accuracy: ",train_accuracy)
        result_set['iterations'].append((epoch_no+1))
        result_set['train_error'].append(trainBatch_error)
        result_set['train_accuracy'].append(trainBatch_accuracy)
        result_set['test_error'].append(test_error)
        result_set['test_accuracy'].append(test_accuracy)
        result_set['validation_error'].append(validation_error)
        result_set['validation_accuracy'].append(validation_accuracy)

   print("Final_train_accuracy:",result_set['train_accuracy'][-1])
   print("Final_test_accuracy:",result_set['test_accuracy'][-1])
   print("Final_validation_accuracy:",result_set['validation_accuracy'][-1])



    sess.close()

    #plot epochs vs errors

    plt.plot(result_set['iterations'],result_set['train_error'],c='r',label='Training Error')
    plt.plot(result_set['iterations'],result_set['test_error'],c='b',label='Test Error')
    plt.plot(result_set['iterations'],result_set['validation_error'],c='g', label='Validation Error')

    plt.xlabel('Epochs')
    plt.ylabel('Errors')
    plt.legend()
    plt.title('Loss Curves:'+hyperparamters_setting)

#    plt.show()
    plt.savefig(graph_directory+"loss_curve_{}.png".format(hyperparamters_setting))
    plt.close()


    #plot epochs vs accuracies
    plt.plot(result_set['iterations'],result_set['train_accuracy'],c='r',label='Training Accuracy')
    plt.plot(result_set['iterations'],result_set['test_accuracy'],c='b',label='Test Accuracy')
    plt.plot(result_set['iterations'],result_set['validation_accuracy'],c='g', label='Validation Accuracy')

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curves:'+hyperparamters_setting)

#    plt.show()
    plt.savefig(graph_directory+"accuracy_curve_{}_.png".format(hyperparamters_setting))
    plt.close()
    return result_set
#####################################################  End of Functions   ####################################
########################################################  3.1.2 Solution   ###########################################################
#Implementing Stochastic Gradient Descent
result_MSE=SGD(epochs=700,minibatch_size = 500, lossType="MSE")
########################################################  3.1.3 Solution   ###########################################################
# Batch Size Investigation
for batch_size in [100,700,1750]:
    SGD(epochs=700,minibatch_size = batch_size,lossType="MSE")

########################################################  3.1.4 Solution   ###########################################################
#Hyperparamter Investigation
#Investogating impact of Beta1
for beta1 in [0.95,0.99]:
    SGD_with_hyperparamters(epochs=700,minibatch_size=500,learning_rate=0.001,beta1=beta1, lossType="MSE")


#Investogating impact of Beta2
for beta2 in [0.99,0.9999]:
    SGD_with_hyperparamters(epochs=700,minibatch_size=500,learning_rate=0.001,beta2=beta2, lossType="MSE")

#Investogating impact of epsilon
for epsilon in [1e-9,1e-4]:
    SGD_with_hyperparamters(epochs=700,minibatch_size=500,learning_rate=0.001,epsilon=epsilon, lossType="MSE")



########################################################  3.1.5 Solution   ###########################################################
#Cross Entropy Loss Investigation
result_set_CE=SGD(epochs=700,minibatch_size = 500, lossType="CE")

for batch_size in [100,700,1750]:
    SGD(epochs=700,minibatch_size = batch_size,lossType="CE")


#Investogating impact of Beta1
for beta1 in [0.95,0.99]:
    SGD_with_hyperparamters(epochs=700,minibatch_size=500,learning_rate=0.001,beta1=beta1, lossType="CE")

#Investogating impact of Beta2
for beta2 in [0.99,0.9999]:
    SGD_with_hyperparamters(epochs=700,minibatch_size=500,learning_rate=0.001,beta2=beta2, lossType="CE")

#Investogating impact of epsilon
for epsilon in [1e-9,1e-4]:
    SGD_with_hyperparamters(epochs=700,minibatch_size=500,learning_rate=0.001,epsilon=epsilon, lossType="CE")

########################################################  3.1.6 Solution   ###########################################################
#Comparison against Batch GD
#---------------------------------------Accuracy Graphs------------------------------------------------
#MSE Accuracy Graph
plt.plot( result_MSE['iterations'],result_MSE['train_accuracy'],c='r',label='Training Accuracy' )
plt.plot( result_MSE['iterations'],result_MSE['test_accuracy'],c='b',label='Test Accuracy' )
plt.plot( result_MSE['iterations'],result_MSE['validation_accuracy'],c='g', label='Validation Accuracy' )
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(graph_directory+"accuracy_MSE.png")
plt.close()
#CE Accuracy Graph
plt.plot( result_set_CE['iterations'],result_set_CE['train_accuracy'],c='y',label='Training Accuracy' )
plt.plot( result_set_CE['iterations'],result_set_CE['test_accuracy'],c='m',label='Test Accuracy' )
plt.plot( result_set_CE['iterations'],result_set_CE['validation_accuracy'],c='c', label='Validation Accuracy' )
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(graph_directory+"accuracy_CE.png")
plt.close()
#MSE vs CE Graphs
#Training Accuracy
plt.plot( result_MSE['iterations'],result_MSE['train_accuracy'],c='r',label='Training AccuracyMSE' )
plt.plot( result_set_CE['iterations'],result_set_CE['train_accuracy'],c='y',label='Training AccuracyCE' )
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(graph_directory+"TrainingAccurayComparision.png")
plt.close()
#Test Accuracy
plt.plot( result_MSE['iterations'],result_MSE['test_accuracy'],c='r',label='Test AccuracyMSE' )
plt.plot( result_set_CE['iterations'],result_set_CE['test_accuracy'],c='y',label='Test AccuracyCE' )
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(graph_directory+"TestAccurayComparision.png")
plt.close()
#Validation Accuracy
plt.plot( result_MSE['iterations'],result_MSE['validation_accuracy'],c='r',label='Validation AccuracyMSE' )
plt.plot( result_set_CE['iterations'],result_set_CE['validation_accuracy'],c='y',label='Validation AccuracyCE' )
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(graph_directory+"Validation AccuracyAccurayComparision.png")
plt.close()

#---------------------------------------Loss Graphs------------------------------------------------
#MSE loss Graph
plt.plot(result_MSE['iterations'],result_MSE['train_error'],c='r',label='Training Error')
plt.plot(result_MSE['iterations'],result_MSE['test_error'],c='b',label='Test Error')
plt.plot(result_MSE['iterations'],result_MSE['validation_error'],c='g', label='Validation Error')
plt.xlabel('Epochs')
plt.ylabel('Errors')
plt.legend()
plt.savefig(graph_directory+"Loss_MSE.png")
plt.close()
#CE Loss Graph
plt.plot(result_set_CE['iterations'],result_set_CE['train_error'],c='r',label='Training Error')
plt.plot(result_set_CE['iterations'],result_set_CE['test_error'],c='b',label='Test Error')
plt.plot(result_set_CE['iterations'],result_set_CE['validation_error'],c='g', label='Validation Error')
plt.xlabel('Epochs')
plt.ylabel('Errors')
plt.legend()
plt.savefig(graph_directory+"Loss_CE.png")
plt.close()
#---------------------------------------MSE vs CE Graphs------------------------------
#Training Error
plt.plot(result_MSE['iterations'],result_MSE['train_error'],c='r',label='Training ErrorMSE')
plt.plot(result_set_CE['iterations'],result_set_CE['train_error'],c='g',label='Training ErrorCE')
plt.xlabel('Epochs')
plt.ylabel('Errors')
plt.legend()
plt.savefig(graph_directory+"TrainingLossComparision.png")
plt.close()
#Test Error
plt.plot(result_MSE['iterations'],result_MSE['test_error'],c='r',label='Test ErrorMSE')
plt.plot(result_set_CE['iterations'],result_set_CE['test_error'],c='g',label='Test ErrorCE')
plt.xlabel('Epochs')
plt.ylabel('Errors')
plt.legend()
plt.savefig(graph_directory+"TestLossComparision.png")
plt.close()
#Validation Error
plt.plot(result_MSE['iterations'],result_MSE['validation_error'],c='r', label='Validation ErrorMSE')
plt.plot(result_set_CE['iterations'],result_set_CE['validation_error'],c='g', label='Validation ErrorCE')
plt.xlabel('Epochs')
plt.ylabel('Errors')
plt.legend()
plt.savefig(graph_directory+"ValidationLossComparision.png")
plt.close()
