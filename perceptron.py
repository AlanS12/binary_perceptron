
# Student Name: Alan Soby
# Student ID: 201687565 

import numpy as np
import pandas as pd

# reading the training and testing data from the respective files
# do change the file paths if needed
trainData = pd.read_csv('CA1data/train.data')
testData = pd.read_csv('CA1data/test.data')

# define a 'perceptron' function
    # get the feature vector, its length as arguments
    # initialize the weight vector
    # add x0 = 1 and its weight w0 - basically introduce the bias term
    # for loop:
        # calculate the activation score 
            # for loop:
                # traverse through the weight and feature vector 
                # get the activation score for that instance 
        # if activation score > 0:
            # no change
        # else if activation score <= 0:
            # update the weight vector including the bias for that instance
        # shuffle the training dataset for each iteration
        # do it for all the instances of the dataset
    # return the weight vector (including the bias)

def binaryPerceptron(processedData, reg_coeff):

    featureTrainMatrix, trainLabels = getFeaturesLabels(processedData)
    # we get the data separated into a feature matrix and a class vector
    featureTrainMatrix=featureTrainMatrix.values
    trainLabels=trainLabels.values
    # initialising the weights and the bias
    weight = np.zeros(4)
    bias = 0 
    
    # now we are training the model on the training data
    for iteration in range(20):      
        # the limit of iterations has been set to 20
        predictedTrainLabels = []
        actualTrainLabels = []
        for instance in range(len(featureTrainMatrix)):
            # for every instance or object in the training data, activation score and prediction are obtained
            activ_score = np.dot(weight, featureTrainMatrix[instance]) + bias
            predictedLabel = np.sign(activ_score)

            # if there is a misclassification by the prediction model, the weights and the bias are updated 
            if trainLabels[instance]*predictedLabel <= 0:
                for i in range(len(featureTrainMatrix[instance])):
                    weight[i] = (1-(2*reg_coeff))*weight[i] + trainLabels[instance]*featureTrainMatrix[instance][i]
                    bias += trainLabels[instance]
            # the obtained predictions and the actual labels are appended to a list for calculation of accuracy
            predictedTrainLabels.append(predictedLabel)
            actualTrainLabels.append(trainLabels[instance])

    # after 20 iterations...
    trainAccuracy = calculatingAccuracies(predictedTrainLabels, actualTrainLabels)
    print('Accuracy for training data: ', trainAccuracy, '\n')
    return weight, bias

    
def testPerceptron(weight, bias, processedData):
    # the test data is passed through the trained model to get the accuracy 
    featureTestMatrix, testLabels = getFeaturesLabels(processedData)
    featureTestMatrix = featureTestMatrix.values
    testLabels = testLabels.values
    predictedTestLabels = []
    actualTestLabels = []
    # getting the test data predictions
    for instance in range(len(featureTestMatrix)):
        testActivation = np.dot(weight, featureTestMatrix[instance]) + bias
        predictedTestLabels.append(np.sign(testActivation))
        actualTestLabels.append(testLabels[instance])
    testAccuracy = calculatingAccuracies(predictedTestLabels, actualTestLabels)
    print('Accuracy for test data: ', testAccuracy)


def getFeaturesLabels(data):
    featureMatrix = data.iloc[:, :4]
    labels = data.iloc[:, 4]
    return featureMatrix, labels

def oneClassProcessor(Data, positiveClass):
    # the work oneClassProcessor does is similar to what getProcessedClasses does. 
    # its just that this function does not drop any class from the data
    data = Data.copy()
    for instance in range(len(data)):
        if data.iloc[instance,4] == positiveClass:
            data.iloc[instance,4] = 1
        else:
            data.iloc[instance,4] = -1
    return data

def getProcessedClasses(Data, className):
    data = Data.copy()         
    # creating a duplicate of the original training data

    data = data[data.iloc[:,4]!=className]
    # excluding the class which is equal to 'className' from the data to classify the other two classes

    # changing the labels to 1 and -1
    if className == 'class-1':
        for i in range(data.shape[0]):
            if data.iloc[i,4] == 'class-2':
                data.iloc[i,4] = 1
            else:
                data.iloc[i,4] = -1
    elif className == 'class-2':
        for i in range(data.shape[0]):
            if data.iloc[i,4] == 'class-1':
                data.iloc[i,4] = 1
            else:
                data.iloc[i,4] = -1
    elif className == 'class-3':
        for i in range(data.shape[0]):
            if data.iloc[i,4] == 'class-1':
                data.iloc[i,4] = 1
            else:
                data.iloc[i,4] = -1

    return data

def calculatingAccuracies(predictedLabels, actualLabels):
    correctPredictions = 0
    totalInstances = len(predictedLabels)

    for instance in range(totalInstances):
        if predictedLabels[instance] == actualLabels[instance]:
            correctPredictions += 1
    accuracy = correctPredictions/totalInstances
    return round(accuracy*100, 2)

def multiclassClassification(reg_coeff):
    newTrainData = trainData.copy()
    newTestData = testData.copy()

    print('-------Prediction model for Class-1--------')
    class1 = oneClassProcessor(newTrainData, 'class-1')
    weight1, bias1 = binaryPerceptron(class1, reg_coeff)
    print('-------Prediction model for Class-2--------')
    class2 = oneClassProcessor(newTrainData, 'class-2')
    weight2, bias2 = binaryPerceptron(class2, reg_coeff)
    print('-------Prediction model for Class-3--------')
    class3 = oneClassProcessor(newTrainData, 'class-3')
    weight3, bias3 = binaryPerceptron(class3, reg_coeff)

    # the confidence score determines as to which class does an incoming instance or object belong to
    confidence_score = 0
    predictedTestLabels = []
    actualTestLabels = []
    featureTestMatrix, testLabels = getFeaturesLabels(newTestData)
    predictedTrainLabels = []
    actualTrainLabels = []
    featureTrainMatrix, trainLabels = getFeaturesLabels(newTrainData)

    for instance in range(featureTrainMatrix.shape[0]):
        # each train data instance is passed through all the prediction models and 
        # it is regarded as belonging to that class whose prediction model gives the highest activation score 
        activationTrain1 = np.dot(weight1, featureTrainMatrix.values[instance]) + bias1
        activationTrain2 = np.dot(weight2, featureTrainMatrix.values[instance]) + bias2
        activationTrain3 = np.dot(weight3, featureTrainMatrix.values[instance]) + bias3

        train_confidence_score = max(activationTrain1, activationTrain2, activationTrain3)
        if train_confidence_score == activationTrain1:
            predictedLabel_train = 'class-1'
        elif train_confidence_score == activationTrain2:
            predictedLabel_train = 'class-2'
        elif train_confidence_score == activationTrain3:
            predictedLabel_train = 'class-3'
        predictedTrainLabels.append(predictedLabel_train)
        actualTrainLabels.append(trainLabels.values[instance])

    for instance in range(featureTestMatrix.shape[0]):
        # each test data instance is passed all the prediction models and 
        # it is regarded as belonging to that class whose prediction model gives the highest activation score 
        activation1 = np.dot(weight1, featureTestMatrix.values[instance]) + bias1
        activation2 = np.dot(weight2, featureTestMatrix.values[instance]) + bias2
        activation3 = np.dot(weight3, featureTestMatrix.values[instance]) + bias3

        confidence_score = max(activation1, activation2, activation3)
        if confidence_score == activation1:
            predictedLabel_test = 'class-1'
        elif confidence_score == activation2:
            predictedLabel_test = 'class-2'
        elif confidence_score == activation3:
            predictedLabel_test = 'class-3'
        predictedTestLabels.append(predictedLabel_test)
        actualTestLabels.append(testLabels.values[instance])

    return predictedTrainLabels, actualTrainLabels, predictedTestLabels, actualTestLabels


def main():

    reg_coeff = 0
    # looping and removing a class to do classification on the other remaining classes
    for className in ['class-1', 'class-2', 'class-3']:
        if className == 'class-1':
            print('\n\n----------- Classifier for class-2 and class-3 -----------\n')
            processedTrainData = getProcessedClasses(trainData, className)
            weight, bias = binaryPerceptron(processedTrainData, reg_coeff)
            processedTestData = getProcessedClasses(testData, className)
            testPerceptron(weight, bias, processedTestData)
        elif className == 'class-2':
            print('\n\n----------- Classifier for class-1 and class-3 -----------\n')
            processedTrainData = getProcessedClasses(trainData, className)
            weight, bias = binaryPerceptron(processedTrainData, reg_coeff)
            processedTestData = getProcessedClasses(testData, className)
            testPerceptron(weight, bias, processedTestData)
        elif className == 'class-3':
            print('\n\n----------- Classifier for class-1 and class-2 -----------\n')
            processedTrainData = getProcessedClasses(trainData, className)
            weight, bias = binaryPerceptron(processedTrainData, reg_coeff)
            processedTestData = getProcessedClasses(testData, className)
            testPerceptron(weight, bias, processedTestData)

    # applying the 1-vs-rest approach to classify data having more than two classes 
    print('\n\n\n-------------------------------1-vs-rest Approach-------------------------------\n')
    trainPredictions, trainActuals, testPredictions, testActuals = multiclassClassification(reg_coeff)
    multiTrainAccuracy = calculatingAccuracies(trainPredictions, trainActuals)
    print('\nAccuracy for multiclass classification train data: ', multiTrainAccuracy)
    multiTestAccuracy = calculatingAccuracies(testPredictions, testActuals)
    print('\nAccuracy for multiclass classification test data: ', multiTestAccuracy)

    # adding an L2 regularisation term to the model 
    print('\n\n\n-----------------------Classification using L2 Regularisation--------------------------\n')
    reg_list = [0.01, 0.1, 1.0, 10.0, 100.0]
    # we take each regularisation coefficient and get different results after using them in the prediction models
    for reg_coeff in reg_list:
        print(f'--------- Results for Multi-class Classification (reg_coeff = {reg_coeff}) ---------\n')
        trainPredictions, trainActuals, testPredictions, testActuals = multiclassClassification(reg_coeff)
        multiTrainAccuracy = calculatingAccuracies(trainPredictions, trainActuals)
        print('Accuracy for multiclass classification train data: ', multiTrainAccuracy)
        multiTestAccuracy = calculatingAccuracies(testPredictions, testActuals)
        print('Accuracy for multiclass classification test data: ', multiTestAccuracy, '\n')


main()




