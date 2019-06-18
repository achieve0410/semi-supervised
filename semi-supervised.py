import csv
import math
import numpy as np
import random
from statsmodels.api import OLS


def readCSVfile(filename):
    header = []
    csv_data = []

    ## reset line_counter variable
    line_counter = 0

    ## read data from csv file
    with open(filename) as g:
        while 1:
            data = g.readline().replace("\n","")
            # print(data)
            if not data: break
            if line_counter == 0:
                header = data.split(",") #
            else:
                csv_data.append(data.split(","))
            line_counter = line_counter + 1

    return csv_data

def checkLineNum(list, limitNum, findNum):
    i=0
    while 1:
        if(i>=limitNum): break
        if(list[i]==findNum):
            return True
        i+=1
    return False

def divideData(trainOrTest):
    ## variables for array
    line_counter = 0
    compo_counter = 0

    rowLength = len(trainOrTest)
    colLength = len(trainOrTest[0])

    ## handling train data
    if(trainOrTest == train_data):        
        labeled_counter = 0
        unlabeled_counter = 0
        labeled_ratio = 0.05

        labeled_x = np.zeros( [int(rowLength*labeled_ratio), colLength-1] )
        labeled_y = np.zeros( [int(rowLength*labeled_ratio), 1] )
        unlabeled_x = np.zeros( [int(rowLength*(1-labeled_ratio)), colLength-1] )
        unlabeled_y = np.zeros( [int(rowLength*(1-labeled_ratio)), 1] )

        tempList = list(range(0, rowLength, 1))
        random.shuffle(tempList)
        labeledList = tempList[0:int(rowLength*labeled_ratio)]

        while 1:
            data = trainOrTest
            if len(data)<=line_counter: break

            ## labeled_x, labeled_y
            if(checkLineNum(labeledList, int(rowLength*labeled_ratio), line_counter)):
                if compo_counter == len(data[0])-1:
                    labeled_y[labeled_counter][0] = int(data[line_counter][compo_counter])
                    labeled_counter += 1
                    line_counter += 1
                    compo_counter = 0
                else :
                    labeled_x[labeled_counter][compo_counter] = int(data[line_counter][compo_counter])
                    compo_counter += 1

            ## unlabeled_x, unlabeled_y
            else:
                if compo_counter == len(data[0])-1:
                    unlabeled_y[unlabeled_counter][0] = int(data[line_counter][compo_counter])
                    unlabeled_counter += 1
                    line_counter += 1
                    compo_counter = 0
                else :
                    unlabeled_x[unlabeled_counter][compo_counter] = int(data[line_counter][compo_counter])
                    compo_counter += 1

        return labeled_x, labeled_y, unlabeled_x, unlabeled_y

    ## handling test data
    else:
        ## set test data
        x_test = np.zeros( [rowLength, colLength] )
        y_test = np.zeros( [rowLength, 1] )

        while 1:
            data = trainOrTest
            if len(data)<=line_counter: break

            if compo_counter == len(data[0])-1:
                x_test[line_counter][compo_counter] = int(data[line_counter][compo_counter])
                line_counter += 1
                compo_counter = 0
            else :
                x_test[line_counter][compo_counter] = int(data[line_counter][compo_counter])
                compo_counter += 1

        return x_test, y_test

def calcError(groundTruth, predictValue):
    se = 0
    i=0
    while 1:
        if(i>=len(groundTruth)): break
        se += math.sqrt(pow(groundTruth[i]-predictValue[i], 2))
        i+=1
    return se/len(groundTruth)

def normalization(data):
    column_counter = 0

    while 1:
        if(column_counter>=len(data[0])): break

        if (int(data.max(axis=0)[column_counter]) != 0):
            data[:,column_counter] = data[:,column_counter] / data.max(axis=0)[column_counter]
        else:
            data[:,column_counter] = data[:,column_counter] / 0.001
        column_counter += 1

    return  data

def makearray(array):
    retArray = np.zeros( [len(array), 1] )

    i=0
    while 1:
        if(i>=len(array)): break
        retArray[i,:]=array[i]
        i+=1
    return retArray

###### Train data #######

train_data = readCSVfile('modify_train_dataset.csv')

labeled_x, labeled_y, unlabeled_x, unlabeled_y = divideData(train_data)

###### Test data #######

test_data = readCSVfile('modify_test_dataset.csv')

x_test, y_test = divideData(test_data)


## output ##

# print(len(labeled_x))         ## 73
# print(len(labeled_x[0]))      ## 25
# print(len(labeled_y))         ## 73
# print(len(labeled_y[0]))      ## 1

# print(len(unlabeled_x))       ## 1387
# print(len(unlabeled_x[0]))    ## 25
# print(len(unlabeled_y))       ## 1387
# print(len(unlabeled_y[0]))    ## 1

# print(len(x_test))            ## 1459
# print(len(x_test[0]))         ## 25
# print(len(y_test))            ## 1459
# print(len(y_test[0]))         ## 1


#############################################################################################################################################################################

## normalization
norm_labeled_x = normalization(labeled_x)

## create and summary model
bModel = OLS(labeled_y, norm_labeled_x)
bPrediction = bModel.fit()
# print(bPrediction.summary())

## predict the answer
bpred = bPrediction.predict(unlabeled_x)
bpred_y = makearray(bpred)

## merge dataset
new_labeled_x = np.vstack((labeled_x, unlabeled_x))
new_labeled_y = np.vstack((labeled_y, bpred_y))



## debug ##

# print(labeled_x.shape)          ## 73, 25
# print((unlabeled_x.shape))      ## 1387, 25
# print((labeled_y.shape))        ## 73, 1
# print((bpred_y.shape))           ## 1387, 1

# print(new_labeled_x.shape)      ## 1460, 25
# print(new_labeled_y.shape)      ## 1460, 1

###########


## normalization
norm_new_labeled_x = normalization(new_labeled_x)

## create and summary new model
aModel = OLS(new_labeled_y, norm_new_labeled_x)
aPrediction = aModel.fit()
# print(aPrediction.summary())

## predict the answer using two models and compare result
apred = aPrediction.predict(x_test)
apred_y = makearray(apred)

cpred = bPrediction.predict(x_test)
cpred_y = makearray(cpred)

# print(apred_y)
# print(cpred_y)
## print error
# error = calcError(apred_y, cpred_y)
# print(error)

