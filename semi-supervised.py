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
    labeled_counter = 0
    unlabeled_counter = 0
    compo_counter = 0

    rowLength = len(trainOrTest)
    colLength = len(trainOrTest[0])

    if(trainOrTest == train_data):
        ## set train data
        # x_train = np.zeros( [rowLength, colLength-1] )
        # y_train = np.zeros( [rowLength, 1] )

        labeled_x = np.zeros( [int(rowLength*0.05), colLength-1] )
        labeled_y = np.zeros( [int(rowLength*0.05), 1] )
        unlabeled_x = np.zeros( [int(rowLength*0.95), colLength-1] )
        unlabeled_y = np.zeros( [int(rowLength*0.95), 1] )

        tempList = list(range(0, rowLength, 1))
        random.shuffle(tempList)
        labeledList = tempList[0:int(rowLength*0.05)]

        while 1:
            data = trainOrTest
            if len(data)<=line_counter: break

            if(checkLineNum(labeledList, int(rowLength*0.05), line_counter)): ## labeled_x, labeled_y
                if compo_counter == len(data[0])-1:
                    # y_train[line_counter][0] = int(data[line_counter][compo_counter])
                    # line_counter += 1
                    # compo_counter = 0
                    labeled_y[labeled_counter][0] = int(data[line_counter][compo_counter])
                    labeled_counter += 1
                    line_counter += 1
                    compo_counter = 0
                else :
                    labeled_x[labeled_counter][compo_counter] = int(data[line_counter][compo_counter])
                    compo_counter += 1

            else: ## unlabeled_x, unlabeled_y
                if compo_counter == len(data[0])-1:
                    unlabeled_y[unlabeled_counter][0] = int(data[line_counter][compo_counter])
                    unlabeled_counter += 1
                    line_counter += 1
                    compo_counter = 0
                else :
                    unlabeled_x[unlabeled_counter][compo_counter] = int(data[line_counter][compo_counter])
                    compo_counter += 1

        return labeled_x, labeled_y, unlabeled_x, unlabeled_y

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


###### Train data #######

train_data = readCSVfile('modify_train_dataset.csv')

labeled_x, labeled_y, unlabeled_x, unlabeled_y = divideData(train_data)

###### Test data #######

test_data = readCSVfile('modify_test_dataset.csv')

x_test, y_test = divideData(test_data)


## output

print(len(labeled_x))         ## 73
print(len(labeled_x[0]))      ## 25
print(len(labeled_y))         ## 73
print(len(labeled_y[0]))      ## 1

print(len(unlabeled_x))       ## 1387
print(len(unlabeled_x[0]))    ## 25
print(len(unlabeled_y))       ## 1387
print(len(unlabeled_y[0]))    ## 1

print(len(x_test))          ## 1459
print(len(x_test[0]))       ## 25
print(len(y_test))          ## 1459
print(len(y_test[0]))       ## 1


#############################################################################################################################################################################

# pred = np.zeros( [len(test_data), 1] )
# loss = np.zeros( [len(test_data), 1] )

# ## reshape datasets
# np.reshape(x_train, (-1, 1))
# np.reshape(y_train, (1, -1))

# ## create and summary model
# model = OLS(y_train, x_train)
# y_pred = model.fit()
# #print(y_pred.summary())

# ## predict the answer
# pred = y_pred.predict(x_train)

# ## calculate RSME
# line_counter = 0
# while 1:
#     if len(data)<=line_counter: break

#     loss[line_counter][0] = abs( y_train[line_counter][0]-int(pred[line_counter]) )
#     line_counter += 1

# RSME = math.sqrt( sum( pow(loss, 2) ) / len(y_train) )
# print("RSME : ", RSME)
