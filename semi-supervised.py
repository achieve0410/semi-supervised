import csv
import math
import numpy as np
from statsmodels.api import OLS



## Empty list for save dataset
header = []
train_data = []
test_data = []

###### Train data #######

## variable for count # of lines
line_counter = 0

## read data from csv file
with open('modify_train_dataset.csv') as f:
    while 1:
        data = f.readline().replace("\n","")
        # print(data)
        if not data: break
        if line_counter == 0:
            header = data.split(",") # 
        else:
            train_data.append(data.split(","))
        line_counter = line_counter + 1

## set train data
x_train = np.zeros( [len(train_data), len(train_data[0])-1] )
y_train = np.zeros( [len(train_data), 1] )


## variables for array
line_counter = 0
compo_counter = 0

## train_data to x_train, y_train data
while 1:
    data = train_data
    if len(data)<=line_counter: break

    if compo_counter == len(data[0])-1:
        y_train[line_counter][0] = int(data[line_counter][compo_counter])
        line_counter += 1
        compo_counter = 0

    else :
        x_train[line_counter][compo_counter] = int(data[line_counter][compo_counter])
        compo_counter += 1



###### Test data #######


# ## reset line_counter variable
line_counter = 0

## read data from csv file
with open('modify_test_dataset.csv') as g:
    while 1:
        data = g.readline().replace("\n","")
        # print(data)
        if not data: break
        if line_counter == 0:
            header = data.split(",") # 
        else:
            test_data.append(data.split(","))
        line_counter = line_counter + 1

## set test data
x_test = np.zeros( [len(test_data), len(test_data[0])] )
y_test = np.zeros( [len(test_data), 1] )


## variables for array
line_counter = 0
compo_counter = 0

## test_data to x_test, y_test data
while 1:
    data = test_data
    if len(data)<=line_counter: break

    if compo_counter == len(data[0])-1:
        x_test[line_counter][compo_counter] = int(data[line_counter][compo_counter])
        line_counter += 1
        compo_counter = 0

    else :
        x_test[line_counter][compo_counter] = int(data[line_counter][compo_counter])
        compo_counter += 1



## output

# print(x_train[-1,:])
# print(y_train[-1,:])
# print(x_test[-1,:])
# print(y_test[-1,:])

# print(len(x_train))         ## 1460
# print(len(x_train[0]))      ## 26
# print(len(y_train))         ## 1460
# print(len(y_train[0]))      ## 1

# print(len(x_test))          ## 1459
# print(len(x_test[0]))       ## 26
# print(len(y_test))          ## 1459
# print(len(y_test[0]))       ## 1


#############################################################################################################################################################################

# pred = np.zeros( [len(test_data), 1] )
# loss = np.zeros( [len(test_data), 1] )

# ## variables for array
# line_counter = 0
# compo_counter = 0

# ## temp_data to train/test data ( list (1 dim.) -> array (2 dim.) )
# while 1:
#     data = temp_data
#     if len(data)<=line_counter: break

#     if compo_counter == 3:
#         y_train[line_counter][0] = data[line_counter][compo_counter]
#         line_counter += 1
#         compo_counter = 0

#     else :
#         x_train[line_counter][compo_counter] = data[line_counter][compo_counter]
#         compo_counter += 1


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