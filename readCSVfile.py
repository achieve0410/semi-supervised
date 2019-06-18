import csv

header = []
csv_data = []

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
            csv_data.append(data.split(","))
        line_counter = line_counter + 1