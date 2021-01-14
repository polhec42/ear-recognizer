import json
import pandas as pd


root_dir = 'data/ears/awe'

readed_csv = pd.read_csv('data/ears/train.csv')

output_file = open('data/ears/train-gender.csv', 'w')

output_file.write("AWE-Full image path,AWE image path,Gender\n")

number = [0]*6

for i in range(0, len(readed_csv)):
    arg1 = readed_csv.iloc[i, 0]
    arg2 = readed_csv.iloc[i, 1]
    
    path = root_dir + "/" + str(arg2).split("/")[0] + "/annotations.json"

    json_file = open(path, 'r')

    person_dict = json.loads(json_file.read())

    #gender = 1
    #if person_dict['gender'] == 'm':
    #    gender = 2

    number[person_dict['ethnicity']-1] += 1

    output_file.write("{},{},{}\n".format(arg1, arg2, person_dict['ethnicity']))

output_file.close()

#maks = max(number)

#for i in range(0,6):
#    number[i] = maks / number[i]
#
for i in range(0,6):
    number[i] /= 750*5

print(number)

readed_csv = pd.read_csv('data/ears/awe-test.csv')

output_file = open('data/ears/test-gender.csv', 'w')

output_file.write("AWE-Full image path,AWE image path,Gender\n")
counter = 0

for i in range(0, len(readed_csv)):
    arg1 = readed_csv.iloc[i, 0]
    arg2 = readed_csv.iloc[i, 1]
    
    path = root_dir + "/" + str(arg2).split("/")[0] + "/annotations.json"

    json_file = open(path, 'r')

    person_dict = json.loads(json_file.read())


    #gender = 1
    #if person_dict['gender'] == 'm':
    #    print(counter)
    #    counter += 1
    #    gender = 2

    output_file.write("{},{},{}\n".format(arg1, arg2, person_dict['ethnicity']))

output_file.close()
