from csv import reader
import uuid
import os
# open file in read mode
with open('Results_MC1.csv', 'r') as read_obj:
    # pass the file object to reader() to get the reader object
  csv_reader = reader(read_obj)
  # Iterate over each row in the csv using reader object
  for row in csv_reader:
    # row variable is a list that represents a row in csv
    filename = str(uuid.uuid4()) + ".out"
    a_file = open(os.path.join(os.getcwd(),"federico",filename), "w")
    a_file.write('1\n')
    for node in row:
      a_file.write(node + '\n')
    a_file.close()
