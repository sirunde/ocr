import os
import sys
import csv

path = 'data/test'
labels = os.path.join(path, 'labels.csv')
with open(labels, 'w', newline='') as csvfile:
    writing = csv.writer(csvfile, quoting=csv.QUOTE_NONE,
                         delimiter= ' ',
                         escapechar=' ',
                         )

    for dir, _, files in os.walk(path):
        print(files)
        for file in files:
            if file == "labels.csv":
                pass
            else:
                writing.writerow([file+",", file[0]])