import cv2
import csv
import numpy as np

samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)
    for line in reader:
        samples.append(line)
       
    
name = './data/IMG/'+samples[0][0].split('/')[-1]
center_image = cv2.imread(name)
print(np.max(center_image))


