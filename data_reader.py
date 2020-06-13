import csv
import numpy as np

def load_data():

    time_step = []
    sunspots = []

    with open('/data/Sunports.csv') as csvfile:
        reader = csv.reader(csvfile, delimiter = ',')
        next(reader)

        for row in reader:
            sunspots.append(float(row[2]))
            time_step.append(float(row[0]))
    
    return np.array(sunspots), np.array(time_step)



