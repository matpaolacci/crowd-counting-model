import numpy as np
import json
import os

PATH_JSONS = "/path/to/jsons"

def main(directory):
    count = []
    for filename in os.listdir(directory):
        filePath = os.path.join(directory, filename)
        
        # checking if it is a file
        if os.path.isfile(filePath):
            with open(filePath, 'r') as f:
                js = json.load(f)
                count.append(js['human_num'])
    count = np.array(count)
    print("Max: {}".format(count.max()))
    print("Min: {}".format(count.min()))
    print("std: {}".format(count.std()))
    print("Mean: {}".format(count.mean()))
    print("Mean value under the mean: {}".format(count[count<count.mean()].mean()))
    print("Mean value over the mean: {}".format(count[count>count.mean()].mean()))
    print("Number of values over the mean {}".format(count[count>=count.mean()].shape[0]))
    print("Number of values under the mean {}".format(count[count<=count.mean()].shape[0]))

main(PATH_JSONS)