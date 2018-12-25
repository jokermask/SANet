import os
import sys
from sys import argv
import json
#use like: python filepath2jso.py ./data/part_A_final/train_data/images/ A_train.json
if len(sys.argv)!=3:
    print("please enter path and json name")
else:
    paths = os.listdir(sys.argv[1])
    for i in range(len(paths)):
        paths[i] = sys.argv[1]+paths[i]
    json_res = json.dumps(paths)
    with open(sys.argv[2],'w') as fp:
        fp.write(json_res)

