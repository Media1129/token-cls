import json
import csv
import random
import copy


# input file path
test_slot_file = "./data_tag/test.json"
test_result_file = "result_output_v2/predictions.txt"

# output file path
test_analyze_file = "./analyze_result.txt"

with open(test_slot_file, 'r') as f:
    test_slot = json.load(f)

test_result = []
with open(test_result_file, 'r') as f:
    for line in f:
        test_result.append(line)


        
# save augment sentence
with open(test_analyze_file, 'w') as f:
    for idx, sent in enumerate(test_slot):
        f.write(' '.join(sent['tokens'])+'\n')
        f.write('ground truth: '+' '.join(sent['tags'])+'\n')
        f.write(test_result[idx]+'\n')
        

