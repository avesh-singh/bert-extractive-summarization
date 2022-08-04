import json
from functools import reduce

with open('2006_rouge.json', 'r') as file:
    scores = json.load(file)

print(reduce(lambda total, key: total + scores[key]['rouge1']['fmeasure'], scores, 0) / len(scores))

