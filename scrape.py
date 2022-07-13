from os import walk
import re

data_dir = "raw_data/DUC2006_Summarization_Documents/duc2006_docs"
article_set = "D0601A"
filenames = next(walk(f"{data_dir}/{article_set}"), (None, None, []))[2]

articles = []

for article in filenames:
    with open(f"{data_dir}/{article_set}/{article}", 'r') as f:
        articles += [f.read()]

paras = re.match("(<P>((\s+.*)+)<\/P>)+", articles[0])

if paras:
    print(paras.groups())

