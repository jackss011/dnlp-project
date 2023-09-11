papers = {}

# add valid papers to data (some papers do not have an abstract)
with open('./data/outputacm.txt', encoding='utf-8') as f:
    index = None
    title = None
    abstract = None
    refs = []

    for line in f.readlines():
        if line.startswith('#*'):
            title = line.replace('#*', '').strip()
        elif line.startswith('#!'):
            abstract = line.replace('#!', '').strip()
        elif line.startswith('#index'):
            index = 'idx' + line.replace('#index', '').strip()
        elif line.startswith('#%'):
            ref_id = 'idx' + line.replace('#%', '').strip()
            refs.append(ref_id)

        elif len(line.strip().replace('\n', '')) == 0:
            if index and title and abstract:
                papers[index] = {"index": index, "title": title, "abstract": abstract, "refs": refs}
            index = None
            title = None
            abstract = None
            refs = []

print("Number of papers:", len(papers.keys()))


# prune invalid paper ids
ref_count = 0
missing_ref_count = 0
valid_ref_count = 0

for paper in papers.values():
    valid_item_refs = []
    for ref_id in paper['refs']:
        ref_count += 1
        if papers.get(ref_id, None):
            valid_ref_count += 1
            valid_item_refs.append(ref_id)
        else:
            missing_ref_count += 1
    paper['refs'] = valid_item_refs

ref_count, missing_ref_count, valid_ref_count

# compute hard negatives
hn_count = 0

for paper in papers.values():
    hard_negatives_ids = set()
    for ref_id in paper['refs']:
        ref_paper = papers[ref_id]
        for hn_id in ref_paper['refs']:
            if hn_id not in paper['refs'] and hn_id != paper['index']:
              hard_negatives_ids.add(hn_id)
    hn_count += len(hard_negatives_ids)
    paper['hard_negatives'] = list(hard_negatives_ids)

print("Hard negative links count:", hn_count)


# create training files
data = {}
metadata = {}

for id, paper in papers.items():
    metadata[id] = {'paper_id':paper['index'], 'title': paper['title'], 'abstract': paper['abstract']}
    if len(paper['refs']) == 0:
        continue

    data[id] = {}
    for ref_id in paper['refs']:
        data[id][ref_id] = {"count": 5}
    for hn_id in paper['hard_negatives']:
        assert(hn_id not in data[id].keys())
        data[id][hn_id] = {"count": 1}

import json
import os

os.makedirs('full', exist_ok=True)

with open('full/data.json', 'w') as f:
    json.dump(data, f, indent=4)

with open('full/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=4)


# create test, train, eval splits
from sklearn.model_selection import train_test_split

ids = list(metadata.keys())
ids_train, ids_test = train_test_split(ids, test_size=0.2, random_state=42)
ids_train, ids_val = train_test_split(ids_train, test_size=0.25, random_state=42)

with open('full/train.txt', 'w') as f:
   f.writelines([x + '\n' for x in ids_train])

with open('full/val.txt', 'w') as f:
   f.writelines([x + '\n' for x in ids_val])

with open('full/test.txt', 'w') as f:
   f.writelines([x + '\n' for x in ids_test])