import numpy as np
import pandas as pd
import os
import math
import matplotlib.pyplot as plt
import argparse
import json
import datasets
from datasets import Dataset, DatasetDict

import subprocess

if not os.path.isfile("./data/utterances.jsonl"): 
    subprocess.call(['sh', 'download_data.sh'])

utt_dataset = pd.read_json("./data/utterances.jsonl", lines=True)

utt_dataset = utt_dataset.drop(columns=['user', 'meta', 'timestamp'])
print(utt_dataset.head())
utt_dataset['replies'] = utt_dataset['reply-to'].values.astype('int')
utt_dataset = utt_dataset.drop(columns=['reply-to'])
# utt_dataset['reply-to'] = utt_dataset['reply-to'].apply(lambda x: x.astype(int) if not math.isnan(x) else x)
print(utt_dataset.head())
