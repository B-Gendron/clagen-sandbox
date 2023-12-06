import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import argparse
import json
import datasets
from datasets import Dataset, DatasetDict

from convokit import Corpus, download

# Wikipedia talk pages corpus
# corpus = Corpus("wiki-corpus")

utt_dataset = pd.read_json("wiki-corpus/utterances.jsonl", lines=True)

utt_dataset = utt_dataset.drop(columns=['user', 'meta', 'timestamp'])
print(utt_dataset.head())