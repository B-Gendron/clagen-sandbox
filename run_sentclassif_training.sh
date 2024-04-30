#!/bin/bash
cd code

MODEL_NAME='google-bert/bert-base-uncased'
DATA_DIR="../data/${MODEL_NAME}_tokenized/train"

# if the data has not been preprocessed for this model, then process it
if [ ! -d "$DATA_DIR" ]; then
    echo $DATA_DIR does not exist
    echo "Training on $MODEL_NAME requires preprocessing."
    python3 preproc_sentclassif.py -m $MODEL_NAME
else
    echo "Preprocessed data for $MODEL_NAME already exists."
fi

# train PLM on sentiment analysis
python3 train_sentclassif.py -m $MODEL_NAME