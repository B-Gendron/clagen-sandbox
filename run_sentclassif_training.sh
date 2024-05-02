#!/bin/bash

cd code

MODEL_NAME='google-bert/bert-base-uncased' # <-- CHANGE THE ANNOTATOR MODEL HERE
DATA_DIR="../data/${MODEL_NAME}_tokenized"
MODEL_FILE="../models/${MODEL_NAME}_best.pt"

# potentially create the required directories to save data/models
mkdir -p $(dirname $MODEL_FILE)
mkdir -p $(dirname $DATA_DIR)

# if the data has not been preprocessed for this model, then process it
if [ ! -d "$DATA_DIR" ]; then
    echo $DATA_DIR does not exist
    echo "Training on $MODEL_NAME requires preprocessing."
    python3 preproc_sentclassif.py -m $MODEL_NAME
else
    echo "Preprocessed data for $MODEL_NAME already exists."
fi

# if the annotator model has not been fine-tuned yet, then fine-tune it
if [ ! -f "$MODEL_FILE" ]; then
    echo "This annotator model needs to be fined-tuned. Starts fine-tuning..."
    python3 train_sentclassif.py -m "$MODEL_NAME"
else
    echo "Fined-tuned annotator model already exists. Skipping this step."
fi

# now that everything is ready, perform classification/generation fine-tuning
python3 finetune.py