cd code

MODEL_NAME='google-bert/bert-large-uncased'

python3 preproc_sentclassif.py -m $MODEL_NAME
python3 train_sentclassif.py -m $MODEL_NAME