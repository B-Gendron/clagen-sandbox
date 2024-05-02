# Classification-generation LLMs fine-tuning

## How to use it very quickly

```
git clone https://github.com/B-Gendron/clagen-sandbox.git
cd clagen-sandbox
pip3 install -r requirements.txt
./run_senclassif_training.sh
```

## Some details

The script `./run_senclassif_training.sh` is designed to perform 3 steps, where the first two will be performed only the first time the script is executed on your machine for a given model:

- Preprocess data from Twitter US Airlines Sentiment dataset, using a tokenizer adated to the desired annotator model. Default annotator model is BERT base. This can be changed in the script, at the indicated spot.

- Fine-tune the annotator model on sentiment analysis (binary classification, 0=negative, 1=positive) using the aforementioned dataset. By default, 40 epochs are performed and the best model is saved in a `models/` folder.

- Fine-tune the LLM using the classification/generation strategy, and the annotator model as a provider of gold labels. Of course, once the two previous steps are performed once, running the script for the same annotator model will be equivalent to performing this step only. This is the buggy part 🐞.