# OntoGPT

The code for babyLLM is inspired from [this tutorial](https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing#scrollTo=nql_1ER53oCf). I will elaborate this in very near future, but here is just a very quick way to launch `babyLLM` training.  

- Setup
```bash 
git clone https://github.com/B-Gendron/OntoGPT.git
pip3 install -r requirements.txt
```
- Launch data download if needed, then model training
```bash
cd babyLM/code
python3 train.py
```

Many hyperparameters can be changed from default configuration. In partcular, for faster training it is possible to reduce the number of iterations (=epochs) from 5000, the default value, to 300 by doing:
```bash
python3 train.py -i 300
```
Note that the display is set to be at every 100 epochs.