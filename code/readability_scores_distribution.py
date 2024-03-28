import pandas as pd
import lftk # for readability scores
import spacy
import matplotlib.pyplot as plt
from tqdm import tqdm
import swifter # to parallelize apply on all available cores

NLP = spacy.load('en_core_web_sm')
tqdm.pandas()

def calculate_scores(sentence):
    # extract scores
    doc = NLP(sentence)
    LFTK = lftk.Extractor(docs=doc)
    readability_metrics = LFTK.extract(features=["fkgl", "fogi", "cole", "smog", "auto"])

    return readability_metrics


def update_scores(row):
    scores = calculate_scores(row['sentence'])
    for col, val in scores.items():
        row[col] = val
    return row


def display_scores_distributions(df):
    # Define the columns to display
    columns_to_display = ['fkgl', 'fogi', 'cole', 'smog', 'auto']
    
    # Create subplots for each column
    fig, axs = plt.subplots(1, len(columns_to_display), figsize=(15, 5))
    
    # Plot each column's distribution
    for i, col in enumerate(columns_to_display):
        ax = axs[i]
        ax.hist(df[col], bins=20, alpha=0.7)
        ax.set_title(col)
        ax.set_xlabel('Value Range')
        ax.set_ylabel('Frequency')

    plt.savefig('../objects/readability_scores_ditribution.png', dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()


if __name__=='__main__':

    # read txt file
    with open('../bookreviews/data.txt', 'r', encoding='utf-8') as file:
        text = file.read()

    # split sentences at dot and make a df out of it
    sentences = text.split('.')
    df = pd.DataFrame({'sentence': sentences})
    df = pd.DataFrame({'sentence': [sentence.strip() + '.' for sentence in sentences if sentence.strip()]})

    # remove empty entries and reset index
    df = df[df['sentence'] != '']
    df.reset_index(drop=True, inplace=True)

    # initialize columns for readability scores

    df['fkgl'] = 0
    df['fogi'] = 0
    df['cole'] = 0
    df['smog'] = 0
    df['auto'] = 0

    # apply mapping + display distribution
    df = df.swifter.apply(update_scores, axis=1)
    df.to_csv('../objects/readability_scores_dataframe.csv', index=False)
    display_scores_distributions(df)