# Importing the libraries needed
import pandas as pd
import torch
import transformers
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertModel, DistilBertTokenizer

# Setting up the device for GPU usage

from torch import cuda


def main():
    device = 'cuda' if cuda.is_available() else 'cpu'
    print('cuda is avaiable', cuda.is_available() )


    # Import the csv into pandas dataframe and add the headers
    df = pd.read_csv('./data/newsCorpora.csv', sep='\t', names=['ID','TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP'])
    # df.head()
    # # Removing unwanted columns and only leaving title of news and the category which will be the target
    df = df[['TITLE','CATEGORY']]
    # df.head()

    # # Converting the codes to appropriate categories using a dictionary
    my_dict = {
        'e':'Entertainment',
        'b':'Business',
        't':'Science',
        'm':'Health'
    }
    df['CATEGORY'] = df['CATEGORY'].apply(lambda x: update_cat(x))
    encode_dict = {}
    df['ENCODE_CAT'] = df['CATEGORY'].apply(lambda x: encode_cat(x))


    def update_cat(x):
        return my_dict[x]


    def encode_cat(x):
        if x not in encode_dict.keys():
            encode_dict[x]=len(encode_dict)
        return encode_dict[x]
    

    # Defining some key variables that will be used later on in the training
    MAX_LEN = 512
    TRAIN_BATCH_SIZE = 4
    VALID_BATCH_SIZE = 2
    EPOCHS = 1
    LEARNING_RATE = 1e-05
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')



