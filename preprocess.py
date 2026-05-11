import pandas as pd
import numpy as np
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import parameters as params

def main():
    params = params.params
    df_train = pd.read_csv(params['root_dir'])

    print(df_train.columns)