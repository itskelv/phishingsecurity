# preprocess.py
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from parameters import params

def preprocessing():
    print("start preprocessing")
    if not os.path.exists(params['final_dir']):
        print("create final directory")
        os.makedirs(params['final_dir'])

    print(f"Loading dataset from {params['root_dir']}...")
    df = pd.read_csv(params['root_dir'])

    # label 0 = Phishing, 1 = Legitimate
    df = df[['URL', 'label']]

    # train test split
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['label']
    )

    train_path = os.path.join(params['final_dir'], 'train.csv')
    test_path = os.path.join(params['final_dir'], 'test.csv')

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Preprocessing complete. Files saved in {params['final_dir']}")
    print(f"Train samples: {len(train_df)} | Test samples: {len(test_df)}")

if __name__ == "__main__":
    preprocessing()