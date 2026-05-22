import os
import time
import torch
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import BertTokenizer, BertForSequenceClassification, DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments

from parameters import params

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    return {'accuracy': acc, 'f1': f1}

def training():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    
    train_df = pd.read_csv(os.path.join(params['final_dir'], 'train.csv'))
    test_df = pd.read_csv(os.path.join(params['final_dir'], 'test.csv'))

    # tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def tokenize_data(df):
        encodings = tokenizer(
            df['URL'].tolist(), 
            truncation=True, 
            padding='max_length', 
            max_length=params['max_length']
        )
        dataset = Dataset.from_dict(encodings)
        dataset = dataset.add_column("labels", df['label'].tolist())
        return dataset

    train_dataset = tokenize_data(train_df)
    test_dataset = tokenize_data(test_df)

    # model = DistilBertForSequenceClassification.from_pretrained(
    #     'distilbert-base-uncased', 
    #     num_labels=2,
    #     seq_classif_dropout=params['dropout']
    # )

    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased', 
        num_labels=2,
        hidden_dropout_prob=params['dropout']
    )
    
    model.to(device)
    dummy_input = tokenizer("http://test-phish-link.com", return_tensors="pt").to(device)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=params['epochs'],
        per_device_train_batch_size=params['batch_size'],
        learning_rate=params['learning_rate'],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        weight_decay=0.01,
        logging_dir='./logs'
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    print("Starting training...")
    trainer.train()

    results = trainer.evaluate()
    print(f"\nPerformance Metrics:\nAccuracy: {results['eval_accuracy']:.4f}\nF1: {results['eval_f1']:.4f}")
    
    with torch.no_grad():
        # Warm up
        for _ in range(10):
            model(**dummy_input)
            
        torch.cuda.synchronize()
        start = time.time()
        
        model(**dummy_input)
        
        torch.cuda.synchronize()
        end = time.time()
    
    print(f"\nEfficiency Metrics:\nInference Latency: {(end-start)*1000:.2f} ms")

if __name__ == "__main__":
    training()