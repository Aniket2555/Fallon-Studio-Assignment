import numpy as np
import pandas as pd
import torch
import re, os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, pipeline
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

def download_nltk_resources():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    except Exception as e:
        print(f"NLTK download warning: {e}")
        print("Continuing with basic preprocessing...")

def preprocess_txt(txt):
    txt = str(txt).lower()
    txt = re.sub(r"http\S+|www\S+|https\S+", "", txt)  
    txt = re.sub(r"@\w+|#\w+", "", txt)                
    txt = re.sub(r"[^\w\s]", "", txt)                  
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

def load_and_encode():
    df = pd.read_csv("Tweets/tweets.csv")
    df = df[["text", "airline_sentiment"]]
    
    lab_to_id = {"negative":0, "neutral":1, "positive":2}
    id_to_lab = {v:k for k,v in lab_to_id.items()}
    
    df["label"] = df["airline_sentiment"].map(lab_to_id)
    return df, lab_to_id, id_to_lab

def preprocess_pipeline(txt):
    txt = preprocess_txt(txt)
    tokens = word_tokenize(txt)
    stop_words = set(stopwords.words("english"))
    
    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
    
    lemm = WordNetLemmatizer()
    tokens = [lemm.lemmatize(w) for w in tokens]
    return " ".join(tokens)

def clean_dataset(df):
    df["cleaned_text"] = df["text"].apply(preprocess_pipeline)
    df = df[df["cleaned_text"].str.len() > 10]
    df = df.dropna()
    print("after cleaning:", df.shape)
    return df

def make_datasets(df, lab_to_id):
    tr_txt, te_txt, tr_y, te_y = train_test_split(
        df["cleaned_text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )
    
    model_nm = "bert-base-uncased"
    tok = AutoTokenizer.from_pretrained(model_nm)
    
    tr_enc = tok(list(tr_txt), padding="max_length", max_length=128)
    te_enc = tok(list(te_txt), padding="max_length", max_length=128)
    
    train_ds = Dataset.from_dict({
        "input_ids": tr_enc["input_ids"],
        "attention_mask": tr_enc["attention_mask"],
        "labels": list(tr_y)
    })
    test_ds = Dataset.from_dict({
        "input_ids": te_enc["input_ids"],
        "attention_mask": te_enc["attention_mask"],
        "labels": list(te_y)
    })
    return tok, train_ds, test_ds, model_nm

def build_model(model_nm, id_to_lab, lab_to_id):
    return AutoModelForSequenceClassification.from_pretrained(
        model_nm, num_labels=3, id2label=id_to_lab, label2id=lab_to_id
    )

def metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    p,r,f1,_ = precision_recall_fscore_support(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}


def train(model, tr_ds, te_ds, tok):
    args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",               
        save_strategy="epoch",             
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=20,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir="./logs"
    )
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tr_ds.shuffle(seed=42).select(range(1000)),  
        eval_dataset=te_ds.shuffle(seed=42).select(range(200)),
        tokenizer=tok,
        compute_metrics=metrics,
    )
    
    print(">>> Training started...")
    trainer.train()
    
    res = trainer.evaluate()
    print("Evaluation Results:", res)
    return trainer



def evaluate(trainer, te_ds):
    preds = trainer.predict(te_ds)
    y_pred = preds.predictions.argmax(-1)
    y_true = preds.label_ids
    
    acc = accuracy_score(y_true, y_pred)
    p,r,f1,_ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    
    print("\nFinal scores:")
    print("acc:", round(acc,4), "prec:", round(p,4), "rec:", round(r,4), "f1:", round(f1,4))
    return acc,p,r,f1

def prompt_demo(model, tok):
    tweet = "Flight delayed again for 3 hours, worst experience ever!"
    clf = pipeline("text-classification", model=model, tokenizer=tok, return_all_scores=True)
    
    prompts = [
        f"Tweet: {tweet}",
        f"Analyze sentiment: {tweet}",
        f"Classify this airline feedback: {tweet}"
    ]
    for i,p in enumerate(prompts,1):
        out = clf(p)
        scores = out[0]
        best = max(scores, key=lambda x: x["score"])
        print(f"Prompt {i}: {best['label']} ({best['score']:.3f})")

def main():
    download_nltk_resources()
    
    df, lab_to_id, id_to_lab = load_and_encode()
    df = clean_dataset(df)
    
    tok, tr_ds, te_ds, model_nm = make_datasets(df, lab_to_id)
    model = build_model(model_nm, id_to_lab, lab_to_id)
    
    trainer = train(model, tr_ds, te_ds, tok)
    evaluate(trainer, te_ds)
    prompt_demo(model, tok)

if __name__ == "__main__":
    main()
