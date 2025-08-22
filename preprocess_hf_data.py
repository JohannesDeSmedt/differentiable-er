import os
import pandas as pd
from datasets import load_dataset

def to_event_log(hf_dataset):
    df = pd.DataFrame()
    for row in hf_dataset:
        tmp = pd.DataFrame(row)
        df = pd.concat([df, tmp], ignore_index=True)
    return df.reset_index(drop=True)
        
ds = load_dataset("easytpp/taobao")
df_train = to_event_log(ds["train"])
df_test = to_event_log(ds["test"])

# os.mkdir("data")#, exist_ok=True)
df_train.to_csv("data/taobao_train.csv", index=False)
df_test.to_csv("data/taobao_test.csv", index=False)