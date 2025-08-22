import pandas as pd

from typing import Tuple

import torch
import os 

from sklearn.preprocessing import LabelEncoder
from skpm.event_logs import (
    BPI12,
    BPI17,
    BPI19,
    BPI20PrepaidTravelCosts,
    BPI20TravelPermitData,
    BPI20RequestForPayment,
)
from skpm.event_logs.split import unbiased
import FullTransformerSDFA
from preprocessing import encode_activities, extract_daily_prefixes_with_shifted_targets
from FullTransformerSDFA import SDFA_model, train_model, evaluate_model

EVENT_LOGS = {
    "BPI12": BPI12,
    "BPI17": BPI17,
    # "BPI19": BPI19,
    # "BPI20PrepaidTravelCosts": BPI20PrepaidTravelCosts,
    # "BPI20TravelPermitData": BPI20TravelPermitData,
    # "BPI20RequestForPayment": BPI20RequestForPayment,
}

le = LabelEncoder()
prefix_len = 5

def prepare_data(df: pd.DataFrame, unbiased_split_params: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.loc[:, ["case:concept:name", "concept:name", "time:timestamp"]]
    cases_to_drop = df.groupby("case:concept:name").size() > 2
    cases_to_drop = cases_to_drop[cases_to_drop].index
    df = df[df["case:concept:name"].isin(cases_to_drop)]

    df = df.sort_values(by=["case:concept:name", "time:timestamp"])
    train, test = unbiased(df, **unbiased_split_params)

    train = train.rename(columns={"case:concept:name": "case_id", "concept:name": "activity"})
    test = test.rename(columns={"case:concept:name": "case_id", "concept:name": "activity"})
    
    le.fit(train["activity"])

    df_train = encode_activities(train, le)
    df_test = encode_activities(test, le)        
    
    train_seqs = extract_daily_prefixes_with_shifted_targets(df_train, prefix_len, le)
    test_seqs = extract_daily_prefixes_with_shifted_targets(df_test, prefix_len, le)

    return train_seqs, test_seqs


log = EVENT_LOGS['BPI12']()
        
train_loader, test_loader = prepare_data(log.dataframe, log.unbiased_split_params) 

vocab_size = len(le.classes_) 
max_len = prefix_len  

model = SDFA_model(vocab_size, d_model=64, sdfa_shape=(vocab_size, vocab_size)).to(FullTransformerSDFA.device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train_model(model, le, train_loader, optimizer, max_len, num_epochs=10)
evaluated_model = evaluate_model(model, le, test_loader, max_len)