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
from preprocessing import encode_activities, extract_daily_prefixes_with_shifted_targets, extract_prefix_suffix_pairs
from FullTransformerSDFA import SDFA_model, SDFA_suffix_model, SDFA_NAP_model, NAP_model
from FullTransformerSDFA import train_model, train_suffix_model, evaluate_model, evaluate_suffix_model, train_NAP_model, evaluate_nap_model


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
    cases_to_drop = df.groupby("case:concept:name").size() > 5
    cases_to_drop = cases_to_drop[cases_to_drop].index
    df = df[df["case:concept:name"].isin(cases_to_drop)]

    df = df.sort_values(by=["case:concept:name", "time:timestamp"])
    train, test = unbiased(df, **unbiased_split_params)

    train = train.rename(columns={"case:concept:name": "case_id", "concept:name": "activity"})
    test = test.rename(columns={"case:concept:name": "case_id", "concept:name": "activity"})
    
    le.fit(train["activity"])

    df_train = encode_activities(train, le)
    # df_train = df_train.sample(100)
    df_test = encode_activities(test, le)        
    
    if suffix_prediction or nap_prediction:
        return df_train, df_test
    else:
        train_seqs = extract_daily_prefixes_with_shifted_targets(df_train, prefix_len, le)
        test_seqs = extract_daily_prefixes_with_shifted_targets(df_test, prefix_len, le)
        return train_seqs, test_seqs
        


log = EVENT_LOGS['BPI17']()

suffix_prediction = False
nap_prediction = True
er_prediction = False
er_loss = True

train_loader, test_loader = prepare_data(log.dataframe, log.unbiased_split_params) 

vocab_size = len(le.classes_) 
max_len = prefix_len  

if not os.path.exists('results_suffix_prediction_er_loss.csv'):
    write_file = open('results_suffix_prediction_er_loss.csv', 'a')
    write_file.write('lambda, d_model, er_loss, avg_dl_distance\n')
    write_file.close()

if not os.path.exists('results_nap_prediction_er_loss.csv'):
    write_file = open('results_nap_prediction_er_loss.csv', 'a')
    write_file.write('lambda, d_model, er_loss, recall, precision, f1, avg_dl_distance\n')
    write_file.close()

for mix_lambda in [0, 0.1, 0.2, 0.5]:
    for d_model_p in [8, 32, 64]:
        if nap_prediction and er_loss:
            model = SDFA_NAP_model(vocab_size, d_model=d_model_p, sdfa_shape=(vocab_size, vocab_size)).to(FullTransformerSDFA.device)
            print('Dit')
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            train_NAP_model(model, le, train_loader, optimizer, max_len, er_loss, mix_lambda, num_epochs=10)
            er_loss, accuracy, precision, recall, f1 = evaluate_nap_model(model, le, test_loader, max_len, er_loss)
            # write_file = open('results_nap_prediction_combined_er_loss.csv', 'a')
            # write_file.write(f'{mix_lambda}, {d_model_p}, {er_loss}, {recall}, {precision}, {f1}, \n')
            # write_file.close()
        if nap_prediction and not er_loss:
            model = NAP_model(vocab_size, d_model=d_model_p, sdfa_shape=(vocab_size, vocab_size)).to(FullTransformerSDFA.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            train_NAP_model(model, le, train_loader, optimizer, max_len, er_loss, mix_lambda, num_epochs=10)
            er_loss, accuracy, precision, recall, f1  = evaluate_nap_model(model, le, test_loader, max_len, er_loss)
            # write_file = open('results_nap_prediction_no_er_loss.csv', 'a')
            # write_file.write(f'{mix_lambda}, {d_model_p}, {er_loss}, {recall}, {precision}, {f1}, \n')
            # write_file.close()
        if suffix_prediction:
            model = SDFA_model(vocab_size, d_model=d_model_p, sdfa_shape=(vocab_size, vocab_size)).to(FullTransformerSDFA.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            train_model(model, le, train_loader, optimizer, max_len, num_epochs=10)
            evaluated_model = evaluate_model(model, le, test_loader, max_len)
        if er_prediction:
            model = SDFA_suffix_model(vocab_size, d_model=d_model_p, sdfa_shape=(vocab_size, vocab_size)).to(FullTransformerSDFA.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            train_suffix_model(model, le, train_loader, optimizer, max_len, er_loss, mix_lambda, num_epochs=10)
            er_loss, evaluated_model = evaluate_suffix_model(model, le, test_loader, max_len, er_loss)
            print('Done')
            print(er_loss)
            print(evaluated_model)
            # write_file = open('results_suffix_prediction_er_loss.csv', 'a')
            # write_file.write(f'{mix_lambda}, {d_model_p}, {er_loss}, {evaluated_model}\n')
            # write_file.close()



