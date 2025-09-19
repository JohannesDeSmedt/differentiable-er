import pandas as pd
import numpy as np

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
from preprocessing import encode_activities, extract_daily_prefixes_with_shifted_targets, extract_prefix_suffix_pairs, PaddedLabelEncoder
from FullTransformerSDFA import SDFA_model, SDFA_suffix_model, SDFA_NAP_model, NAP_model, suffix_model
from FullTransformerSDFA import train_model, train_suffix_model, evaluate_model, evaluate_suffix_model, train_NAP_model, evaluate_nap_model


def write_results_to_csv(filename, params, results):
    file_exists = os.path.isfile(filename)
    with open(filename, 'a') as f:
        if not file_exists:
            header = ','.join(params.keys()) + ',' + ','.join(results.keys()) + '\n'
            f.write(header)
        line = ','.join(str(v) for v in params.values()) + ',' + ','.join(str(v) for v in results.values()) + '\n'
        f.write(line)


def add_start_end_rows(group):
    earliest_ts = group['time:timestamp'].min()
    latest_ts = group['time:timestamp'].max()

    start_row = pd.DataFrame([{'case:concept:name': group.name, 'time:timestamp': earliest_ts - pd.Timedelta(seconds=1), 'concept:name': 'Start'}])
    end_row = pd.DataFrame([{'case:concept:name': group.name, 'time:timestamp': latest_ts + pd.Timedelta(seconds=1), 'concept:name': 'End'}])

    return pd.concat([group, start_row, end_row], ignore_index=True)


EVENT_LOGS = {
    "BPI12": BPI12,
    "BPI17": BPI17,
    # "BPI19": BPI19,
    # "BPI20PrepaidTravelCosts": BPI20PrepaidTravelCosts,
    # "BPI20TravelPermitData": BPI20TravelPermitData,
    # "BPI20RequestForPayment": BPI20RequestForPayment,
}

le = PaddedLabelEncoder()

def prepare_data(df: pd.DataFrame, unbiased_split_params: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.loc[:, ["case:concept:name", "concept:name", "time:timestamp"]]
    cases_to_drop = df.groupby("case:concept:name").size() > 5
    cases_to_drop = cases_to_drop[cases_to_drop].index
    df = df[df["case:concept:name"].isin(cases_to_drop)]

    df_with_new_rows = df.groupby("case:concept:name").apply(add_start_end_rows)
    df = df_with_new_rows.reset_index(drop=True)

    df = df.sort_values(by=["case:concept:name", "time:timestamp"])
    train, test = unbiased(df, **unbiased_split_params)


    train = train.rename(columns={"case:concept:name": "case_id", "concept:name": "activity"})
    test = test.rename(columns={"case:concept:name": "case_id", "concept:name": "activity"})
    
    le.fit(train["activity"])

    df_train = encode_activities(train, le)
    # df_train = df_train.sample(2000)
    df_test = encode_activities(test, le)  
    # df_test = df_test.sample(2000)      
    
    # if suffix_prediction or nap_prediction:
    return df_train, df_test
    # else:
    #     train_seqs = extract_daily_prefixes_with_shifted_targets(df_train, prefix_len, le)
    #     test_seqs = extract_daily_prefixes_with_shifted_targets(df_test, prefix_len, le)
    #     return train_seqs, test_seqs
        


log = EVENT_LOGS['BPI12']()

suffix_prediction = True

train_loader, test_loader = prepare_data(log.dataframe, log.unbiased_split_params) 

vocab_size = len(le.classes_) 
max_len = 5  
write = True
no_epochs = 10
seed = 56

torch.manual_seed(seed)
np.random.seed(seed)

for seed in [ 56]:#8, 15, 76, 23]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    for er_loss_use in [True]:#, False]:
        for d_model_p in [8, 16, 32]:
            if suffix_prediction:
                if er_loss_use:
                    for mix_lambda in [0.1, 0.2, 0.5]:
                        print('1')
                        # model = SDFA_model(vocab_size, d_model=d_model_p, sdfa_shape=(vocab_size, vocab_size)).to(FullTransformerSDFA.device)
                        model = SDFA_suffix_model(vocab_size, d_model=d_model_p, sdfa_shape=(vocab_size, vocab_size)).to(FullTransformerSDFA.device)
                        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                        train_suffix_model(model, le, train_loader, optimizer, max_len, er_loss_use, mix_lambda, num_epochs=no_epochs)
                        # er_loss, dl_distance = evaluate_model(model, le, test_loader, max_len)
                        er_loss, dl_distance = evaluate_suffix_model(model, le, test_loader, max_len, er_loss_use)
                        if write:
                            write_results_to_csv('results_suffix_prediction.csv',
                                params={'seed': seed, 'model': model.__class__.__name__, 'lambda': mix_lambda, 'd_model': d_model_p, 'er_loss': er_loss_use, 'no_epochs': no_epochs},
                                results={'avg_dl_distance': dl_distance}
                            )
                else:
                    print('2')
                    model = suffix_model(vocab_size, d_model=d_model_p, sdfa_shape=(vocab_size, vocab_size)).to(FullTransformerSDFA.device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
                    train_suffix_model(model, le, train_loader, optimizer, max_len, er_loss_use, mix_lambda, num_epochs=no_epochs)
                    er_loss, dl_distance = evaluate_suffix_model(model, le, test_loader, max_len, er_loss_use)
                    if write:
                        write_results_to_csv('results_suffix_prediction.csv',
                                params={'seed': seed, 'model': model.__class__.__name__, 'lambda': mix_lambda, 'd_model': d_model_p, 'er_loss': er_loss_use, 'no_epochs': no_epochs},
                                results={'avg_dl_distance': dl_distance}
                        )
            else:
                for max_len in [5, 10, 20]:
                    if er_loss_use:
                        for mix_lambda in [0.1, 0.2, 0.5]:
                            print('3')
                            model = SDFA_NAP_model(vocab_size, d_model=d_model_p, sdfa_shape=(vocab_size, vocab_size)).to(FullTransformerSDFA.device)
                            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
                            train_NAP_model(model, le, train_loader, optimizer, max_len, er_loss_use, mix_lambda, num_epochs=no_epochs)
                            er_loss, accuracy, precision, recall, f1 = evaluate_nap_model(model, le, test_loader, max_len, er_loss_use)
                            if write:
                                write_results_to_csv('results_nap_prediction_new.csv',
                                params={'seed': seed, 'model': model.__class__.__name__, 'lambda': mix_lambda, 'd_model': d_model_p, 'er_loss': er_loss_use, 'no_epochs': no_epochs, 'max_len': max_len},
                                results={'recall': recall, 'precision': precision, 'f1': f1, 'accuracy': accuracy}
                                )
                    else:
                        print('4')
                        model = NAP_model(vocab_size, d_model=d_model_p, sdfa_shape=(vocab_size, vocab_size)).to(FullTransformerSDFA.device)
                        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
                        train_NAP_model(model, le, train_loader, optimizer, max_len, er_loss_use, mix_lambda, num_epochs=no_epochs)
                        er_loss, accuracy, precision, recall, f1  = evaluate_nap_model(model, le, test_loader, max_len, er_loss_use)
                        if write:
                            write_results_to_csv('results_nap_prediction_new.csv',
                                params={'seed': seed, 'model': model.__class__.__name__, 'lambda': mix_lambda, 'd_model': d_model_p, 'er_loss': er_loss_use, 'no_epochs': no_epochs, 'max_len': max_len},
                                results={'recall': recall, 'precision': precision, 'f1': f1, 'accuracy': accuracy}
                            )





