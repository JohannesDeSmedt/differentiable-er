import pandas as pd
import numpy as np

from typing import Tuple

import torch
import torch.nn as nn
import time
import os 
import sys

from sklearn.preprocessing import LabelEncoder
from skpm.event_logs import (
    BPI12,
    BPI17,
    BPI15,
    BPI19,
    BPI20PrepaidTravelCosts,
    BPI20TravelPermitData,
    BPI20RequestForPayment,
)
from skpm.event_logs.split import unbiased
from preprocessing import encode_activities, PaddedLabelEncoder, encode_activities_with_dict
from nap_models import SDFA_NAP_model, NAP_model, train_NAP_model, evaluate_nap_model
from suffix_models import SDFA_suffix_model, SDFA_suffix_model_LSTM, suffix_model, train_suffix_model, evaluate_suffix_model


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

    start_row = pd.DataFrame([{'case:concept:name': group.name, 'time:timestamp': earliest_ts - pd.Timedelta(seconds=1), 'concept:name': 'SOC'}])
    end_row = pd.DataFrame([{'case:concept:name': group.name, 'time:timestamp': latest_ts + pd.Timedelta(seconds=1), 'concept:name': 'EOC'}])

    return pd.concat([group, start_row, end_row], ignore_index=True)


EVENT_LOGS = {
    "BPI12": BPI12,
    "BPI17": BPI17,
    "BPI19": BPI19,
    # "BPI19": BPI19,
    "BPI20PrepaidTravelCosts": BPI20PrepaidTravelCosts,
    "BPI20TravelPermitData": BPI20TravelPermitData,
    "BPI20RequestForPayment": BPI20RequestForPayment,
}

le = PaddedLabelEncoder()

def prepare_data(df: pd.DataFrame, unbiased_split_params: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.loc[:, ["case:concept:name", "concept:name", "time:timestamp"]]
    # cases_to_drop = df.groupby("case:concept:name").size() > 5
    # cases_to_drop = cases_to_drop[cases_to_drop].index
    # df = df[df["case:concept:name"].isin(cases_to_drop)]

    df_with_new_rows = df.groupby("case:concept:name").apply(add_start_end_rows)
    df = df_with_new_rows.reset_index(drop=True)

    df = df.sort_values(by=["case:concept:name", "time:timestamp"])
    train, test = unbiased(df, **unbiased_split_params)

    case_lengths = df.groupby("case:concept:name").size()
    max_case_len = int(case_lengths.quantile(quantile))
    valid_cases = case_lengths[case_lengths <= max_case_len].index
    df = df[df["case:concept:name"].isin(valid_cases)]

    print(f'Max case length ({quantile} percentile):', max_case_len)

    train = train.rename(columns={"case:concept:name": "case_id", "concept:name": "activity"})
    test = test.rename(columns={"case:concept:name": "case_id", "concept:name": "activity"})
    
    le.fit(train["activity"])

    # my_little_dict = {label: int(le.transform([label])[0]) for label in le.classes_}
    # print('Activity to index mapping:', my_little_dict)


    df_train = encode_activities(train, le)
    # df_train = df_train.sample(10000)
    df_test = encode_activities(test, le)  
    # df_test = df_test.sample(10000)      
    # df_train = encode_activities_with_dict(train, xiaomeng_dict)
    # df_test = encode_activities_with_dict(test, xiaomeng_dict)

    # df_train['log_ts_pre'] = 0
    # df_train['trace_ts_pre'] = 0
    # df_train['trace_ts_start'] = 0
    # df_test['log_ts_pre'] = 0
    # df_test['trace_ts_pre'] = 0
    # df_test['trace_ts_start'] = 0
    
    # print(df_train.head())

    # df_train.index = range(0, len(df_train))
    # df_test.index = range(len(df_train), len(df_train) + len(df_test))
    # df_train['event_idx'] = range(1, len(df_train) + 1)
    # df_test['event_idx'] = range(len(df_train) + 1, len(df_train) + len(df_test) + 1)
    # print(df_train)

    # df_train.to_csv('df_train_joh.csv', index=True)
    # df_test.to_csv('df_test_joh.csv', index=True)


    # load other dataset

    # train = pd.read_csv('df_before_split.csv')
    # test = pd.read_csv('df_after_split.csv')
    # print(train.head())

    # train['concept:name'] = train['concept:name'].map(reverse_dict)
    # test['concept:name'] = test['concept:name'].map(reverse_dict)
    # print(train.head())

    # train = train.rename(columns={"case:concept:name": "case_id", "concept:name": "activity"})
    # test = test.rename(columns={"case:concept:name": "case_id", "concept:name": "activity"})
    # train = train.drop(['event_idx',  'log_ts_pre',  'trace_ts_pre',  'trace_ts_start'], axis=1)
    # test = test.drop(['event_idx',  'log_ts_pre',  'trace_ts_pre',  'trace_ts_start'], axis=1)

    # print(train.head())

    # df_train = train.encode_activities = encode_activities(train, le)
    # df_test = test.encode_activities = encode_activities(test, le)

    # print(train.head())

    return df_train, df_test, max_case_len

        
# arguments = sys.argv
# dataset = arguments[1]
# if arguments[2].lower() == 'true':
#     suffix_prediction = True
# else:
#     suffix_prediction = False
# quantile = float(arguments[3])

suffix_prediction = True
quantile = 0.8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

for dataset in ["BPI19"]: #["BPI20PrepaidTravelCosts", "BPI20RequestForPayment", "BPI20TravelPermitData"]:

    for suffix_prediction in [suffix_prediction]:
        log = EVENT_LOGS[dataset]()
        log_name = dataset

        train_loader, test_loader, max_case_len = prepare_data(log.dataframe, log.unbiased_split_params) 

        vocab_size = len(le.classes_) 
        max_len = 5  
        write = True
        no_epochs = 2
        seed = 56

        torch.manual_seed(seed)
        np.random.seed(seed)

        for seed in [56, 8, 15, 76, 23]:
            torch.manual_seed(seed)
            np.random.seed(seed)
            for local in [False, True]:
                for bs in [16, 32]:#, 256]:
                    for no_epochs in [10]:
                        for er_loss_use in [False, True]:#, False]:
                            for d_model_p in [16, 32, 64]:
                                if suffix_prediction:
                                    if er_loss_use:
                                        for mix_lambda in [0.1, 0.2, 0.5]:
                                            model = SDFA_suffix_model(vocab_size, d_model=d_model_p, sdfa_shape=(vocab_size, vocab_size)).to(device)
                                            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

                                            comp_time = train_suffix_model(model, le, train_loader, optimizer, max_case_len, er_loss_use, mix_lambda, device, local=local, batch_size=bs, num_epochs=no_epochs)
                                            er_loss, dl_distance = evaluate_suffix_model(model, le, test_loader, max_case_len, er_loss_use, device, local, bs)
                                            if write:
                                                write_results_to_csv(f'results_suffix_prediction_{log_name}.csv',
                                                    params={'seed': seed, 'local':local, 'quantile':quantile, 'model': model.__class__.__name__, 'lambda': mix_lambda, 'd_model': d_model_p, 'er_loss': er_loss_use, 'no_epochs': no_epochs},
                                                    results={'avg_dl_distance': dl_distance, 'time': comp_time}
                                                )
                                    else:
                                        if local:
                                            continue
                                        mix_lambda = 0
                                        model = suffix_model(vocab_size, d_model=d_model_p).to(device)
                                        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
                                        comp_time = train_suffix_model(model, le, train_loader, optimizer, max_case_len, er_loss_use, mix_lambda, device, local=local, batch_size=bs, num_epochs=no_epochs)
                                        er_loss, dl_distance = evaluate_suffix_model(model, le, test_loader, max_case_len, er_loss_use, device, local, bs)
                                        if write:
                                            write_results_to_csv(f'results_suffix_prediction_{log_name}.csv',
                                                    params={'seed': seed,  'local':local, 'quantile':quantile, 'model': model.__class__.__name__, 'lambda': mix_lambda, 'd_model': d_model_p, 'er_loss': er_loss_use, 'no_epochs': no_epochs},
                                                    results={'avg_dl_distance': dl_distance, 'time': comp_time}
                                            )
                                else:
                                    for max_len in [10, 20]:
                                        if er_loss_use:
                                            for mix_lambda in [0.1, 0.2, 0.5]:
                                                model = SDFA_NAP_model(vocab_size, d_model=d_model_p, sdfa_shape=(vocab_size, vocab_size)).to(device)
                                                optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
                                                comp_time = train_NAP_model(model, le, train_loader, optimizer, max_len, er_loss_use, mix_lambda, device, local=local, num_epochs=no_epochs, batch_size=bs)
                                                er_loss, accuracy, precision, recall, f1 = evaluate_nap_model(model, le, test_loader, max_len, er_loss_use, device, local)
                                                if write:
                                                    write_results_to_csv(f'results_nap_prediction_{log_name}.csv',
                                                    params={'seed': seed,  'local':local, 'quantile':quantile, 'model': model.__class__.__name__, 'lambda': mix_lambda, 
                                                            'd_model': d_model_p, 'er_loss': er_loss_use, 'batch_size': bs, 'no_epochs': no_epochs, 'max_len': max_len},
                                                    results={'recall': recall, 'precision': precision, 'f1': f1, 'accuracy': accuracy, 'time': comp_time}
                                                    )
                                        else:
                                            if local:
                                                continue
                                            mix_lambda = 0
                                            model = NAP_model(vocab_size, d_model=d_model_p, sdfa_shape=(vocab_size, vocab_size)).to(device)
                                            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
                                            comp_time = train_NAP_model(model, le, train_loader, optimizer, max_len, er_loss_use, mix_lambda, device, local=local, num_epochs=no_epochs, batch_size=bs)
                                            er_loss, accuracy, precision, recall, f1  = evaluate_nap_model(model, le, test_loader, max_len, er_loss_use, device, local)
                                            if write:
                                                write_results_to_csv(f'results_nap_prediction_{log_name}.csv',
                                                    params={'seed': seed, 'local':local, 'quantile':quantile, 'model': model.__class__.__name__, 'lambda': mix_lambda, 'd_model': d_model_p, 'er_loss': er_loss_use,
                                                            'batch_size': bs,  'no_epochs': no_epochs, 'max_len': max_len},
                                                    results={'recall': recall, 'precision': precision, 'f1': f1, 'accuracy': accuracy, 'time': comp_time}
                                            )





