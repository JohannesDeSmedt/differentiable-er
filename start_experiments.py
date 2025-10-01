import pandas as pd
import numpy as np

from typing import Tuple

import torch
import torch.nn as nn
import os 

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
from preprocessing import encode_activities, PaddedLabelEncoder
from nap_models import SDFA_NAP_model, NAP_model, train_NAP_model, evaluate_nap_model
from suffix_models import SDFA_suffix_model, suffix_model, train_suffix_model, evaluate_suffix_model


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
    "BPI19": BPI19,
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
    
    #max_case_len = 70 # df.groupby("case:concept:name").size().max()
    case_lengths = df.groupby("case:concept:name").size()
    max_case_len = int(case_lengths.quantile(0.95))
    valid_cases = case_lengths[case_lengths <= max_case_len].index
    df = df[df["case:concept:name"].isin(valid_cases)]

    print('Max case length (95th percentile):', max_case_len)

    train = train.rename(columns={"case:concept:name": "case_id", "concept:name": "activity"})
    test = test.rename(columns={"case:concept:name": "case_id", "concept:name": "activity"})
    
    le.fit(train["activity"])

    df_train = encode_activities(train, le)
    # df_train = df_train.sample(10000)
    df_test = encode_activities(test, le)  
    # df_test = df_test.sample(10000)      
    
    # if suffix_prediction or nap_prediction:
    return df_train, df_test, max_case_len
    # else:
    #     train_seqs = extract_daily_prefixes_with_shifted_targets(df_train, prefix_len, le)
    #     test_seqs = extract_daily_prefixes_with_shifted_targets(df_test, prefix_len, le)
    #     return train_seqs, test_seqs
        
device = torch.device("cpu" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

log = EVENT_LOGS['BPI19']()
log_name = 'BPI19'

suffix_prediction = False

train_loader, test_loader, max_case_len = prepare_data(log.dataframe, log.unbiased_split_params) 

vocab_size = len(le.classes_) 
max_len = 5  
write = True
no_epochs = 10
seed = 56

torch.manual_seed(seed)
np.random.seed(seed)

for seed in [ 56, 8, 15, 76, 23]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    for no_epochs in [10]:
        for er_loss_use in [True, False]:#, False]:
            for d_model_p in [8, 16, 32, 64]:
                if suffix_prediction:
                    if er_loss_use:
                        for mix_lambda in [0, 0.1, 0.2]:#, 0.5]:
                            print('1')
                            print('Max case len is ', max_case_len)
                            # model = SDFA_model(vocab_size, d_model=d_model_p, sdfa_shape=(vocab_size, vocab_size)).to(FullTransformerSDFA.device)
                            model = SDFA_suffix_model(vocab_size, d_model=d_model_p, sdfa_shape=(vocab_size, vocab_size)).to(device)
                            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

                            train_suffix_model(model, le, train_loader, optimizer, max_case_len, er_loss_use, mix_lambda, device, num_epochs=no_epochs)
                            # er_loss, dl_distance = evaluate_model(model, le, test_loader, max_len)
                            er_loss, dl_distance = evaluate_suffix_model(model, le, test_loader, max_case_len, er_loss_use, device)
                            if write:
                                write_results_to_csv('results_suffix_prediction.csv',
                                    params={'seed': seed, 'model': model.__class__.__name__, 'lambda': mix_lambda, 'd_model': d_model_p, 'er_loss': er_loss_use, 'no_epochs': no_epochs},
                                    results={'avg_dl_distance': dl_distance}
                                )
                    else:
                        print('2')
                        model = suffix_model(vocab_size, d_model=d_model_p, sdfa_shape=(vocab_size, vocab_size)).to(device)
                        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
                        train_suffix_model(model, le, train_loader, optimizer, max_len, er_loss_use, mix_lambda, num_epochs=no_epochs)
                        er_loss, dl_distance = evaluate_suffix_model(model, le, test_loader, max_len, er_loss_use)
                        if write:
                            write_results_to_csv('results_suffix_prediction.csv',
                                    params={'seed': seed, 'model': model.__class__.__name__, 'lambda': mix_lambda, 'd_model': d_model_p, 'er_loss': er_loss_use, 'no_epochs': no_epochs},
                                    results={'avg_dl_distance': dl_distance}
                            )
                else:
                    for max_len in [5, 10, 20, 30]:
                        if er_loss_use:
                            for mix_lambda in [0.1, 0.2, 0.5]:
                                print('3')
                                model = SDFA_NAP_model(vocab_size, d_model=d_model_p, sdfa_shape=(vocab_size, vocab_size)).to(device)
                                optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
                                comp_time = train_NAP_model(model, le, train_loader, optimizer, max_len, er_loss_use, mix_lambda, device, num_epochs=no_epochs)
                                er_loss, accuracy, precision, recall, f1 = evaluate_nap_model(model, le, test_loader, max_len, er_loss_use, device)
                                if write:
                                    write_results_to_csv(f'results_nap_prediction_{log_name}.csv',
                                    params={'seed': seed, 'model': model.__class__.__name__, 'lambda': mix_lambda, 'd_model': d_model_p, 'er_loss': er_loss_use, 'no_epochs': no_epochs, 'max_len': max_len},
                                    results={'recall': recall, 'precision': precision, 'f1': f1, 'accuracy': accuracy, 'time': comp_time}
                                    )
                        else:
                            mix_lambda = 0
                            print('4')
                            model = NAP_model(vocab_size, d_model=d_model_p, sdfa_shape=(vocab_size, vocab_size)).to(device)
                            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
                            comp_time = train_NAP_model(model, le, train_loader, optimizer, max_len, er_loss_use, mix_lambda, device, num_epochs=no_epochs)
                            er_loss, accuracy, precision, recall, f1  = evaluate_nap_model(model, le, test_loader, max_len, er_loss_use, device)
                            if write:
                                write_results_to_csv(f'results_nap_prediction_{log_name}.csv',
                                    params={'seed': seed, 'model': model.__class__.__name__, 'lambda': mix_lambda, 'd_model': d_model_p, 'er_loss': er_loss_use, 'no_epochs': no_epochs, 'max_len': max_len},
                                    results={'recall': recall, 'precision': precision, 'f1': f1, 'accuracy': accuracy, 'time': comp_time}
                                )





