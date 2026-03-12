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
from suffix_models import suffix_dfg_model, suffix_automaton_model


def keep_first_of_consecutive_repeats(g: pd.DataFrame) -> pd.DataFrame:
    # `shift()` looks at the previous row inside the group
    return g.loc[g["concept:name"].ne(g["concept:name"].shift())]


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
    "BPI20PrepaidTravelCosts": BPI20PrepaidTravelCosts,
    "BPI20TravelPermitData": BPI20TravelPermitData,
    "BPI20RequestForPayment": BPI20RequestForPayment,
}

le = PaddedLabelEncoder()

def prepare_data(df: pd.DataFrame, unbiased_split_params: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.loc[:, ["case:concept:name", "concept:name", "time:timestamp"]]

    df_with_new_rows = df.groupby("case:concept:name").apply(add_start_end_rows)
    df = df_with_new_rows.reset_index(drop=True)
    df = (
        df.groupby("case:concept:name", group_keys=False)  
        .apply(keep_first_of_consecutive_repeats)
        .reset_index(drop=True)
    )

    case_lengths = df.groupby("case:concept:name").size()
    max_case_len = int(case_lengths.quantile(quantile))
    valid_cases = case_lengths[case_lengths <= max_case_len].index
    df = df[df["case:concept:name"].isin(valid_cases)]
    print(f'Max case length ({quantile} percentile):', max_case_len)

    df = df.sort_values(by=["case:concept:name", "time:timestamp"])
    train, test = unbiased(df, **unbiased_split_params)

    train = train.rename(columns={"case:concept:name": "case_id", "concept:name": "activity"})
    max_len = train.groupby("case_id").size().max() - 1
    test = test.rename(columns={"case:concept:name": "case_id", "concept:name": "activity"})
    
    le.fit(train["activity"])

    df_train = encode_activities(train, le)
    # df_train = df_train.sample(10000)
    df_test = encode_activities(test, le)  
    # df_test = df_test.sample(10000)      

    return df_train, df_test, max_len

import ssl

# ssl._create_default_https_context = ssl._create_stdlib_context
        
arguments = sys.argv
dataset = arguments[1]
# if arguments[2].lower() == 'true':
#     suffix_prediction = True
# else:
#     suffix_prediction = False
# seed = int(arguments[3])
# quantile = float(arguments[3])

quantile = 0.95
write = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# dataset = 'BPI12'

for dataset in [dataset]: #'BPI20PrepaidTravelCosts', "BPI20RequestForPayment", "BPI20TravelPermitData"]:
    log = EVENT_LOGS[dataset]()
    log_name = dataset

    train_loader, test_loader, max_len = prepare_data(log.dataframe, log.unbiased_split_params) 
    vocab_size = len(le.classes_) 

    for seed in [56, 8, 15, 76, 23]: #[42, 4, 108, 16, 1089]:#56, 8, 15, 76, 23, 42, 4, 108, 16, 1089]:
        torch.manual_seed(seed)
        np.random.seed(seed)
        for bs in [32]: #, 32, 64, 128]:
            for no_epochs in [20]:
                for er_loss_use in [False]:
                    for d_model_p in [32, 64]:

                        mix_lambda = 0
                        for type_model in [ suffix_model, suffix_dfg_model, suffix_automaton_model]: 
                            model = type_model(vocab_size, d_model=d_model_p).to(device)
                            print(model)
                            # model = suffix_model(vocab_size, d_model=d_model_p).to(device)
                            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
                            comp_time = train_suffix_model(log_name, model, le, train_loader, optimizer, er_loss_use, mix_lambda, device, local=False, batch_size=bs, num_epochs=no_epochs)
                            er_loss, dl_distance, _, _ = evaluate_suffix_model(model, le, test_loader, er_loss_use, device, False, bs)
                            if write:
                                write_results_to_csv(f'results_ns_suffix_prediction_ND_{log_name}.csv',
                                        params={'seed': seed,  'local':False, 'quantile':quantile, 'model': model.__class__.__name__, 'lambda': mix_lambda, 'd_model': d_model_p, 'er_loss': er_loss_use,'batch_size': bs, 'no_epochs': no_epochs},
                                        results={'avg_dl_distance': dl_distance, 'time': comp_time, 'er_pred': 0, 'er_target': 0}
                                )
                        