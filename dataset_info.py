import pandas as pd
import numpy as np

from typing import Tuple
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

import pandas as pd
import numpy as np


def _dataset_stats(df: pd.DataFrame, dataset:str, stage: str) -> dict:
    """
    Return a dictionary with the meta‑information you are interested in.
    """
    # Helper to compute the case‑length series once
    case_len = df.groupby("case:concept:name").size()

    return {
        "Dataset": dataset,
        "Stage": stage,
        "# unique activities": df["concept:name"].nunique(),
        "Avg case length": case_len.mean(),
        "Median case length": case_len.median(),
        "Total cases": df["case:concept:name"].nunique(),
        "Max case length": case_len.max(),
    }

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



def prepare_data(df: pd.DataFrame, unbiased_split_params: dict, dataset:str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.loc[:, ["case:concept:name", "concept:name", "time:timestamp"]]



    # print('Initial dataframe:', df.head(), len(df))
    print('Number of activities before processing:', df['concept:name'].nunique())
    print('Average case length before processing:', df.groupby("case:concept:name").size().mean())
    print('Median case length before processing:', df.groupby("case:concept:name").size().median())
    print('Total number cases before processing:', df["case:concept:name"].nunique())
    print('Maximum case length before processing:', df.groupby("case:concept:name").size().max())


    stats.append(_dataset_stats(df, dataset, "Original"))

    df_with_new_rows = df.groupby("case:concept:name").apply(add_start_end_rows)
    df = df_with_new_rows.reset_index(drop=True)
    # print('Dataframe after adding SOC and EOC events:', df.head(), len(df))
    df = (
        df.groupby("case:concept:name", group_keys=False)  
        .apply(keep_first_of_consecutive_repeats)
        .reset_index(drop=True)
    )
    # print('Dataframe after removing consecutive duplicates:', df.head(), len(df))


    print('Number of activities before processing:', df['concept:name'].nunique())
    print('Average case length before processing:', df.groupby("case:concept:name").size().mean())
    print('Median case length before processing:', df.groupby("case:concept:name").size().median())
    print('Total number cases before processing:', df["case:concept:name"].nunique())
    print('Maximum case length before processing:', df.groupby("case:concept:name").size().max())

    df = df.sort_values(by=["case:concept:name", "time:timestamp"])
    train, test = unbiased(df, **unbiased_split_params)

    case_lengths = df.groupby("case:concept:name").size()
    max_case_len = int(case_lengths.quantile(quantile))
    valid_cases = case_lengths[case_lengths <= max_case_len].index
    df = df[df["case:concept:name"].isin(valid_cases)]

    print(f'\nMax case length ({quantile} percentile):', max_case_len)
    print('Number of activities before processing:', df['concept:name'].nunique())
    print('Average case length before processing:', df.groupby("case:concept:name").size().mean())
    print('Median case length before processing:', df.groupby("case:concept:name").size().median())
    print('Total number cases before processing:', df["case:concept:name"].nunique())
    print('Maximum case length before processing:', df.groupby("case:concept:name").size().max())


    df_after_filter = df.copy()
    stats.append(_dataset_stats(df_after_filter, dataset, "After case‑length filtering"))


    # latex_tbl = meta_df.to_latex(
    #     float_format="%.2f",
    #     caption="Meta‑information of the dataset at three stages of preprocessing",
    #     label="tab:dataset_meta",
    #     index=True,
    #     longtable=False,        # set True if you want a longtable
    # )

    return

import ssl

ssl._create_default_https_context = ssl._create_stdlib_context
        

suffix_prediction = True
quantile = 0.95

stats = []

for dataset in ["BPI12", "BPI17", "BPI19", "BPI20PrepaidTravelCosts", "BPI20RequestForPayment", "BPI20TravelPermitData"]:
    print('\n\nProcessing dataset:', dataset)
    log = EVENT_LOGS[dataset]()
    log_name = dataset

    prepare_data(log.dataframe, log.unbiased_split_params, dataset) 
  

meta_df = pd.DataFrame(stats).set_index("Stage")
float_cols = ["Avg case length", "Median case length", "Max case length"]
meta_df[float_cols] = meta_df[float_cols].round(2)

print(meta_df)

meta_df.to_csv('dataset_statistics.csv')

