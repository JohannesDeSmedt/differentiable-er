# --------------------------------------------------------------------------- #
# 1. Imports
# --------------------------------------------------------------------------- #
import os
from pathlib import Path
from typing import Dict, Tuple, Iterable, List, Type

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# Local helpers (make sure the modules are in your PYTHONPATH)
from skpm.event_logs import BPI12, BPI17, BPI19, BPI20PrepaidTravelCosts, \
    BPI20TravelPermitData, BPI20RequestForPayment
from skpm.event_logs.split import unbiased
from preprocessing import encode_activities, PaddedLabelEncoder
from suffix_models import (
    SDFA_suffix_model, suffix_model, train_suffix_model, evaluate_suffix_model
)
from nap_models import (
    SDFA_NAP_model, NAP_model, train_NAP_model, evaluate_nap_model
)
# --------------------------------------------------------------------------- #
# 2. Utility helpers
# --------------------------------------------------------------------------- #
def write_results_to_csv(path: Path, params: Dict, results: Dict) -> None:
    """
    Append a row to *path* (CSV).  If the file does not exist, a header is written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.is_file()
    with path.open("a", newline="") as f:
        if not file_exists:
            header = ",".join(params.keys()) + "," + ",".join(results.keys()) + "\n"
            f.write(header)
        row = ",".join(str(v) for v in params.values()) + "," + ",".join(str(v) for v in results.values()) + "\n"
        f.write(row)


def add_start_end_rows(group: pd.DataFrame) -> pd.DataFrame:
    """
    Add synthetic 'Start' and 'End' events to a case.
    """
    start_ts = group["time:timestamp"].min() - pd.Timedelta(seconds=1)
    end_ts = group["time:timestamp"].max() + pd.Timedelta(seconds=1)
    start_row = pd.DataFrame(
        {"case:concept:name": group.name, "time:timestamp": start_ts, "concept:name": "Start"}
    )
    end_row = pd.DataFrame(
        {"case:concept:name": group.name, "time:timestamp": end_ts, "concept:name": "End"}
    )
    return pd.concat([group, start_row, end_row], ignore_index=True)


# --------------------------------------------------------------------------- #
# 3. Data preparation
# --------------------------------------------------------------------------- #
def prepare_data(
    df: pd.DataFrame,
    unbiased_split_params: Dict,
    le: PaddedLabelEncoder,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return the train / test *encoded* dataframes.
    """
    df = df[["case:concept:name", "concept:name", "time:timestamp"]].copy()

    # Keep only cases with at most 5 events (the original logic kept >5 which seems wrong)
    keep_cases = df.groupby("case:concept:name").size() <= 5
    df = df[df["case:concept:name"].isin(keep_cases[keep_cases].index)]

    # Add synthetic start / end events
    df = df.groupby("case:concept:name", group_keys=False).apply(add_start_end_rows)

    # Sort and split
    df = df.sort_values(["case:concept:name", "time:timestamp"])
    train, test = unbiased(df, **unbiased_split_params)

    # Rename columns to the names used by the models
    train = train.rename(columns={"case:concept:name": "case_id", "concept:name": "activity"})
    test = test.rename(columns={"case:concept:name": "case_id", "concept:name": "activity"})

    # Fit encoder on the training activities and encode both sets
    le.fit(train["activity"])
    train_enc = encode_activities(train, le)
    test_enc = encode_activities(test, le)

    return train_enc, test_enc


# --------------------------------------------------------------------------- #
# 4. Generic training wrapper
# --------------------------------------------------------------------------- #
def run_experiment(
    *,
    model_cls: Type,
    train_loader,
    test_loader,
    le: LabelEncoder,
    vocab_size: int,
    d_model: int,
    sdfa_shape: Tuple[int, int],
    epochs: int,
    max_len: int,
    er_loss_use: bool,
    mix_lambda: float,
    device: torch.device,
    write_csv: Path | None,
    csv_params: Dict,
    eval_fn,  # evaluation function specific to the model type
) -> None:
    """
    Generic training / evaluation loop.
    """
    # Instantiate the model
    model = model_cls(vocab_size, d_model=d_model, sdfa_shape=sdfa_shape).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3 if er_loss_use else 1e-4)

    # Train
    if model_cls in {SDFA_suffix_model, suffix_model}:
        train_suffix_model(
            model,
            le,
            train_loader,
            optimizer,
            max_len,
            er_loss_use,
            mix_lambda,
            device,
            num_epochs=epochs,
        )
    else:  # NAP models
        train_NAP_model(
            model,
            le,
            train_loader,
            optimizer,
            max_len,
            er_loss_use,
            mix_lambda,
            device,
            num_epochs=epochs,
        )

    # Evaluate
    metrics = eval_fn(
        model, le, test_loader, max_len, er_loss_use, device
    )

    # Log to CSV
    if write_csv:
        write_results_to_csv(write_csv, csv_params, metrics)


# --------------------------------------------------------------------------- #
# 5. Main experiment loop
# --------------------------------------------------------------------------- #
def main() -> None:
    # ---- Config ---- #
    event_logs = {"BPI12": BPI12, "BPI17": BPI17, "BPI19": BPI19}
    log_name = "BPI12"           # change if you want another dataset
    suffix_prediction = True     # switch between suffix / NAP

    seed_list = [56]  # Add more seeds if you want
    er_loss_use_list = [True]  # switch off if you want
    d_model_list = [8]#, 16, 32]
    mix_lambda_list = [0.1]#, 0.2, 0.5] if suffix_prediction else [0.0]
    max_len_list = [5]#, 10, 20] if not suffix_prediction else [5]  # only 5 for suffix
    epochs = 10
    write_csv = None #Path("results") / ("results_suffix_prediction.csv" if suffix_prediction else "results_nap_prediction_new.csv")

    # ---- Setup device & seed ---- #
    device = torch.device("cpu" if not torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- Load dataset ---- #
    log = event_logs[log_name]()
    le = PaddedLabelEncoder()

    # ---- Prepare data once (train/test split) ---- #
    train_df, test_df = prepare_data(log.dataframe, log.unbiased_split_params, le)

    # DataLoaders (you might want to add collate_fn, batch_size, etc.)
    # For simplicity we assume each row is a separate sequence
    train_loader = torch.utils.data.DataLoader(train_df, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_df, batch_size=32, shuffle=False)

    vocab_size = len(le.classes_)
    sdfa_shape = (vocab_size, vocab_size)

    # ---- Hyper‑parameter sweep ---- #
    for seed in seed_list:
        torch.manual_seed(seed)
        np.random.seed(seed)

        for d_model in d_model_list:
            for max_len in max_len_list:
                for er_loss_use in er_loss_use_list:
                    for mix_lambda in mix_lambda_list:

                        # Choose the right model and evaluation function
                        if suffix_prediction:
                            if er_loss_use:
                                model_cls = SDFA_suffix_model
                                eval_fn = evaluate_suffix_model
                            else:
                                model_cls = suffix_model
                                eval_fn = evaluate_suffix_model
                        else:
                            if er_loss_use:
                                model_cls = SDFA_NAP_model
                                eval_fn = evaluate_nap_model
                            else:
                                model_cls = NAP_model
                                eval_fn = evaluate_nap_model

                        # CSV params
                        csv_params = {
                            "seed": seed,
                            "model": model_cls.__name__,
                            "lambda": mix_lambda,
                            "d_model": d_model,
                            "er_loss": er_loss_use,
                            "no_epochs": epochs,
                            "max_len": max_len,
                        }

                        # Run experiment
                        run_experiment(
                            model_cls=model_cls,
                            train_loader=train_loader,
                            test_loader=test_loader,
                            le=le,
                            vocab_size=vocab_size,
                            d_model=d_model,
                            sdfa_shape=sdfa_shape,
                            epochs=epochs,
                            max_len=max_len,
                            er_loss_use=er_loss_use,
                            mix_lambda=mix_lambda,
                            device=device,
                            write_csv=write_csv,
                            csv_params=csv_params,
                            eval_fn=eval_fn,
                        )

    print("All experiments finished.")


# --------------------------------------------------------------------------- #
# 6. Entry point
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
