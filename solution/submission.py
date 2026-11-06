
"""CatBoost submission for bacterial gene expression prediction."""

import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from benchmark_utils.base_solver import GeneCompetSolver
from benchmark_utils.nb_loss import nb_nll


def extract_features(df):
    """Extract categorical IDs and numerical normalization offsets."""

    cat_cols = ["genome_id", "feature_id", "run_accession"]
    num_cols = ["log_length_ratio", "log_size_factor"]

    X = pd.DataFrame(index=df.index)

    for col in cat_cols:
        if col in df.columns:
            values = df[col]

            if isinstance(values.dtype, pd.CategoricalDtype):
                if values.isna().any():
                    values = (
                        values.cat
                        .add_categories(["__MISSING__"])
                        .fillna("__MISSING__")
                    )
            else:
                values = values.fillna("__MISSING__").astype("category")

            X[col] = values
        else:
            X[col] = pd.Series(
                pd.Categorical(["__MISSING__"] * len(df)),
                index=df.index,
            )

    for col in num_cols:
        if col in df.columns:
            X[col] = df[col].fillna(0.0)
        else:
            X[col] = 0.0

    return X, cat_cols


class CatBoostSubmission:

    def __init__(self, iterations=150, lr=0.1):
        self.iterations = iterations
        self.lr = lr
        self.model = None
        self.sigma = 1.0

    def fit(self, train_df, samples_df=None, genomes_dir=None):

        target_col = (
            "readcount" if "readcount" in train_df.columns else "counts"
        )

        if len(train_df) > 2_000_000:
            train_df = train_df.sample(
                n=2_000_000,
                random_state=42,
                )
        y = train_df[target_col].to_numpy(dtype=float)
        X, cat_cols = extract_features(train_df)

        # Split data to select the NB dispersion.
        n = len(y)

        if n >= 2:
            rng = np.random.default_rng(42)
            indices = rng.permutation(n)

            val_size = min(max(int(0.15 * n), 10), n - 1)

            val_idx = indices[:val_size]
            train_idx = indices[val_size:]

            X_train = X.iloc[train_idx]
            X_val = X.iloc[val_idx]

            y_train = y[train_idx]
            y_val = y[val_idx]

            validation_model = CatBoostRegressor(
                iterations=self.iterations,
                learning_rate=self.lr,
                cat_features=cat_cols,
                loss_function="RMSE",
                verbose=0,
                thread_count=4,
                random_seed=42,
                allow_writing_files=False,
            )

            validation_model.fit(
                X_train,
                np.log1p(np.clip(y_train, 0.0, None)),
            )

            pred_log = validation_model.predict(X_val)
            mu_val = np.maximum(
                np.expm1(np.clip(pred_log, -15.0, 15.0)),
                0.5,
            )

            # Select dispersion using validation NB-NLL.
            candidates = [0.5, 1.0, 2.0, 5.0, 10.0]

            losses = [
                nb_nll(y_val, mu_val, r)
                for r in candidates
            ]

            self.sigma = float(candidates[np.argmin(losses)])

        
        self.model = validation_model

        return self

    def predict(self, test_df, samples_df=None, genomes_dir=None):

        X, _ = extract_features(test_df)

        pred_log = self.model.predict(X)

        mu = np.maximum(
            np.expm1(np.clip(pred_log, -15.0, 15.0)),
            0.5,
        )

        sigma = np.full(len(test_df), self.sigma)

        return np.column_stack([mu, sigma])


class Solver(GeneCompetSolver):

    name = "submission"

    requirements = ["pip::catboost"]

    parameters = {
        "iterations": [150],
        "lr": [0.1],
    }

    def get_model(self, meta):
        return CatBoostSubmission(
            iterations=self.iterations,
            lr=self.lr,
        )

    def fit(self, model, train_df, samples_df, genomes_dir):
        model.fit(train_df, samples_df, genomes_dir)
