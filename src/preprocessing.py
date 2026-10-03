"""Fitted preprocessing for the late-delivery exam (Set C).

Order of operations (all statistics learned on the 192 FIT records only):
  1. median imputation of the four numeric predictors
  2. engineered_feature = load / (staff + 1)   (imputed, original scale)
  3. StandardScaler on the 5 numeric features (4 originals + engineered_feature)
  4. One-hot encoding of `group` (handle_unknown='ignore'), left UNSCALED

The target (`late`) and `record_id` are never used.
"""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ["distance", "load", "traffic", "staff"]
CATEGORICAL = "group"
ENGINEERED = "engineered_feature"
SCALED_COLS = NUMERIC + [ENGINEERED]


class Preprocessor:
    def fit(self, df: pd.DataFrame):
        self.imputer_ = SimpleImputer(strategy="median").fit(df[NUMERIC])
        numeric_unscaled = self.impute_engineer(df)
        self.scaler_ = StandardScaler().fit(numeric_unscaled[SCALED_COLS])
        self.encoder_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(df[[CATEGORICAL]])
        self.feature_names_ = SCALED_COLS + [f"group_{c}" for c in self.encoder_.categories_[0]]
        return self

    def impute_engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        """Median-imputed numeric columns + engineered feature, ORIGINAL scale."""
        out = pd.DataFrame(self.imputer_.transform(df[NUMERIC]), columns=NUMERIC, index=df.index)
        out[ENGINEERED] = out["load"] / (out["staff"] + 1)
        return out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        unscaled = self.impute_engineer(df)
        scaled = pd.DataFrame(self.scaler_.transform(unscaled[SCALED_COLS]),
                              columns=SCALED_COLS, index=df.index)
        onehot = pd.DataFrame(self.encoder_.transform(df[[CATEGORICAL]]),
                              columns=self.feature_names_[len(SCALED_COLS):], index=df.index)
        return pd.concat([scaled, onehot], axis=1)
