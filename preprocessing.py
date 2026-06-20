import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler

TARGET = "Credit_Score"
TARGET_MAPPING = {"Poor": 0, "Standard": 1, "Good": 2}

SUPPOSEDLY_NUMERIC = [
    "Age", "Changed_Credit_Limit", "Annual_Income", "Monthly_Inhand_Salary",
    "Outstanding_Debt", "Total_EMI_per_month", "Amount_invested_monthly",
    "Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate",
    "Num_of_Loan", "Num_of_Delayed_Payment", "Num_Credit_Inquiries",
    "Credit_Utilization_Ratio", "Monthly_Balance",
]

DROP_COLS = ["Unnamed: 0", "ID", "Name", "SSN"]
ID_COL = "Customer_ID"

PLACEHOLDERS = {
    "Occupation": "_______",
    "Credit_Mix": "_",
    "Payment_Behaviour": "!@9#%8",
    "Payment_of_Min_Amount": "NM",
}

PAYMENT_PATTERN = r"([A-Za-z]+)_spent_([A-Za-z]+)_value"

# unsensible-value handling
UNSENSIBLE_COLS = [
    "Age", "Interest_Rate", "Num_Bank_Accounts", "Num_Credit_Card",
    "Num_of_Loan", "Num_of_Delayed_Payment", "Num_Credit_Inquiries",
    "Delay_from_due_date",
]
DOMAIN_FLOORS = {
    "Age": 18.0,
    "Interest_Rate": 1e-6,
    "Num_Bank_Accounts": 1.0,
    "Num_Credit_Card": 0.0,
    "Num_of_Loan": 0.0,
    "Num_of_Delayed_Payment": 0.0,
    "Num_Credit_Inquiries": 0.0,
    "Delay_from_due_date": 0.0,
}
BUFFER_FACTORS = np.linspace(0.0, 1.5, 7)[::-1].tolist()
AGE_CEILING = 122.0

# encoding
CREDIT_MIX_MAP = {"Unknown": -9, "Bad": 1, "Standard": 2, "Good": 3}
SPENDING_MAP = {"Unknown": -9, "Low": 1, "High": 2}
VALUE_MAP = {"Unknown": -9, "Small": 1, "Medium": 2, "Large": 3}
ORDINAL_COLS = ["Credit_Mix", "Payment_Spending", "Payment_Value"]
OHE_COLS = ["Occupation", "Payment_of_Min_Amount"]

MONTH_MAP = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12,
}

CREDIT_HISTORY_PATTERN = r"^(\d+)\s+Years\s+and\s+(\d+)\s+Months$"

class CreditPreprocessor(BaseEstimator, TransformerMixin):

    def _clean(self, x):
        df = x.copy()

        for col in SUPPOSEDLY_NUMERIC:
            if col in df.columns:
                s = df[col].astype(str).str.replace(r"[^0-9\.\-]", "", regex=True)
                df[col] = s.replace("", np.nan).astype("float64")

        if "Credit_History_Age" in df.columns:
            parts = df["Credit_History_Age"].astype("string").str.extract(CREDIT_HISTORY_PATTERN)
            df["Credit_History_Age"] = pd.to_numeric(parts[0]) * 12 + pd.to_numeric(parts[1])

        df = df.drop(columns=DROP_COLS, errors="ignore")

        if "Occupation" in df.columns:
            df["Occupation"] = df["Occupation"].str.strip().str.lower()

        for col, bad in PLACEHOLDERS.items():
            if col in df.columns:
                df[col] = df[col].replace(bad, "Unknown")

        if "Payment_Behaviour" in df.columns:
            extracted = df["Payment_Behaviour"].str.extract(PAYMENT_PATTERN)
            df["Payment_Spending"] = extracted[0].fillna(df["Payment_Behaviour"])
            df["Payment_Value"] = extracted[1].fillna(df["Payment_Behaviour"])
            df = df.drop(columns=["Payment_Behaviour"])

        return df

    def _explored_train_range(self, train_df, id_col, column, buffer_factor):
        g = (train_df.dropna(subset=[column])
             .groupby([id_col, column]).size()
             .reset_index(name="n")
             .sort_values([id_col, "n", column], ascending=[True, False, True]))

        reliable_modes = g[g["n"] >= 2].drop_duplicates(id_col, keep="first")[column]
        single_rows = g[g["n"] == 1][column]

        if reliable_modes.empty:
            return max(0.0, train_df[column].min()), train_df[column].max()

        base_min = max(0.0, reliable_modes.min())
        base_max = reliable_modes.max()

        spread = base_max - base_min
        guardrail_limit = base_max + spread * buffer_factor
        valid_singles = single_rows[(single_rows >= base_min) & (single_rows <= guardrail_limit)]
        final_max = max(base_max, valid_singles.max()) if not valid_singles.empty else base_max
        return base_min, final_max

    def _apply_bounds(self, df):
        out = df.copy()
        for col, (lo, hi) in self.bounds_.items():
            if col in out.columns:
                out[col] = out[col].astype("float64")
                mask = out[col].notna() & ((out[col] < lo) | (out[col] > hi))
                out.loc[mask, col] = np.nan
        return out

    def _loan_flags(self, df):
        out = df.copy()
        loans = out["Type_of_Loan"].fillna("Not Specified")
        for loan in self.loan_list_:
            col = f"Loan_{loan.replace(' ', '_')}"
            out[col] = loans.str.contains(loan, na=False, regex=False).astype(int)
        return out.drop(columns=["Type_of_Loan"])

    def _ordinal(self, df):
        out = df.copy()
        out["Credit_Mix"] = out["Credit_Mix"].map(CREDIT_MIX_MAP).fillna(CREDIT_MIX_MAP["Unknown"]).astype(int)
        out["Payment_Spending"] = out["Payment_Spending"].map(SPENDING_MAP).fillna(SPENDING_MAP["Unknown"]).astype(int)
        out["Payment_Value"] = out["Payment_Value"].map(VALUE_MAP).fillna(VALUE_MAP["Unknown"]).astype(int)
        return out

    def _apply_ohe(self, df):
        encoded = self.ohe_.transform(df[OHE_COLS])
        names = self.ohe_.get_feature_names_out(OHE_COLS)
        ohe_df = pd.DataFrame(encoded, columns=names, index=df.index)
        return pd.concat([df.drop(columns=OHE_COLS), ohe_df], axis=1)

    def _cyclical_month(self, df):
        out = df.copy()
        month = out["Month"].map(MONTH_MAP).fillna(self.month_fill_)
        out["Month_sin"] = np.sin(2 * np.pi * month / 12)
        out["Month_cos"] = np.cos(2 * np.pi * month / 12)
        return out.drop(columns=["Month"])

    def _assemble(self, df):
        df = df.copy()
        df[self.numeric_cols_] = df[self.numeric_cols_].fillna(self.train_medians_)
        df = self._loan_flags(df)
        df = self._ordinal(df)
        df = self._apply_ohe(df)
        df = self._cyclical_month(df)
        return df

    def fit(self, x, y=None):
        df = self._clean(x)

        # per-customer unsensible-value bounds
        self.bounds_ = {}
        self.buffer_factor_ = BUFFER_FACTORS[-1]
        if ID_COL in df.columns:
            for factor in BUFFER_FACTORS:
                candidate = {
                    col: self._explored_train_range(df, ID_COL, col, factor)
                    for col in UNSENSIBLE_COLS if col in df.columns
                }
                if "Age" in candidate and candidate["Age"][1] <= AGE_CEILING:
                    self.buffer_factor_ = factor
                    break
            else:
                candidate = {
                    col: self._explored_train_range(df, ID_COL, col, self.buffer_factor_)
                    for col in UNSENSIBLE_COLS if col in df.columns
                }
            self.bounds_ = {
                col: (max(DOMAIN_FLOORS.get(col, 0.0), lo), hi)
                for col, (lo, hi) in candidate.items()
            }
        df = self._apply_bounds(df)
        df = df.drop(columns=[ID_COL], errors="ignore")

        # imputation
        self.numeric_cols_ = df.select_dtypes(include="number").columns.tolist()
        self.train_medians_ = df[self.numeric_cols_].median()
        self.month_fill_ = int(df["Month"].map(MONTH_MAP).mode().iloc[0])

        # loan vocabulary
        loan_series = (
            df["Type_of_Loan"].dropna()
            .str.replace(" and ", ", ", regex=False)
            .str.split(", ").explode().str.strip(" ,")
        )
        self.loan_list_ = [l for l in loan_series.unique() if l not in ("Not Specified", "")]

        # fit encoder and scaler
        df = df.copy()
        df[self.numeric_cols_] = df[self.numeric_cols_].fillna(self.train_medians_)
        df = self._loan_flags(df)
        df = self._ordinal(df)
        self.ohe_ = OneHotEncoder(drop=["Unknown"] * len(OHE_COLS), handle_unknown="ignore", sparse_output=False).fit(df[OHE_COLS])
        df = self._apply_ohe(df)
        df = self._cyclical_month(df)

        exclude = ORDINAL_COLS + ["Month_sin", "Month_cos"]
        self.scale_cols_ = [
            c for c in df.select_dtypes(include="number").columns
            if c not in exclude
            and not c.startswith("Loan_")
            and not c.startswith("Occupation_")
            and not c.startswith("Payment_of_Min_Amount_")
        ]
        self.scaler_ = RobustScaler().fit(df[self.scale_cols_])
        self.feature_names_ = df.columns.tolist()
        return self

    def transform(self, x):
        df = self._apply_bounds(self._clean(x))
        df = df.drop(columns=[ID_COL], errors="ignore")
        df = self._assemble(df)
        df = df.reindex(columns=self.feature_names_, fill_value=0)
        df[self.scale_cols_] = self.scaler_.transform(df[self.scale_cols_])
        return df

    def run(self, data_file, seed=2802405030):
        df = pd.read_csv(data_file)
        x = df.drop(columns=[TARGET])
        y = df[TARGET]
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=seed, stratify=y)
        x_train_mat = self.fit_transform(x_train)
        x_test_mat = self.transform(x_test)
        y_train_enc = y_train.map(TARGET_MAPPING).values
        y_test_enc = y_test.map(TARGET_MAPPING).values
        return x_train_mat, x_test_mat, y_train_enc, y_test_enc, list(TARGET_MAPPING)
