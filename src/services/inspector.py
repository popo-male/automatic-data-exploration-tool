import pandas as pd
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class Metadata:
    """data transfer object for dataset metadata"""
    total_rows: int
    total_columns: int
    numeric_cols: List[str]
    categorical_cols: List[str]
    text_cols: List[str]
    datetime_cols: List[str]
    missing_values: Dict[str, int]

class Inspector:
    """analyzes the dataframe to classify column types and extract metadata"""
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def get_metadata(self) -> Metadata:
        numeric, categoric, text, datetime = self._classify_columns()

        return Metadata(
            total_rows=len(self.df),
            total_columns=len(self.df.columns),
            numeric_cols=numeric,
            categorical_cols=categoric,
            text_cols=text,
            datetime_cols=datetime,
            missing_values=self.df.isnull().sum().to_dict()
        )

    def _classify_columns(self):
        numeric_cols = []
        categorical_cols = []
        text_cols = []
        datetime_cols = []

        for col in self.df.columns:
            if pd.api.types.is_datetime64_any_dtype(self.df[col]):
                datetime_cols.append(col)
                continue

            # convert object col to dateime to check compatibility
            if self.df[col].dtype == 'object':
                try:
                    sample = self.df[col].dropna().head(100)
                    if not sample.empty:
                        pd.to_datetime(sample, errors='raise')
                        datetime_cols.append(col)
                        self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                        continue
                except (ValueError, TypeError):
                    pass

            if pd.api.types.is_numeric_dtype(self.df[col]):
                numeric_cols.append(col)
                continue

            # Heuristic: If unique values < 5% of total rows OR < 50 unique items, it's a category.
            unique_count = self.df[col].nunique(dropna=True)
            total_count = len(self.df)

            if unique_count < 50 or (unique_count / total_count) < 0.05:
                categorical_cols.append(col)
            else:
                text_cols.append(col)

        return numeric_cols, categorical_cols, text_cols, datetime_cols