import pandas as pd
import plotly.express as px


class MissingValueProcessor:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def get_missing_report(self):
        nulls = self.df.isnull().sum()
        # only show columns with missing values
        nulls = nulls[nulls > 0]

        if nulls.empty:
            return None

        df_null = pd.DataFrame(
            {
                "Column": nulls.index,
                "Missing Values": nulls.values,
                "Percentage": (nulls / len(self.df)) * 100,
            }
        )
        return df_null.sort_values(by="Percentage", ascending=False)

    def plot_missing_bar(self, df_null: pd.DataFrame):
        fig = px.bar(
            df_null,
            x="Column",
            y="Percentage",
            title="Missing Values by Column (%)",
            color="Percentage",
            color_continuous_scale="Reds",
        )
        return fig
