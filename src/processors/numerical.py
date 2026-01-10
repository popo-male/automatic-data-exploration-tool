import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff


class NumericalProcessor:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def get_summary_statistics(self, column: str) -> pd.DataFrame:
        """Returns summary statistics for a numerical column."""
        desc = self.df[column].describe()
        return desc.to_frame().T

    def plot_histogram(self, column: str):
        """Generates a Histogram with a Box Plot margin."""
        fig = px.histogram(
            self.df,
            x=column,
            marginal="box",  # Adds the boxplot on top
            title=f"Distribution of {column}",
            template="plotly_white",
            color_discrete_sequence=["#636EFA"],
        )
        return fig

    def plot_heatmap(self, cols: list):
        """Generates a heatmap for selected columns."""
        if len(cols) < 2:
            return None

        corr_matrix = self.df[cols].corr()

        fig = px.imshow(
            corr_matrix,
            text_auto=".2f",  # type: ignore[arg-type]
            aspect="auto",
            title="Correlation Matrix",
            color_continuous_scale="RdBu_r",
        )
        return fig
