import pandas as pd
import plotly.express as px


class CategoricalProcessor:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def get_counts(self, column: str) -> pd.DataFrame:
        """Returns value counts and percentage for a category"""
        counts = self.df[column].value_counts().reset_index()
        counts.columns = [column, "Count"]
        counts["Percentage"] = (counts["Count"] / len(self.df)) * 100
        return counts

    def plot_bar_chart(self, column: str):
        """Top 10 category bar chart"""
        data = self.df[column].value_counts().head(10)
        counts = self.get_counts(column)
        fig = px.bar(
            data,
            orientation="h",
            title=f"Top 10 Categories in {column}",
            template="plotly_white",
            color_discrete_sequence=["#EF553B"],
        )
        fig.update_layout(yaxis=dict(autorange="reversed"))
        return fig

    def plot_pie_chart(self, column: str):
        data = self.df[column].value_counts().head(10)
        fig = px.pie(
            values=data.values,
            names=data.index,
            title=f"Composition of {column}",
            hole=0.4,  # donut chart style
        )
        return fig
