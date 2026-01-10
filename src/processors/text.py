import pandas as pd
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
from collections import Counter
import matplotlib.pyplot as plt


class TextProcessor:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def plot_wordcloud(self, column: str):
        text_data = " ".join(self.df[column].dropna().astype(str).tolist())

        wc = WordCloud(
            width=800,
            height=400,
            background_color="white",
            stopwords=STOPWORDS,
            min_font_size=10,
        ).generate(text_data)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        return fig

    def plot_length_distribution(self, column: str):
        text_lengths = self.df[column].dropna().astype(str).apply(len)

        fig = px.histogram(
            text_lengths,
            x=text_lengths,
            title="Distribution of Text Lengths",
            template="plotly_white",
            labels={"x": "Character Count"},
            color_discrete_sequence=["#00CC96"],  # Green-ish
        )
        return fig

    def get_top_ngrams(self, column: str, n=10):
        text_data = " ".join(self.df[column].dropna().astype(str).tolist()).lower()
        words = text_data.split()

        # filter stopwords
        filtered_words = [
            word for word in words if word not in STOPWORDS and len(word) > 2
        ]

        count = Counter(filtered_words)
        most_common = count.most_common(n)

        df_frequency = pd.DataFrame(most_common, columns=["Word", "Count"])

        fig = px.bar(
            df_frequency,
            x="Count",
            y="Word",
            orientation="h",
            title=f"Top {n} Frequent Words",
            template="plotly_white",
        )
        fig.update_layout(yaxis=dict(autorange="reversed"))
        return fig
