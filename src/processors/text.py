import re
import pandas as pd
import plotly.express as px
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS
from collections import Counter
import matplotlib.pyplot as plt


class TextProcessor:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.stopwords = set(STOPWORDS)

    def _clean_words(self, column: str):
        """Helper to get a clean list of words (no numbers, no stopwords)."""
        text_data = " ".join(self.df[column].dropna().astype(str).tolist()).lower()
        
        # Remove non-alphabetic characters (keep spaces)
        # This removes "123" or "user123" if you only want pure words
        # Regex explanation: Keep only a-z and whitespace
        text_only = re.sub(r'[^a-z\s]', '', text_data)
        
        words = text_only.split()
        
        # Filter: 
        # 1. strictly alphabetic (double check)
        # 2. not in stopwords
        # 3. length > 2 (removes "is", "a", "at")
        clean_words = [
            w for w in words 
            if w.isalpha() and w not in self.stopwords and len(w) > 2
        ]
        return clean_words

    def plot_wordcloud(self, column: str):
        clean_words = self._clean_words(column)
        text_data = " ".join(clean_words)
        
        if not text_data: 
            return None

        wc = WordCloud(
            width=800,
            height=400,
            background_color="white",
            stopwords=self.stopwords,
            min_font_size=10,
        ).generate(text_data)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        plt.tight_layout()
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

    def get_top_ngrams(self, column: str, n_gram=1, top_n=10):
        text_data = " ".join(self.df[column].dropna().astype(str).tolist()).lower()
        words = text_data.split()

        # filter stopwords
        filtered_words = [
            word for word in words if word not in STOPWORDS and len(word) > 2
        ]

        if n_gram==1:
            grams = words
        else:
            grams = zip(*[words[i:] for i in range(n_gram)])
            grams = [" ".join(gram) for gram in grams]

        count = Counter(filtered_words)
        most_common = count.most_common(top_n)

        df_frequency = pd.DataFrame(most_common, columns=["N-gram", "Count"])
        title_map = {1: "Unigrams", 2: "Bigrams", 3: "Trigrams"}

        fig = px.bar(
            df_frequency,
            x="Count",
            y="N-gram",
            orientation="h",
            title=f"Top {top_n} {title_map.get(n_gram, 'N-grams')}",
            template="plotly_white",
            color_discrete_sequence=["#636EFA"]
        )
        fig.update_layout(yaxis=dict(autorange="reversed"))
        return fig

    def plot_sentiment(self, column: str):
        sample = self.df[column].dropna().astype(str).head(1000)
        sentiments = sample.apply(lambda x: TextBlob(x).sentiment.polarity) # type: ignore[attr-defined]

        def get_label(score):
            if score > 0.1:
                return "Positive"
            elif score < -0.1:
                return "Negative"
            else:
                return "Neutral"

        labels = sentiments.apply(get_label)
        counts = labels.value_counts()
        fig = px.pie(
            values=counts.values,
            names=counts.index,
            title="Sentiment Distribution (Sampled)",
            color=counts.index,
            color_discrete_map={
                "Positive": "#00CC96",  # Green
                "Negative": "#EF553B",  # Red
                "Neutral": "#636EFA",  # Blue
            },
        )
        return fig
