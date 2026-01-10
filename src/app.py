import streamlit as st
from services.reader import Reader
from services.inspector import Inspector
from processors.numerical import NumericalProcessor
from processors.categorical import CategoricalProcessor
from processors.text import TextProcessor


def main():
    st.set_page_config(
        page_title="EDA Tool",
        page_icon=":rocket:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Sidebar (Input)
    with st.sidebar:
        st.title("📂 Data Loader")
        uploaded_file = st.file_uploader(
            "Upload your dataset", type=["csv", "xlsx", "json"]
        )
        st.divider()
        st.info("Supported formats: CSV, Excel, JSON")

    st.title("Exploratory Data Analysis")

    if uploaded_file:
        df = Reader.read_file(uploaded_file)

        if df is not None:
            inspector = Inspector(df)
            meta = inspector.get_metadata()

            # Top Level Metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Rows", meta.total_rows)
            c2.metric("Total Columns", meta.total_columns)
            c3.metric("Numeric Cols", len(meta.numeric_cols))
            c4.metric("Categorical Cols", len(meta.categorical_cols))
            st.divider()

            # Navigation Tabs
            tab_overview, tab_num, tab_cat, tab_text = st.tabs(
                ["Data Overview", "Numerical", "Categorical", "Textural"]
            )

            # Tab overview
            with tab_overview:
                st.subheader("Raw Data Preview")
                st.dataframe(df.head(20))

                st.subheader("Column Types")
                st.json(
                    {
                        "Numerical": meta.numeric_cols,
                        "Categorical": meta.categorical_cols,
                        "Textural": meta.text_cols,
                        "Datetime": meta.datetime_cols,
                    },
                    expanded=True,
                )

            # Tab Numerical
            with tab_num:
                if not meta.numeric_cols:
                    st.info("No numerical columns found.")
                else:
                    num_processor = NumericalProcessor(df)

                    col_select = st.selectbox(
                        "Select Column to Analyze", meta.numeric_cols
                    )
                    st.subheader("Key Statistics")
                    stats_df = num_processor.get_summary_statistics(col_select)
                    st.dataframe(stats_df, use_container_width=True)

                    st.subheader("Distribution")
                    fig_dist = num_processor.plot_histogram(col_select)
                    st.plotly_chart(fig_dist, use_container_width=True)

                    st.divider()
                    st.subheader("Correlation Heatmap")
                    if len(meta.numeric_cols) > 1:
                        fig_corr = num_processor.plot_heatmap(meta.numeric_cols)
                        st.plotly_chart(fig_corr, use_container_width=True)

            # Tab Categorical
            with tab_cat:
                if not meta.categorical_cols:
                    st.info("No categorical columns found.")
                else:
                    cat_processor = CategoricalProcessor(df)

                    # Sidebar for this tab
                    col_cat_select = st.selectbox(
                        "Select Category", meta.categorical_cols
                    )

                    c1, c2 = st.columns(2)

                    with c1:
                        st.subheader("Bar Chart")
                        fig_bar = cat_processor.plot_bar_chart(col_cat_select)
                        st.plotly_chart(fig_bar, use_container_width=True)

                    with c2:
                        st.subheader("Composition (Pie)")
                        fig_pie = cat_processor.plot_pie_chart(col_cat_select)
                        st.plotly_chart(fig_pie, use_container_width=True)

                    with st.expander("View Data Table"):
                        st.dataframe(
                            cat_processor.get_counts(col_cat_select),
                            use_container_width=True,
                        )

            # Tab Textural
            with tab_text:
                if not meta.text_cols:
                    st.info("No textural columns found.")
                else:
                    text_processor = TextProcessor(df)

                    col_text_select = st.selectbox(
                        "Select Text Column", meta.text_cols
                    )

                    st.subheader("Word Cloud")
                    fig_wc = text_processor.plot_wordcloud(col_text_select)
                    st.pyplot(fig_wc)

                    c1, c2 = st.columns(2)

                    with c1:
                        st.subheader("Text Length Distribution")
                        fig_len = text_processor.plot_length_distribution(col_text_select)
                        st.plotly_chart(fig_len, use_container_width=True)
                        
                    with c2:
                        st.subheader("Top Frequent Words")
                        fig_freq = text_processor.get_top_ngrams(col_text_select)
                        st.plotly_chart(fig_freq, use_container_width=True)
        else:
            st.error("Failed to read the uploaded file.")
    else:
        st.markdown(
            """
        ### Welcome!
        Please upload a file from the sidebar to begin analysis.
        """
        )


if __name__ == "__main__":
    main()
