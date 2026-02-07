import streamlit as st
import pandas as pd

from src.services.reader import Reader
from src.services.inspector import Inspector
from src.processors.numerical import NumericalProcessor
from src.processors.categorical import CategoricalProcessor
from src.processors.text import TextProcessor
from src.processors.missing import MissingValueProcessor
from src.utils.converter import Converter


def main():
    st.set_page_config(
        page_title="EDA Tool",
        page_icon=":rocket:",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # --- STATE MANAGEMENT ---
    if "df" not in st.session_state:
        # check if dataframe exists in session state
        st.session_state.df = None
    if "dashboard" not in st.session_state:
        # initialize dashboard state
        st.session_state.dashboard = False
    if "schema_df" not in st.session_state:
        # initialize schema dataframe state
        st.session_state.schema_df = None

    # --- HEADER ---
    st.markdown(
        """
        <div style='text-align: center; margin-bottom: 20px;'>
            <h1>Exploratory Data Analysis Tool</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- UPLOADER ---
    uploaded_file = st.file_uploader(
        "📂 Drag and drop dataset here (CSV, Excel, JSON)",
        type=["csv", "xlsx", "json"],
    )

    if uploaded_file:
        current_file_name = uploaded_file.name
        # Reset state if new file is uploaded
        if (
            "last_file" not in st.session_state
            or st.session_state.last_file != current_file_name
        ):
            st.session_state.last_file = current_file_name
            st.session_state.dashboard = False

            # Read File
            df = Reader.read_file(uploaded_file)
            st.session_state.df = df

            # Generate Initial Schema
            inspector = Inspector(df)
            meta = inspector.get_metadata()

            schema_data = []
            for col in df.columns:
                dtype = "Text"
                if col in meta.numeric_cols:
                    dtype = "Numerical"
                elif col in meta.categorical_cols:
                    dtype = "Categorical"
                elif col in meta.datetime_cols:
                    dtype = "Datetime"
                elif col in meta.text_cols:
                    dtype = "Text"

                schema_data.append(
                    {"Column": col, "Detected Type": dtype, "Target Type": dtype}
                )

            st.session_state.schema_df = pd.DataFrame(schema_data)

    # --- CONFIGURATION STAGE ---
    if st.session_state.df is not None and not st.session_state.dashboard:
        st.divider()
        st.subheader("🛠️ Data Type Configuration")

        # Raw Data
        st.markdown("##### 1. Raw Data Preview")
        st.dataframe(
            st.session_state.df.head(10),
            use_container_width=True,  # Expands to 100%
        )
        st.caption(f"Showing first 10 rows of {len(st.session_state.df)} total rows.")

        st.write("")  # Spacer

        # Editor
        st.markdown("##### 2. Data Types Classification")
        st.info(
            "Select the correct **Target Type** in the table below if the detection is wrong."
        )

        edited_schema = st.data_editor(
            st.session_state.schema_df,
            column_config={
                "Column": st.column_config.Column(disabled=True),
                "Detected Type": st.column_config.Column(disabled=True),
                "Target Type": st.column_config.SelectboxColumn(
                    "Target Type",
                    options=["Numerical", "Categorical", "Text", "Datetime"],
                    required=True,
                ),
            },
            hide_index=True,
            use_container_width=True,  # Expands to 100%
            key="editor",
        )

        st.markdown("---")
        if st.button("Start EDA", type="primary", use_container_width=True):
            # perform conversion logic
            df_working = st.session_state.df.copy()
            has_error = False
            progress_bar = st.progress(0, text="Validating columns...")
            total_cols = len(edited_schema)

            for i, row in edited_schema.iterrows():
                col_name = row["Column"]
                target_type = row["Target Type"]
                progress_bar.progress(
                    int((i / total_cols) * 100),  # type: ignore
                    text=f"Processing {col_name}...",
                )

                success, result, error_msg = Converter.convert_column(
                    df_working[col_name], target_type
                )

                if success:
                    df_working[col_name] = result
                else:
                    has_error = True
                    st.error(f"Error in column '{col_name}': {error_msg}")
                    break

            progress_bar.empty()

            if not has_error:
                st.session_state.df = df_working
                st.session_state.dashboard = True
                st.rerun()

    # --- DASHBOARD RENDER ---
    if st.session_state.dashboard:
        st.divider()
        render_dashboard(st.session_state.df)


def render_dashboard(df):
    inspector = Inspector(df)
    meta = inspector.get_metadata()

    # --- SUMMARY ---
    st.markdown("### Report Summary")

    # Calculate key insights
    total_cells = meta.total_rows * meta.total_columns
    total_missing = sum(meta.missing_values.values())
    missing_percentage = (total_missing / total_cells) * 100 if total_cells > 0 else 0
    duplicates = df.duplicated().sum() 

    # Determine dominant data type
    type_counts = {
        "Numerical": len(meta.numeric_cols),
        "Categorical": len(meta.categorical_cols),
        "Text": len(meta.text_cols),
    }
    dominant_type = max(type_counts, key=type_counts.get)  # type: ignore

    # summary container
    with st.container(border=True):
        st.markdown(
            f"""
            **Dataset Overview:**
            The dataset contains **{meta.total_rows:,} rows** and **{meta.total_columns} columns**. 
            It is composed of **{dominant_type}** data.
            
            **Data Quality:**
            - **Missing Values:** {total_missing:,} cells ({missing_percentage:.2f}%) are empty.
            - **Duplicates:** There are {duplicates:,} duplicate rows.
            - **Memory Usage:** {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB.
            
            **Recommendations:**
            """
        )

        # Recommendations
        recommend = []
        if duplicates > 0:
            recommend.append(
                f"- Consider removing the **{duplicates} duplicate rows** to avoid skewed results."
            )
        if missing_percentage > 5:
            recommend.append(
                "- Missing data is significant (>5%). Consider imputation or dropping rows."
            )
        if len(meta.numeric_cols) == 0:
            recommend.append(
                "- No numerical columns found. Limited statistical analysis available."
            )
        if len(meta.text_cols) > 0:
            recommend.append(
                "- Text columns detected. Check the 'Text Analysis' section for sentiment insights."
            )

        if not recommend:
            st.write("- Data looks clean and ready for analysis!")
        else:
            for r in recommend:
                st.write(r)

    st.divider()

    # -- Metrics --
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Rows", meta.total_rows)
    m2.metric("Cols", meta.total_columns)
    m3.metric("Numeric", len(meta.numeric_cols))
    m4.metric("Categorical", len(meta.categorical_cols))
    m5.metric("Missing", sum(meta.missing_values.values()))
    st.divider()

    # -- MIissing values --
    missing_processor = MissingValueProcessor(df)
    df_null = missing_processor.get_missing_report()
    if df_null is not None:
        st.subheader("Missing Values")
        c1, c2 = st.columns([1, 2])
        with c1:
            st.dataframe(df_null, hide_index=True, use_container_width=True)
        with c2:
            st.plotly_chart(missing_processor.plot_missing_bar(df_null), use_container_width=True)
        st.divider()

    # -- NUMERICAL --
    if meta.numeric_cols:
        st.subheader("Numerical Analysis")
        num_sel = st.selectbox("Select Column", meta.numeric_cols)
        num_processor = NumericalProcessor(df)

        # Layout: Graph (Small/Medium) | Table (Wide)
        col_graphs, col_table = st.columns(
            [1, 2]
        )  # 1/3 width for graphs, 2/3 for table

        with col_graphs:
            st.plotly_chart(num_processor.plot_histogram(num_sel), use_container_width=True)
            st.plotly_chart(num_processor.plot_boxplot(num_sel), use_container_width=True)

        with col_table:
            st.markdown("#### Descriptive Statistics")
            st.dataframe(
                num_processor.get_summary_statistics(num_sel), use_container_width=True
            )

        st.divider()

    # -- CATEGORICAL --
    if meta.categorical_cols:
        st.subheader("Categorical Analysis")
        cat_sel = st.selectbox("Select Column", meta.categorical_cols)
        cat_processor = CategoricalProcessor(df)
        c1, c2 = st.columns(2)
        c1.plotly_chart(cat_processor.plot_bar_chart(cat_sel), use_container_width=True)
        c2.plotly_chart(cat_processor.plot_pie_chart(cat_sel), use_container_width=True)
        st.divider()

    # -- TEXT --
    if meta.text_cols:
        st.subheader("Text Analysis")
        text_sel = st.selectbox("Select Column", meta.text_cols)
        text_processor = TextProcessor(df)

        c1, c2 = st.columns(2)
        c1.plotly_chart(text_processor.plot_sentiment(text_sel), use_container_width=True)
        c2.plotly_chart(
            text_processor.plot_length_distribution(text_sel), use_container_width=True
        )

        st.markdown("#### N-Grams")
        n1, n2, n3 = st.columns(3)
        n1.plotly_chart(text_processor.get_top_ngrams(text_sel, 1), use_container_width=True)
        n2.plotly_chart(text_processor.get_top_ngrams(text_sel, 2), use_container_width=True)
        n3.plotly_chart(text_processor.get_top_ngrams(text_sel, 3), use_container_width=True)

        # -- Word Cloud (Centered & Smaller) --
        st.write("")
        st.markdown("#### Word Cloud")

        # Use columns to center the wordcloud and restrict its size
        wc_c1, wc_c2, wc_c3 = st.columns([1, 2, 1])

        with wc_c2:  # Place in the middle column only
            fig = text_processor.plot_wordcloud(text_sel)
            if fig:
                st.pyplot(fig)


if __name__ == "__main__":
    main()
