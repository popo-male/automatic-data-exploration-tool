import streamlit as st
import pandas as pd

from services.reader import Reader
from services.inspector import Inspector
from processors.numerical import NumericalProcessor
from processors.categorical import CategoricalProcessor
from processors.text import TextProcessor
from processors.missing import MissingValueProcessor
from utils.converter import Converter


def main():
    st.set_page_config(
        page_title="EDA Tool",
        page_icon=":rocket:",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # --- STATE MANAGEMENT ---
    if "df" not in st.session_state:
        st.session_state.df = None
    if "dashboard" not in st.session_state:
        st.session_state.dashboard = False
    if "schema_df" not in st.session_state:
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

    # --- CONFIGURATION STAGE (Stacked & Full Width) ---
    if st.session_state.df is not None and not st.session_state.dashboard:
        st.divider()
        st.subheader("🛠️ Data Type Configuration")

        # 1. Raw Data (Full Width)
        st.markdown("##### 1. Raw Data Preview")
        st.dataframe(
            st.session_state.df.head(10),
            use_container_width=True,  # Expands to 100%
        )
        st.caption(f"Showing first 10 rows of {len(st.session_state.df)} total rows.")

        st.write("")  # Spacer

        # 2. Editor (Full Width)
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
                    int((i / total_cols) * 100), text=f"Processing {col_name}..."
                )

                success, result, error_msg = Converter.convert_column(
                    df_working[col_name], target_type
                )

                if success:
                    df_working[col_name] = result
                else:
                    has_error = True
                    st.error(f"❌ Error in column '{col_name}': {error_msg}")
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

    st.title("📊 Data Analysis Report")

    # -- METRICS --
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Rows", meta.total_rows)
    m2.metric("Cols", meta.total_columns)
    m3.metric("Numeric", len(meta.numeric_cols))
    m4.metric("Categorical", len(meta.categorical_cols))
    m5.metric("Missing", sum(meta.missing_values.values()))
    st.divider()

    # -- 1. MISSING VALUES --
    mv_proc = MissingValueProcessor(df)
    df_null = mv_proc.get_missing_report()
    if df_null is not None:
        st.subheader("1. Missing Values")
        c1, c2 = st.columns([1, 2])
        with c1:
            st.dataframe(df_null, hide_index=True, use_container_width=True)
        with c2:
            st.plotly_chart(mv_proc.plot_missing_bar(df_null), use_container_width=True)
        st.divider()

    # -- 2. NUMERICAL (Layout Updated) --
    if meta.numeric_cols:
        st.subheader("2. Numerical Analysis")
        num_sel = st.selectbox("Select Column", meta.numeric_cols)
        num_proc = NumericalProcessor(df)

        # Layout: Graph (Small/Medium) | Table (Wide)
        col_graphs, col_table = st.columns(
            [1, 2]
        )  # 1/3 width for graphs, 2/3 for table

        with col_graphs:
            # Stack graphs vertically in the narrower column
            st.plotly_chart(num_proc.plot_histogram(num_sel), use_container_width=True)
            st.plotly_chart(num_proc.plot_boxplot(num_sel), use_container_width=True)

        with col_table:
            # Table takes the wider column to avoid horizontal scrolling
            st.markdown("#### Statistics")
            st.dataframe(
                num_proc.get_summary_statistics(num_sel), use_container_width=True
            )

        st.divider()

    # -- 3. CATEGORICAL --
    if meta.categorical_cols:
        st.subheader("3. Categorical Analysis")
        cat_sel = st.selectbox("Select Column", meta.categorical_cols)
        cat_proc = CategoricalProcessor(df)
        c1, c2 = st.columns(2)
        c1.plotly_chart(cat_proc.plot_bar_chart(cat_sel), use_container_width=True)
        c2.plotly_chart(cat_proc.plot_pie_chart(cat_sel), use_container_width=True)
        st.divider()

    # -- 4. TEXT --
    if meta.text_cols:
        st.subheader("4. Text Analysis")
        txt_sel = st.selectbox("Select Column", meta.text_cols)
        txt_proc = TextProcessor(df)

        c1, c2 = st.columns(2)
        c1.plotly_chart(txt_proc.plot_sentiment(txt_sel), use_container_width=True)
        c2.plotly_chart(
            txt_proc.plot_length_distribution(txt_sel), use_container_width=True
        )

        st.markdown("#### N-Grams")
        n1, n2, n3 = st.columns(3)
        n1.plotly_chart(txt_proc.get_top_ngrams(txt_sel, 1), use_container_width=True)
        n2.plotly_chart(txt_proc.get_top_ngrams(txt_sel, 2), use_container_width=True)
        n3.plotly_chart(txt_proc.get_top_ngrams(txt_sel, 3), use_container_width=True)

        # -- Word Cloud (Centered & Smaller) --
        st.write("")
        st.markdown("#### Word Cloud")

        # Use columns to center the wordcloud and restrict its size
        wc_c1, wc_c2, wc_c3 = st.columns([1, 2, 1])

        with wc_c2:  # Place in the middle column only
            st.pyplot(txt_proc.plot_wordcloud(txt_sel))


if __name__ == "__main__":
    main()
