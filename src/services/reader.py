import pandas as pd
import streamlit as st
import os


class Reader:
    @staticmethod
    @st.cache_data(show_spinner=False)
    def read_file(buffer) -> pd.DataFrame | None:
        if buffer is None:
            return None

        filename = buffer.name
        extension = os.path.splitext(filename)[1].lower()

        try:
            if extension == ".csv":
                try:
                    return pd.read_csv(buffer)
                except UnicodeDecodeError:
                    buffer.seek(0)
                    return pd.read_csv(buffer, encoding="latin1")
            elif extension in [".xls", ".xlsx"]:
                return pd.read_excel(buffer)
            elif extension == ".json":
                return pd.read_json(buffer)
            elif extension == ".parquet":
                return pd.read_parquet(buffer)
            else:
                st.error(f"Unsupported file format: {extension}")
                return None
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return None
