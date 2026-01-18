import pandas as pd


class Converter:
    @staticmethod
    def convert_column(series: pd.Series, target_type: str):
        """
        Attempts to convert a column to the target type.
        Returns: (success: bool, result: Series or ErrorMessage, error_msg: str)
        """
        try:
            if target_type == "Numerical":
                # Try converting to numbers (raise error if fails)
                return True, pd.to_numeric(series, errors="raise"), None

            elif target_type == "Datetime":
                # Try converting to datetime
                return True, pd.to_datetime(series, errors="raise"), None

            elif target_type == "Categorical":
                # Everything can technically be a category (string)
                return True, series.astype(str), None

            elif target_type == "Text":
                return True, series.astype(str), None

            return False, series, "Unknown type selected"

        except Exception as e:
            # If conversion fails, return the error
            return False, series, str(e)
