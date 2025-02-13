import pandas as pd


def get_col_info(data: pd.DataFrame) -> str:
    res = ""
    for column, dtype in zip(data.columns, data.dtypes):
        res += f"{column}: {dtype}\n"
    return res
