import json
from typing import Any
import pandas as pd

data = pd.read_csv("test_data/63861_GT_output_DV.csv")

column_info = {column: dtype for column, dtype in zip(data.columns, data.dtypes)}


def print_col_info(col_info: dict[str, Any]):
    for key, value in col_info.items():
        print(f"{key}: {value}")


print_col_info(column_info)

import duckdb

data = {"dv_data": data}

print(duckdb.query("SELECT * FROM 'test_data/1583_GT_output_GOOG.csv' LIMIT 2"))
