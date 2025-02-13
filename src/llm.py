import json

import duckdb
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape
from openai import OpenAI

prompt_env = Environment(
    loader=FileSystemLoader(searchpath="prompts"),
    autoescape=select_autoescape(),
)


class PandasQueryLLMAgent:
    def __init__(self, openai_key: str) -> None:
        self._client = OpenAI(api_key=openai_key)

    def get_prompt(self, query: str, data: pd.DataFrame):
        return prompt_env.get_template("pandas_query_prompt.jinja2").render(
            query=query,
            data=data.head(),
        )

    def get_pandas_query(
        self, text: str, dataframe: pd.DataFrame
    ) -> tuple[str, list[str]]:
        response = self._client.chat.completions.create(
            messages=[
                {"role": "assistant", "content": self.get_prompt("text", dataframe)}
            ],
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
        )
        print(response.usage)
        response_json = json.loads(response.choices[0].message.content or "{}")
        return response_json.get("query_string", ""), response_json.get("fields", [])


class DuckDBQueryLLMAgent:
    def __init__(self, openai_key: str) -> None:
        self._client = OpenAI(api_key=openai_key)

    def get_prompt(
        self,
        query: str,
        data: duckdb.DuckDBPyRelation,
        table_var: str,
    ) -> str:
        return prompt_env.get_template("duckdb_query_prompt.jinja2").render(
            user_query=query,
            data=data,
            table_var=table_var,
            column_names=data.columns,
        )

    def select_necessary_columns(self, user_query: str, all_columns: str) -> list[str]:
        prompt = prompt_env.get_template("field_filter_prompt.jinja2").render(
            user_query=user_query, column_info=all_columns
        )
        response = self._client.chat.completions.create(
            messages=[{"role": "assistant", "content": prompt}],
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
        )
        print(response.usage)
        print(response.model)
        response_json = json.loads(response.choices[0].message.content or "{}")
        return response_json["fields"]

    def get_duckdb_query(
        self, text: str, data: duckdb.DuckDBPyRelation, table_var: str
    ) -> str:
        prompt = self.get_prompt(text, data, table_var)
        response = self._client.chat.completions.create(
            messages=[{"role": "assistant", "content": prompt}],
            model="gpt-4o",
            response_format={"type": "text"},
            temperature=0,
        )
        print(response.usage)
        print(response.model)
        return response.choices[0].message.content or ""

    def fix_query(
        self, sql_query: str, user_query: str, exception_info: str, table_name: str
    ) -> str:
        prompt = prompt_env.get_template("duckdb_query_fixer_prompt.jinja2").render(
            user_query=user_query,
            table_name=table_name,
            sql_query=sql_query,
            exception_info=exception_info,
        )
        response = self._client.chat.completions.create(
            messages=[{"role": "assistant", "content": prompt}],
            model="gpt-4o-mini",
            response_format={"type": "text"},
        )
        print(response.usage)
        print(response.model)
        return response.choices[0].message.content or ""
