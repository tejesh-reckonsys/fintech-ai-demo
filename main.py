from contextlib import suppress
import os

import duckdb
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain.agents.agent_types import AgentType
from langchain_core.agents import AgentAction, AgentActionMessageLog
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from src.helper import get_col_info
from src.llm import DuckDBQueryLLMAgent

load_dotenv()

duckdb_query_agent = DuckDBQueryLLMAgent(
    openai_key=os.environ.get("OPENAI_API_KEY", "")
)


data = pd.read_csv("test_data/63861_GT_output_DV.csv")

agent = create_pandas_dataframe_agent(
    ChatOpenAI(
        temperature=0,
        model="gpt-4o-mini",
        api_key=SecretStr(os.environ.get("OPENAI_API_KEY", "")),
    ),
    data,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    return_intermediate_steps=True,
    allow_dangerous_code=True,
)

# Title of the app
st.title("Finance Bot")

# Input field for user to enter a string
user_input = st.text_input("Enter a string:")

# Button to trigger the action
if st.button("Submit"):
    if user_input:
        # output = agent.invoke(user_input)
        # codes = [
        #     step.tool_input["query"]
        #     for step in output["intermediate_steps"][0]
        #     if isinstance(step, AgentActionMessageLog)
        # ]
        # for code in codes:
        #     with suppress(Exception):
        #         x = eval(code)
        #         print(x)

        fields = duckdb_query_agent.select_necessary_columns(
            user_query=user_input, all_columns=get_col_info(data)
        )

        print(fields)

        filtered_data = data[fields]

        query = duckdb_query_agent.get_duckdb_query(
            data=duckdb.query("SELECT * FROM filtered_data LIMIT 10"),
            table_var="filtered_data",
            text=user_input,
        )
        print(query)

        try:
            output = duckdb.query(query).df()
        except Exception as ex:
            print(ex)
            new_query = duckdb_query_agent.fix_query(
                user_query=user_input,
                table_name="filtered_data",
                sql_query=query,
                exception_info=str(ex),
            )
            print(new_query)
            output = duckdb.query(new_query).df()

        st.write(output)

    else:
        st.write("Please enter a string.")
