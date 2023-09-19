import streamlit as st
import psycopg2
from langchain.llms import OpenAI
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import AgentExecutor
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
import os
import openai

os.environ['OPENAI_API_KEY'] = "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"

db = SQLDatabase.from_uri(
    "postgresql+psycopg2://qooptjsutkuvav:8a5745c259da4ccb29190a24df8562fa2fcd18186c839df1fe54013f3ddc2283@ec2-54-78-142-10.eu-west-1.compute.amazonaws.com:5432/df4a5mjdu9ab9n"
)

openai.api_key = "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"
model_lst = openai.Model.list()
name = []
for i in model_lst['data']:
    name.append(i['id'])

prompt = st.text_area('', 'Please enter a question', height = 150)

colb, nulla,coli = st.columns([3,2,1])

with coli:
    p = st.button('submit', type = 'primary')

with colb:
    m = st.selectbox('Model', name, 1, label_visibility = 'collapsed')

llm = OpenAI(temperature=0, openai_api_key = os.environ.get('OPENAI_API_KEY'),  model_name = m)

QUERYCHAIN = """
You will be connecting to a postgresql database. The database is composed by 5 tables. 
The table studyinfo contains general information on clinical trials. 
The table intervention contains information on the intervention (drug, device,....) that have been used on the clinical trials. 
The table condition contains information regarding the condition that is targeted by the clinical trials.
The table outcome contains information regarding the outcome used in the clinical trials.
The table eligibility contains information regarding the exclusion and inclusion criteria used, the gender included, the maximum and minimum age allowed and weteher or not healthy volunteer are nclude in a clinical trials
Always use the nct_id column to join the tables.
Delimit column names with quotation marks when the column name contain uppercase and lowercase letters.
use parenthesis for DISTINCT, COUNT, SUM statements

Given an input question, first create a syntactically correct postgresql query to run then look at the results of the query and return the answer. 
Use the following format: 

Question: Question here
SQLQuery: SQL Query to run
SQLResult: Result of the SQLQuery
Answer: Final answer here

{question}
"""

db_chain = SQLDatabaseChain.from_llm(llm, db, return_intermediate_steps = True)

QUERYAGENT = """You are an agent designed to interact with a PostgreSQL database. 
Given an input question, create a syntactically correct postgresql query to run, then look at the results of the query and return the answer. 
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 10 results. 
You can order the results by a relevant column to return the most interesting examples in the database. 
Never query for all the columns from a specific table, only ask for the relevant columns given the question. 
You have access to tools for interacting with the database. 
Only use the below tools. Only use the information returned by the below tools to construct your final answer. 

sql_db_query: Input to this tool is a detailed and correct SQL query, output is a result from the database. If the query is not correct, an error message will be returned. If an error is returned, rewrite the query, check the query, and try again. If you encounter an issue with Unknown column 'xxxx' in 'field list', using sql_db_schema to query the correct table fields.
sql_db_schema: Input to this tool is a comma-separated list of tables, output is the schema and sample rows for those tables. Be sure that the tables actually exist by calling sql_db_list_tables first! Example Input: 'table1, table2, table3' 
sql_db_list_tables: Input is an empty string, output is a comma separated list of tables in the database. 
sql_db_query_checker: Use this tool to double check if your query is correct before executing it. Always use this tool before executing a query with sql_db_query! 

The database is composed by 5 tables. 
The table studyinfo contains general information on clinical trials. 
The table intervention contains information on the intervention that have been used on the clinical trials. 
The table condition contains information regarding the condition that is targeted by the clinical trials. 
The table outcome contains information regarding the outcome used in the clinical trials. 
The table eligibility contains information regarding the exclusion and inclusion criteria used, the gender included, the maximum and minimum age allowed and whether or not healthy volunteer are allowed in a clinical trials. 

Always use the nct_id column to join the tables. 
When creating the query, delimit column names with quotation marks when the column name contain uppercase and lowercase letters. 
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again. 
DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database. 

If the question does not seem related to the database, just return "I don't know" as the answer. 



Use the following format: 

Question: the input question you must answer 
Thought: you should always think about what to do 
Action: the action to take, should be one of [sql_db_query, sql_db_schema, sql_db_list_tables, sql_db_query_checker] 
Action Input: the input to the action 
Observation: the result of the action 
... (this Thought/Action/Action Input/Observation can repeat N times) 
Thought: I now know the final answer 
Final Answer: the final answer to the original input question 

Begin! 

Question: {input}
Thought: I should look at the tables in the database to see what I can query.  Then I should query the schema of the most relevant tables. {agent_scratchpad} """

toolkit = SQLDatabaseToolkit(db=db, llm=llm)

agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose = True,
    template = QUERYAGENT
)

agent_executor.return_intermediate_steps = True

col1, col2 = st.columns(2)

with col1:
    output = ''
    if p :
        question = QUERYCHAIN.format(question = prompt)
        res = db_chain(question)
        output = """RESPONSE :\n\n""" + res['result'] + """\n\n\nQUERY :\n\n""" + res['intermediate_steps'][1]

    st.text_area('Chain response', value = output, height = 650)

with col2:
    outputAgent = ''
    queryAgent = ''
    #if p :
        #response = agent_executor(prompt)
        #l = len(response['intermediate_steps']) - 1
        #outputAgent = """RESPONSE : """ + response['output'] + """\n\n\nQUERY : """ + response['intermediate_steps'][l][0].tool_input
  
    st.text_area('Agent response', value = outputAgent, height = 650)

