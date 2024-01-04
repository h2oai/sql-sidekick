# Chain of thought for reasoning and task decomposition
# Reference: https://arxiv.org/pdf/2201.11903.pdf
TASK_PROMPT = {
    "system_prompt": "Act as a Data Analyst",
    "user_prompt": """
        ### For table {_table_name}, given an input *Question*, let's work it out in a detailed step by step way and only return specific, detailed and informative tasks as an ordered numeric list for SQL generation to be sure we have the right answer.
        Use values that are explicitly mentioned in the *Question*.
        Use the *History* and *Context* section for co-reference and to infer relationships and identify column names. *Context* contains entity mapping containing keys:values.
        If the words in the *Question* do not match column names *Data* section; Search for them in *Context* section.
        Always use *Context* with highest similarity score with the *Question*.
        If words in the *Question* match more than one key, include both the values using "or" when forming step by step tasks.
        If no information related to the *Question* is found; apply self reasoning and predict for possible tasks.
        Infer the return type of the Question.
        Do not generate SQL response, only return itemized tasks.
        # *Data:* \nFor table {_table_name} schema info is mentioned below,\n{_data_info}
        # *History*: \n{_sample_queries}
        # *Question*: For table {_table_name}, {_question_str}, *Context*: {_context}
        # Output: Tasks: ordered numeric list of tasks
    """,
}

# Few shot learning prompt
## Task Description
## Examples
## Prompt
# Reference: https://arxiv.org/pdf/2005.14165.pdf
QUERY_PROMPT = """
                ### System: Act as a SQL Expert
                # For table {_table_name}, given an input *Question*, only generate syntactically correct {dialect} SQL queries.
                # Let's work it out in a detailed step by step way using the reasoning from *Tasks* section.
                # Pick the SQL query which has the highest average log probability if more than one result is likely to answer the
                candidate *Question*.
                ### {dialect} SQL tables
                ### *Data:* \nFor table {_table_name} schema info is mentioned below,\n{_data_info}
                ### *History*:\n{_sample_queries}
                ### *Question*: For table {_table_name}, {_question}
                # SELECT 1
                ### *Tasks for table {_table_name}*:\n{_tasks}
                ### *Policies for SQL generation*:
                # Avoid overly complex SQL queries, favor concise human readable SQL queries
                # Avoid patterns that might be vulnerable to SQL injection
                # Use values and column names that are explicitly mentioned in the question or in the *Data* section.
                # Do not query for columns that do not exist
                # Validate column names with the table name when needed
                # Don't use aggregate and window function together
                # Use COUNT(1) instead of COUNT(*)
                # Return with LIMIT 100
                # Prefer NOT EXISTS to LEFT JOIN ON null id
                # Avoid using the WITH statement
                # When using DESC keep NULLs at the end
                # If JSONB format found in Table schema, do pattern matching on keywords from the question and use SQL functions such as ->> or ->
                # Use prepared statements with parameterized queries to prevent SQL injection
                # Add explanation and reasoning for each SQL query
            """

DEBUGGING_PROMPT = {
    "system_prompt": "Act as a SQL expert for {dialect} database",
    "user_prompt": """
                ### Help fix syntax errors for provided incorrect SQL Query.
                # Error: {ex_traceback}
                # Query:\n {qry_txt}
                # Output: Add ``` as prefix and ``` as suffix to generated SQL
                """,
}

H2OGPT_DEBUGGING_PROMPT = {
"system_prompt": "Act as a SQL expert for {dialect} database",
"user_prompt": """
Help fix the provided incorrect SQL Query mentioned below in the *Query* section",\n
### Error: {ex_traceback}\n
### Query:\n {qry_txt}\n\n
Output: Add ``` as prefix and ``` as suffix to generated SQL
""",
}

H2OGPT_GUARDRAIL_PROMPT = {
"system_prompt": "Act as a Security expert your job is to detect SQL injection vulnerabilities",
"user_prompt":"""
Help audit SQL injection patterns within the provided the SQL *Query*.
Flag as vulnerable if there are any known SQL injection string pattern is found in the *Query*, few *Examples* are provided below,
### *Examples*:\n
1. SELECT * FROM sleep_health_and_lifestyle_study WHERE UserId = 105; vulnerable: false
2. SELECT * FROM sleep_health_and_lifestyle_study WHERE UserId = 105 OR 1=1; vulnerable: true
\n
Only SELECT queries are allowed, flag as vulnerable if other SQL statements are found in the *Query* (e.g. DROP, INSERT, UPDATE, DELETE, etc.).
If there are more than one possible vulnerabilities, summarize in a single explanation.\n
### Query:\n {query_txt}\n\n
### Output: Return result as a valid dictionary string using the JSON schema format, don't add a separate Explanation section or after the json schema, \n{schema}
"""
}

NSQL_QUERY_PROMPT = """
For {dialect} SQL TABLE '{table_name}' with sample question/answer pairs,\n({sample_queries})

CREATE TABLE '{table_name}'({column_info}
)

Table '{table_name}' has sample values ({data_info_detailed})



-- Using valid and syntactically correct {dialect} SQL syntax, answer the following questions (check for typos, grammatical and spelling errors and fix them) with the information for '{table_name}' provided above; for final SQL only use column names from the CREATE TABLE (Do not query for columns that do not exist).


-- Using reference for TABLES '{table_name}' {context}; {question_txt}?

SELECT"""

# https://colab.research.google.com/drive/13BIKsqHnPOBcQ-ba2p77L5saiepTIwu0#scrollTo=0eI-VpCkf-fN
STARCODER2_PROMPT = """
### Instructions:
Your task is convert a question into a valid {dialect} syntax SQL query, given a {dialect} database schema. Let's work this out step by step to be sure we have the right answer.
Only use the column names from the CREATE TABLE statement.
Adhere to these rules:
- **Deliberately go through the question and database schema word by word** to appropriately answer the question
- **Use Table Aliases** to prevent ambiguity. For example, `SELECT table1.col1, table2.col1 FROM table1 JOIN table2 ON table1.id = table2.id`.
- Only use supplied table names: **{table_name}** for generation
- Only use column names from the CREATE TABLE statement: **{column_info}** for generation
- Avoid overly complex SQL queries, favor concise human readable SQL queries
- Avoid patterns that might be vulnerable to SQL injection, e.g. sanitize inputs
- When creating a ratio, always cast the numerator as float
- Always use COUNT(1) instead of COUNT(*)
- If the question is asking for a rate, use COUNT to compute percentage
- Avoid using the WITH statement
- Don't use aggregate and window function together
- Prefer NOT EXISTS to LEFT JOIN ON null id
- When using DESC keep NULLs at the end
- If JSONB format found in Table schema, do pattern matching on keywords from the question and use SQL functions such as ->> or ->
- Use prepared statements with parameterized queries to prevent SQL injection


### Input:
For SQL TABLE '{table_name}' with sample question/answer pairs,\n({sample_queries}), create a valid SQL (dialect:{dialect}) query to answer the following question:\n{question_txt}.
This query will run on a database whose schema is represented in this string:
CREATE TABLE '{table_name}' ({column_info}
);

-- Table '{table_name}', {context}, has sample values ({data_info_detailed})

### Response:
SELECT"""


RECOMMENDATION_PROMPT="""
Generate {n_questions} simple questions for the given dataset.
Only use the specified column names mentioned in *Data Schema*.

### Data Schema:
{data_schema}


Output: ordered numeric list of questions


### Response:
1.
"""
