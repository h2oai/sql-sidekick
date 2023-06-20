TASK_PROMPT = {
    "system_prompt": "Act as a Data Analyst",
    "user_prompt": """
        ### For table {_table_name}, given an input *Question*, let's work it out in a detailed step by step way and only return specific, detailed and informative tasks as an ordered numeric list for SQL generation to be sure we have the right answer.
        Use values that are explicitly mentioned in the *Question*.
        Use the *History* and *Context* section for co-reference and to infer relationships. *Context* contains entity mapping containing keys:values.
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
                # For table {_table_name}, given an input *Question*, only generate syntactically correct SQL queries.
                # Let's work it out in a detailed step by step way using the reasoning from *Tasks* section.
                # Pick the SQL query which has the highest average log probability if more than one result is likely to answer the
                candidate *Question*.
                ### {_dialect} SQL tables
                ### *Data:* \nFor table {_table_name} schema info is mentioned below,\n{_data_info}
                ### *History*:\n{_sample_queries}
                ### *Question*: For table {_table_name}, {_question}
                # SELECT 1
                ### *Tasks for table {_table_name}*:\n{_tasks}
                ### *Policies for SQL generation*:
                # Avoid overly complex SQL queries
                # Use values that are explicitly mentioned in the question.
                # Don't use aggregate and window function together
                # Use COUNT(1) instead of COUNT(*)
                # Return with LIMIT 100
                # Prefer NOT EXISTS to LEFT JOIN ON null id
                # Avoid using the WITH statement
                # When using DESC keep NULLs at the end
                # If JSONB format found in Table schema, do pattern matching on keywords from the question and use SQL functions such as ->> or ->
                # Add explanation and reasoning for each SQL query
            """

DEBUGGING_PROMPT = {
    "system_prompt": "Act as a SQL expert for PostgreSQL code",
    "user_prompt": """
                ### Fix syntax errors for provided SQL Query.
                # Add ``` as prefix and ``` as suffix to generated SQL
                # Error: {ex_traceback}
                # Add explanation and reasoning for each SQL query
                Query:\n {qry_txt}
                """,
}
