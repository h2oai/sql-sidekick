TASK_PROMPT = {
    "system_prompt": "Act as a Data Analyst",
    "user_prompt": """
        ### Given an input *Question*, only return specific and informative tasks as an ordered numeric list for SQL generation that answer the question.
        Use the *History* and *Context* section for co-reference and to infer relationships.
        If the words in the *Question* do not match column names *Data* section; Search for them in *Context* section.
        Always use *Context* with highest similarity score with the *Question*.
        If no information related to the *Question* is found; attempt to predict and reason for possible tasks.
        Infer the return type of the Question. Do not generate final complete SQL response, only return tasks.
        # *Data:* \nFor table {_table_name} schema info is mentioned below,\n{_data_info}
        # *History*: \n{_sample_queries}
        # *Question*: For Table {_table_name}, {_question_str}, *Context*: {_context}
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
                # Given an input *Question*, only generate syntactically correct SQL queries using step by step reasoning from *Tasks* section.
                # Pick the SQL query which has the highest average log probability of explaining the
                candidate *Question*.
                ### {dialect} SQL tables
                ### *History*:\n{_sample_queries}
                ### *Question*: {_question}
                # SELECT 1
                ### *Tasks*:\n{_tasks}
                ### *Policies for SQL generation*:
                # Avoid overly complex SQL queries
                # Don't use aggregate and window function together
                # Use COUNT(1) instead of COUNT(*)
                # Return with LIMIT 100
                # Prefer NOT EXISTS to LEFT JOIN ON null id
                # Avoid using the WITH statement
                # When using DESC keep NULLs at the end
                # If JSONB format found in Table schema, do pattern matching on keywords from the question
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
