TASK_PROMPT = {
    "system_prompt": "Act as a Data Analyst",
    "user_prompt": """
        ### For Table: {table_name} Given an input (question), only return specific tasks as an ordered itemized list for SQL generation that answer the question. Don't generate SQL code.
        Infer the return type of the question. Add a task to return final output.
        # Example data: \n {samples}
        # (question):\n {question_str}
        # Output format: Tasks: list of tasks
    """,
}

# Few shot learning prompt
## Task Description
## Examples
## Prompt
# Reference: https://arxiv.org/pdf/2005.14165.pdf
QUERY_PROMPT = """
                ### System: Act as a SQL Expert
                # Given an input (question), only generate syntactically correct SQL queries
                # Pick the SQL query which has the highest average log probability of explaining the
                candidate question
                ### {dialect} SQL tables
                Examples:\n{_sample_queries}
                ### *question*:\n{_question};
                # SELECT 1
                ### Tasks:\n{_tasks}
                ### Suggestions:
                # Don't use aggregate and window function together;
                # Avoid COUNT(*) and prefer COUNT(1);
                # Return with LIMIT 100
                # Prefer NOT EXISTS to LEFT JOIN ON null id;
                # Avoid using the WITH statement;
                # When using DESC keep NULLs at the end
                # If JSONB format found in Table schema, do pattern matching on keywords from the question
                # Add explanation
            """

DEBUGGING_PROMPT = {
    "system_prompt": "Act as a SQL expert for PostgreSQL code",
    "user_prompt": """
                ### Fix syntax errors for provided SQL Query.
                # Add ``` as prefix and ``` as suffix to generated SQL
                # Error: {ex_traceback}
                Query:\n {qry_txt}
                """,
}
