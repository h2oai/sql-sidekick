task_prompt_template = {
        "system_prompt": "Act as a Data Analyst",
        "user_prompt": """
        ### For Table: {table_name} Given an input question, only list specific tasks itemized for SQL generation that answer the question. Don't generate SQL code.
        Infer the return type of the question. Add a step for final output.
        # Table schema:\n {schema_info}
        # Example data: \n {samples}
        # question:\n {question_str}
    """
}

query_prompt_template = """
                ### System: Act as a SQL Expert
                # Given an input question, only generate syntactically correct SQL queries
                # Pick the SQL query which has the highest average log probability of explaining the
                candidate question
                ### {dialect} SQL tables
                Examples:\n
                {_sample_queries}
                ### question: {question_format};
                # SELECT 1
                ### Tasks:\n
                {_tasks}
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

debugging_prompt_template = {
                "system_prompt": "Act as a SQL expert for PostgreSQL code",
                "user_prompt": """
                ### Fix syntax errors for provided SQL Query.
                # Add <start_code> as prefix and <end_code> as suffix to generated SQL
                # Error: {ex_traceback}
                Query:\n {qry_txt}
                """
}
