TASK_PROMPT = {
    "system_prompt": "Act as a Data Analyst",
    "user_prompt": """
        ### For Table: {_table_name} Given an input *Question*, only return specific and informative tasks as an ordered itemized list for SQL generation that answer the question.
        Extract all of the proper nouns (generally capitalized, abbreviated) from the Samples section and add to Context section as Key, Value pair.
        Use the Context section and Samples section to establish relationship when tokens from Question does not match column names.
        If information is not found in Context or Samples section, attempt to reason for possible tasks but also ask questions for.
        Infer the return type of the Question. Do not generate final SQL response, only return tasks.
        # Data information: \n{_data_info}
        # Samples: \n{_sample_queries}
        # Context: {_context}
        # *Question*: {_question_str};
        # Output: Tasks: ordered list of tasks
    """,
}

# Few shot learning prompt
## Task Description
## Examples
## Prompt
# Reference: https://arxiv.org/pdf/2005.14165.pdf
QUERY_PROMPT = """
                ### System: Act as a SQL Expert
                # Given an input *Question*, only generate syntactically correct SQL queries using step by step reasoning from Tasks section.
                # Extract all of the proper nouns (generally capitalized, abbreviated) from the Examples section and add to Context section as Key, Value pair.
                # Use the context section to establish relationship when tokens from Question does not match column names.
                # Pick the SQL query which has the highest average log probability of explaining the
                candidate question.
                ### {dialect} SQL tables
                Examples:\n{_sample_queries}
                ### *Question*: {_question};
                # SELECT 1
                ### Tasks:\n{_tasks}
                ### Context: {_context}
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
