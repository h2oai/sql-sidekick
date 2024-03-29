# Reference: https://github.com/openai/openai-cookbook/blob/main/examples/Backtranslation_of_SQL_queries.py
question_query_samples = """
{
    "question": "{}",
    "query": "{}"
}
"""

schema_info_template = {"Column Name": "", "Column Type": "", "Sample Values": []}

data_samples_template = "'{column_name}' contains values similar to {comma_separated_sample_values}."
