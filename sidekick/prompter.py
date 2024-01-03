import gc
import json
import os
from pathlib import Path
from typing import Optional

import click
import openai
import pandas as pd
import sqlparse
import toml
import torch
from colorama import Back as B
from colorama import Fore as F
from colorama import Style
from pandasql import sqldf
from sidekick.db_config import DBConfig
from sidekick.logger import logger
from sidekick.memory import EntityMemory
from sidekick.query import SQLGenerator
from sidekick.schema_generator import generate_schema
from sidekick.utils import (_execute_sql, check_vulnerability,
                            execute_query_pd, extract_table_names,
                            generate_suggestions, save_query, setup_dir)

__version__ = "0.1.8"

# Load the config file and initialize required paths
app_base_path = (Path(__file__).parent / "../").resolve()
# Below check is to handle the case when the app is running on the h2o.ai cloud or locally
default_base_path = app_base_path if os.path.isdir("./.sidekickvenv/bin/") else "/meta_data"
env_settings = toml.load(f"{app_base_path}/sidekick/configs/env.toml")
db_dialect = env_settings["DB-DIALECT"]["DB_TYPE"]
model_name = env_settings["MODEL_INFO"]["MODEL_NAME"]
h2o_remote_url = env_settings["MODEL_INFO"]["RECOMMENDATION_MODEL_REMOTE_URL"]
h2o_key = env_settings["MODEL_INFO"]["RECOMMENDATION_MODEL_API_KEY"]

os.environ["TOKENIZERS_PARALLELISM"] = "False"


def color(fore="", back="", text=None):
    return f"{fore}{back}{text}{Style.RESET_ALL}"


msg = """Welcome to the SQL Sidekick!\nI am an AI assistant that helps you with SQL queries.
I can help you with the following:\n
1. Configure a local database(for schema validation and syntax checking): `sql-sidekick configure db-setup`.\n
2. Learn contextual query/answer pairs: `sql-sidekick learn add-samples`.\n
3. Simply add context: `sql-sidekick learn update-context`.\n
4. Ask a question: `sql-sidekick query`.
"""


@click.group(help=msg)
@click.version_option("-V", "--version", message=f"sql-sidekick - {__version__}")
def cli():
    # Book-keeping
    setup_dir(default_base_path)


@cli.group("configure")
def configure():
    """Helps in configuring local database."""


def enter_table_name():
    val = input(color(F.GREEN, "", "Would you like to create a table for the database? (y/n): "))
    return val


def enter_file_path(table: str):
    val = input(color(F.GREEN, "", f"Please input the CSV file path to table {table} : "))
    return val


@configure.command("log", help="Adjust log settings")
@click.option("--set_level", "-l", help="Set log level (Default: INFO)")
def set_loglevel(set_level):
    env_settings["LOGGING"]["LOG-LEVEL"] = set_level
    # Update settings file for future use.
    f = open(f"{default_base_path}/sidekick/configs/env.toml", "w")
    toml.dump(env_settings, f)
    f.close()


def _get_table_info(cache_path: str, table_name: str = None):
    # Search for the file in the default current path, if not present ask user to enter the path
    if Path(f"{cache_path}/{table_name}_table_info.jsonl").exists():
        table_info_path = f"{cache_path}/{table_name}_table_info.jsonl"  # input schema in jsonl format
    else:
        # Search for table related meta data in tables.json
        # TODO: In future, metadata could be pushed on to a Db.
        if Path(f"{cache_path}/tables.json").exists():
            f = open(f"{cache_path}/tables.json", "r")
            table_metadata = json.load(f)
            current_meta = table_metadata[table_name]
            if "schema_info_path" in current_meta:
                table_info_path = current_meta["schema_info_path"]
                if table_info_path is None:
                    # if table_info_path is None, generate default schema n set path
                    data_path = current_meta["samples_path"]
                    _, table_info_path = generate_schema(data_path, f"{cache_path}/{table_name}_table_info.jsonl")
        table_metadata = {"schema_info_path": table_info_path}
        with open(f"{cache_path}/table_context.json", "w") as outfile:
            json.dump(table_metadata, outfile, indent=4, sort_keys=False)
    return table_info_path


def update_table_info(cache_path: str, table_info_path: str = None, table_name: str = None):
    if Path(f"{cache_path}/table_context.json").exists():
        f = open(f"{cache_path}/table_context.json", "r")
        table_metadata = json.load(f)
        if table_name:
            table_metadata["tables_in_use"] = [table_name]
        if table_info_path:
            table_metadata["schema_info_path"] = table_info_path
    else:
        table_metadata = dict()
        if table_name:
            table_metadata["tables_in_use"] = [table_name]
        if table_info_path:
            table_metadata["schema_info_path"] = table_info_path

    table_metadata["data_table_map"] = {}
    with open(f"{cache_path}/table_context.json", "w") as outfile:
        json.dump(table_metadata, outfile, indent=4, sort_keys=False)

# Experimental, might change in future.
def recommend_suggestions(cache_path: str, table_name: str, n_qs: int=10):
    column_names = []
    if cache_path is None:
        path = f"{default_base_path}/var/lib/tmp/data"
        logger.debug(f"Retrieve meta information for table {table_name}")
        cache_path = _get_table_info(path, table_name)
        logger.debug(f"Updated table info path: {cache_path}")
    if Path(cache_path).exists():
        with open(cache_path, "r") as in_file:
            for line in in_file:
                if line.strip():
                    data = json.loads(line)
                    if "Column Name" in data and "Column Type" in data:
                        col_name = data["Column Name"]
                        column_names.append(col_name)
    try:
        r_url = _key =  None
        # First check for keys in env variables
        logger.debug(f"Checking environment settings ...")
        env_url = os.environ["RECOMMENDATION_MODEL_REMOTE_URL"]
        env_key = os.environ["RECOMMENDATION_MODEL_API_KEY"]
        if env_url and env_key:
            r_url = env_url
            _key = env_key
        elif Path(f"{app_base_path}/sidekick/configs/env.toml").exists():
            # Reload .env info
            logger.debug(f"Checking configuration file ...")
            env_settings = toml.load(f"{app_base_path}/sidekick/configs/env.toml")
            r_url = env_settings["MODEL_INFO"]["RECOMMENDATION_MODEL_REMOTE_URL"]
            _key = env_settings["MODEL_INFO"]["RECOMMENDATION_MODEL_API_KEY"]
        else:
            raise Exception("Model url or key is missing.")

        result = generate_suggestions(remote_url=r_url, client_key=_key, column_names=column_names, n_qs=n_qs
                                      )
    except Exception as e:
        logger.error(f"Something went wrong, check the supplied credentials:\n{e}")
        result = None
    return result



@configure.command(
    "generate_schema", help=f"Helps generate default schema for the selected Database dialect: {db_dialect}"
)
@click.option("--data_path", default="data.csv", help="Enter the path of csv", type=str)
@click.option("--output_path", default="table_info.jsonl", help="Enter the path of generated schema in jsonl", type=str)
def generate_input_schema(data_path, output_path):
    _, o_path = generate_schema(data_path, output_path)
    click.echo(f"Schema generated for the input data at {o_path}")


@configure.command("db-setup", help=f"Enter information to configure {db_dialect} database locally")
@click.option("--db_name", "-n", default="querydb", help="Database name", prompt="Enter Database name")
@click.option("--hostname", "-h", default="localhost", help="Database hostname", prompt="Enter hostname name")
@click.option("--user_name", "-u", default=f"{db_dialect}", help="Database username", prompt="Enter username name")
@click.option(
    "--password",
    "-p",
    default="abc",
    hide_input=True,
    help="Database password",
    prompt="Enter password",
)
@click.option("--port", "-P", default=5432, help="Database port", prompt="Enter port (default 5432)")
@click.option("--table-info-path", "-t", help="Table info path", default=None)
def db_setup_cli(db_name: str, hostname: str, user_name: str, password: str, port: int, table_info_path: str):
    db_setup(
        db_name=db_name,
        hostname=hostname,
        user_name=user_name,
        password=password,
        port=port,
        table_info_path=table_info_path,
        table_samples_path=None,
        table_name=None,
        is_command=True,
    )


def db_setup(
    db_name: str,
    hostname: str,
    user_name: str,
    password: str,
    port: int,
    table_name: str,
    table_info_path: Optional[str] = None,
    table_schema: Optional[list] = None,
    table_samples_path: Optional[str] = None,
    add_sample: bool=True,
    is_command: bool = False,
    local_base_path: Optional[str] = None
):
    """Helps setup local database.
    Args:
        db_name (str): Database name.
        hostname (str): Hostname.
        user_name (str): Username.
        password (str): Password.
        port (int): Port.
        table_name (str): Table name.
        table_info_path (str): Table info path.
        table_schema (list): Table schema.
        table_samples_path (str): Table samples path.
        add_sample (bool): Add sample rows.
        is_command (bool): Is command line interface.
        local_base_path (str): Local base path.
    """
    click.echo(f" Information supplied:\n {db_name}, {hostname}, {user_name}, {password}, {port}")
    try:
        res = err = None
        # To-DO
        # --- Need to remove the below keys from ENV toml --- #
        # env_settings["TABLE_INFO"]["TABLE_INFO_PATH"] = table_info_path
        # env_settings["TABLE_INFO"]["TABLE_SAMPLES_PATH"] = table_samples_path

        # Update settings file for future use.
        # Check if the env.toml exists.
        env_config_fname = f"{app_base_path}/sidekick/configs/env.toml"
        if Path(env_config_fname).exists():
            env_settings["LOCAL_DB_CONFIG"]["HOST_NAME"] = hostname
            env_settings["LOCAL_DB_CONFIG"]["USER_NAME"] = user_name
            env_settings["LOCAL_DB_CONFIG"]["PASSWORD"] = password
            env_settings["LOCAL_DB_CONFIG"]["PORT"] = port
            env_settings["LOCAL_DB_CONFIG"]["DB_NAME"] = db_name
            f = open(env_config_fname, "w")
            toml.dump(env_settings, f)
            f.close()
        base_path = local_base_path if local_base_path else default_base_path
        path = f"{base_path}/var/lib/tmp/data"
        # For current session
        db_obj = DBConfig(db_name, hostname, user_name, password, port, base_path=base_path, dialect=db_dialect)

        # Create Database
        if db_obj.dialect == "sqlite" and not os.path.isfile(f"{base_path}/db/sqlite/{db_name}.db"):
            res, err = db_obj.create_db()
            click.echo("Database created successfully!")
        elif not db_obj.db_exists():
            res, err = db_obj.create_db()
            click.echo("Database created successfully!")
        else:
            click.echo("Database already exists!")

        # Create Table in DB
        val = enter_table_name() if is_command else "y"
        while True:
            if val.lower() != "y" and val.lower() != "n":
                click.echo("In-correct values. Enter Yes(y) or no(n)")
                val = enter_table_name()
            else:
                break

        if table_info_path is None and table_schema is None:
            logger.debug(f"Retrieve meta information for table {table_name}")
            table_info_path = _get_table_info(path, table_name)
            logger.debug(f"Updated table info path: {table_info_path}")

        if val.lower() == "y" or val.lower() == "yes":
            table_value = input("Enter table name: ") if is_command else table_name
            click.echo(f"Table name: {table_value}")
            # set table name
            db_obj.table_name = table_value.lower().replace(" ", "_")
            if table_schema:
                res, err = db_obj.create_table(schema_info=table_schema)
            else:
                if table_info_path:
                    res, err = db_obj.create_table(schema_info_path=table_info_path)

        update_table_info(path, table_info_path, db_obj.table_name)
        # Check if table exists; pending --> and doesn't have any rows
        # Add rows to table
        if db_obj.has_table():
            click.echo(f"Checked table {db_obj.table_name} exists in the DB.")
            val = (
                input(color(F.GREEN, "", "Would you like to add few sample rows (at-least 3)? (y/n):"))
                if is_command and not add_sample
                else "y"
            )
            val = "n" if not add_sample else "y"
            if val.lower().strip() == "y" or val.lower().strip() == "yes":
                val = input("Path to a CSV file to insert data from:") if is_command else table_samples_path
                res, err = db_obj.add_samples(val)
            else:
                click.echo("Exiting...")
                return None, err
        else:
            echo_msg = "Job done. Ask a question now!"
            click.echo(echo_msg)

        if err is None:
            click.echo(
                f"Created a Database {db_name}. Inserted sample values from {table_samples_path} into table {table_name}, please ask questions!"
            )
            return res, None
        else:
            return 0, err
    except Exception as e:
        error_msg = f"Error creating database. Check configuration parameters.\n: {e}"
        click.echo(error_msg)
        if not is_command:
            return 0, error_msg


@cli.group("learn")
def learn():
    """Helps in learning and building memory."""


def _add_context(entity_memory: EntityMemory):
    _FORMAT = '''# Add input Query and Response \n\n
"Query": "<any query>";\n
"Response": """<respective response>"""
'''
    res = click.edit(_FORMAT.replace("\t", ""))
    # Check if user has entered any value
    if res:
        try:
            _ = entity_memory.save_context(res)
        except ValueError as ve:
            logger.info(f"Not a valid input. Try again")


@learn.command("add-samples", help="Helps add contextual query/answer pairs.")
def add_query_response():
    em = EntityMemory(k=5, path=default_base_path)
    _add_context(em)
    _more = "y"
    while _more.lower() != "n" or _more.lower() != "no":
        _more = click.prompt("Would you like to add more samples? (y/n)")
        if _more.lower() == "y":
            _add_context(em)
        else:
            break


@learn.command("update-context", help="Update context in memory for future use")
def update_context():
    """Helps learn context for generation."""
    # Book-keeping
    setup_dir(default_base_path)

    context_dict = """{\n"<new_context_key>": "<new_context_value>"\n}
    """
    content_file_path = f"{default_base_path}/var/lib/tmp/data/context.json"
    context_str = context_dict
    if Path(f"{default_base_path}/var/lib/tmp/data/context.json").exists():
        context_dict = json.load(open(content_file_path, "r"))
        context_dict["<new_context_key>"] = "<new_context_value"
        context_str = json.dumps(context_dict, indent=4, sort_keys=True, default=str)

    updated_context = click.edit(context_str)
    if updated_context:
        context_dict = json.loads(updated_context)
        if "<new_context_key>" in context_dict:
            del context_dict["<new_context_key>"]
        path = f"{default_base_path}/var/lib/tmp/data/"
        with open(f"{path}/context.json", "w") as outfile:
            json.dump(context_dict, outfile, indent=4, sort_keys=False)
    else:
        logger.debug("No content updated ...")


@cli.command()
@click.option("--question", "-q", help="Database name", prompt="Ask a question")
@click.option("--table-info-path", "-t", help="Table info path", default=None)
@click.option("--sample_qna_path", "-s", help="Samples path", default=None)
def query(question: str, table_info_path: str, sample_qna_path: str):
    """Asks question and returns SQL."""
    ask(
        question=question,
        table_info_path=table_info_path,
        sample_queries_path=sample_qna_path,
        table_name=None,
        is_command=True,
    )

def data_preview(table_name):
    hostname = env_settings["LOCAL_DB_CONFIG"]["HOST_NAME"]
    user_name = env_settings["LOCAL_DB_CONFIG"]["USER_NAME"]
    password = env_settings["LOCAL_DB_CONFIG"]["PASSWORD"]
    port = env_settings["LOCAL_DB_CONFIG"]["PORT"]
    db_name = env_settings["LOCAL_DB_CONFIG"]["DB_NAME"]

    db_obj = DBConfig(
        db_name, hostname, user_name, password, port, base_path=default_base_path, dialect=db_dialect
    )
    if not db_obj.table_name:
        db_obj.table_name = table_name
    q_res = db_obj.data_preview(table_name)
    # Convert result to data-frame
    res = pd.DataFrame(q_res[0]) if q_res and q_res[0] else None
    return res

def ask(
    question: str,
    table_info_path: str,
    sample_queries_path: str,
    table_name: str,
    model_name: str = "h2ogpt-sql-nsql-llama-2-7B",
    db_dialect = "sqlite",
    execute_db_dialect="sqlite",
    is_regenerate: bool = False,
    is_regen_with_options: bool = False,
    is_command: bool = False,
    execute_query: bool = True,
    debug_mode: bool = False,
    self_correction: bool = True,
    local_base_path = None,
):
    """Ask a question and returns generate SQL.
    Args:
        question (str): Question to ask.
        table_info_path (str): Path to table info.
        sample_queries_path (str): Path to sample queries.
        table_name (str): Table name.
        model_name (str): Model name.
        db_dialect (str): Database dialect.
        execute_db_dialect (str): Database dialect to execute.
        is_regenerate (bool): Regenerate SQL.
        is_regen_with_options (bool): Regenerate SQL with options.
        is_command (bool): Is command line interface.
        execute_query (bool): Execute SQL.
        debug_mode (bool): Debug mode.
        self_correction (bool): Self correction.
        local_base_path (str): Local base path.

    Returns:
        list: List of results.
    """

    results = []
    err = None  # TODO - Need to handle errors if occurred
    # Book-keeping
    base_path = local_base_path if local_base_path else default_base_path
    setup_dir(base_path)

    # Check if table exists
    path = f"{base_path}/var/lib/tmp/data"
    table_context_file = f"{path}/table_context.json"
    table_context = json.load(open(table_context_file, "r")) if Path(table_context_file).exists() else {}
    table_names = []

    if table_name is not None:
        table_names = [table_name.lower().replace(" ", "_")]
    elif table_context and "tables_in_use" in table_context:
        _tables = table_context["tables_in_use"]
        table_names = [_t.lower().replace(" ", "_") for _t in _tables]
    else:
        # Ask for table name only when more than one table exists.
        table_names = [click.prompt("Which table to use?")]
        table_context["tables_in_use"] = [_t.lower().replace(" ", "_") for _t in table_names]
        with open(f"{path}/table_context.json", "w") as outfile:
            json.dump(table_context, outfile, indent=4, sort_keys=False)
    logger.info(f"Table in use: {table_names}")
    logger.info(f"SQL dialect for generation: {db_dialect}")
    # Check if env.toml file exists
    api_key = os.getenv("OPENAI_API_KEY", None)
    if (model_name == 'gpt-3.5-turbo-0301' or model_name == 'gpt-3.5-turbo-1106') and api_key is None:
        api_key = env_settings["MODEL_INFO"]["OPENAI_API_KEY"]
        if api_key is None:
            if is_command:
                val = input(
                    color(
                        F.GREEN, "", "Looks like API key is not set, would you like to set OPENAI_API_KEY? (y/n):"
                    )
                )
                if val.lower() == "y":
                    api_key = input(color(F.GREEN, "", "Enter OPENAI_API_KEY :"))

            if api_key is None and is_command:
                return ["Looks like API key is not set, please set OPENAI_API_KEY!"], err

            if os.getenv("OPENAI_API_KEY", None) is None:
                os.environ["OPENAI_API_KEY"] = api_key
            env_settings["MODEL_INFO"]["OPENAI_API_KEY"] = api_key

            # Update settings file for future use.
            f = open(f"{app_base_path}/sidekick/configs/env.toml", "w")
            toml.dump(env_settings, f)
            f.close()
    if model_name:
        if 'gpt-3.5' in model_name or 'gpt-4' in model_name:
            openai.api_key = api_key
            logger.info(f"OpenAI key found.")
    sql_g = None
    try:
        # Set context
        logger.info("Setting context...")
        logger.info(f"Question: {question}")
        # Get updated info from env.toml
        host_name = env_settings["LOCAL_DB_CONFIG"]["HOST_NAME"]
        user_name = env_settings["LOCAL_DB_CONFIG"]["USER_NAME"]
        passwd = env_settings["LOCAL_DB_CONFIG"]["PASSWORD"]
        db_name = env_settings["LOCAL_DB_CONFIG"]["DB_NAME"]

        if execute_db_dialect.lower() == "sqlite":
            db_url = f"sqlite:///{base_path}/db/sqlite/{db_name}.db"
        elif execute_db_dialect.lower() == "postgresql":
            db_url = f"{execute_db_dialect}+psycopg2://{user_name}:{passwd}@{host_name}/{db_name}".format(
                user_name, passwd, host_name, db_name
            )
        else:
            db_url = None

        if table_info_path is None:
            table_info_path = _get_table_info(path, table_name)
        logger.debug(f"Table info path: {table_info_path}")

        sql_g = SQLGenerator(
            db_url,
            api_key,
            model_name=model_name,
            job_path=base_path,
            data_input_path=table_info_path,
            sample_queries_path=sample_queries_path,
            is_regenerate_with_options=is_regen_with_options,
            is_regenerate=is_regenerate,
            db_dialect=db_dialect,
            debug_mode=debug_mode,
        )
        if model_name and "h2ogpt-sql" not in model_name and not _execute_sql(question):
            sql_g._tasks = sql_g.generate_tasks(table_names, question)
            results.extend(["I am thinking step by step: \n", sql_g._tasks, "\n"])
            click.echo(sql_g._tasks)

            updated_tasks = None
            if sql_g._tasks is not None and is_command:
                edit_val = click.prompt("Would you like to edit the tasks? (y/n)")
                if edit_val.lower() == "y":
                    updated_tasks = click.edit(sql_g._tasks)
                    click.echo(f"Tasks:\n {updated_tasks}")
                else:
                    click.echo("Skipping edit...")
            if updated_tasks is not None:
                sql_g._tasks = updated_tasks
        alt_res = None
        # The interface could also be used to simply execute user provided SQL
        # Keyword: "Execute SQL: <SQL query>"
        if (
            question is not None
            and "select" in question.lower()
            and (question.lower().startswith("question:") or question.lower().startswith("q:"))
        ):
            _q = question.lower().split("q:")[1].split("r:")[0].strip()
            res = question.lower().split("r:")[1].strip()
            question = _q
        elif _execute_sql(question) and debug_mode:
            logger.info("Executing user provided SQL without generation...")
            res = question.strip().lower().split("execute sql:")[1].strip()
        else:
            logger.info("Computing user request ...")
            _check_cond = question.strip().lower().split("execute sql:")
            if len(_check_cond) > 1:
                question = _check_cond[1].strip()
            res, alt_res = sql_g.generate_sql(table_names, question, model_name=model_name)
        logger.info(f"Input query: {question}")
        logger.info(f"Generated response:\n\n{res}")

        if res is not None:
            updated_sql = None
            res_val = "e"
            if is_command:
                while res_val.lower() in ["e", "edit", "r", "regenerate"]:
                    res_val = click.prompt(
                        "Would you like to 'edit' or 'regenerate' the SQL? Use 'e' to edit or 'r' to regenerate. "
                        "To skip, enter 's' or 'skip'"
                    )
                    if res_val.lower() == "e" or res_val.lower() == "edit":
                        updated_sql = click.edit(res)
                        click.echo(f"Updated SQL:\n {updated_sql}")
                    elif res_val.lower() == "r" or res_val.lower() == "regenerate":
                        click.echo("Attempting to regenerate...")
                        res, alt_res = sql_g.generate_sql(
                            table_names, question, model_name=model_name, _dialect=db_dialect
                        )
                        res = res.replace("“", '"').replace("”", '"')
                        [res := res.replace(s, '"') for s in "‘`’'" if s in res]
                        logger.info(f"Input query: {question}")
                        logger.info(f"Generated response:\n\n{res}")
            pretty_sql = sqlparse.format(res, reindent=True, keyword_case="upper")
            syntax_highlight = f"""``` sql\n{pretty_sql}\n```\n\n"""
            results.extend([f"**Generated response for question,**\n{question}\n", syntax_highlight, "\n"])
            logger.info(f"Alternate responses:\n\n{alt_res}")

            exe_sql = "y"
            if not execute_query:
                if is_command:
                    exe_sql = click.prompt("Would you like to execute the generated SQL (y/n)?")
                else:
                    exe_sql = "n"

            _val = updated_sql if updated_sql else res
            if exe_sql.lower() == "y" or exe_sql.lower() == "yes":
                # For the time being, the default option is Pandas, but the user can be asked to select Database or pandas DF later.
                logger.info(f"Checking for vulnerabilities in the provided SQL: {_val}")
                r, m = check_vulnerability(question)
                q_res = m if r else None
                option = "DB"  # or DB
                if option == "DB" and not r:
                    hostname = env_settings["LOCAL_DB_CONFIG"]["HOST_NAME"]
                    user_name = env_settings["LOCAL_DB_CONFIG"]["USER_NAME"]
                    password = env_settings["LOCAL_DB_CONFIG"]["PASSWORD"]
                    port = env_settings["LOCAL_DB_CONFIG"]["PORT"]
                    db_name = env_settings["LOCAL_DB_CONFIG"]["DB_NAME"]

                    #TODO This call maybe redundant n might need some cleaning
                    db_obj = DBConfig(
                        db_name, hostname, user_name, password, port, base_path=base_path, dialect=db_dialect
                    )

                    # Before executing, check if known vulnerabilities exist in the generated SQL code.
                    _val = _val.replace("“", '"').replace("”", '"')
                    [_val := _val.replace(s, '"') for s in "‘`’'" if s in _val]

                    q_res, err = db_obj.execute_query(query=_val)
                    # Check for runtime/operational errors n attempt auto-correction
                    count = 0
                    if self_correction and err and 'OperationalError' in err:
                        logger.info("Attempting to auto-correct the query...")
                        while count !=2 and err and 'OperationalError' in err:
                            try:
                                logger.debug(f"Attempt: {count+1}")
                                _err = err.split("\n")[0].split("Error occurred :")[1]
                                env_url = os.environ["RECOMMENDATION_MODEL_REMOTE_URL"]
                                env_key = os.environ["RECOMMENDATION_MODEL_API_KEY"]
                                corr_sql =  sql_g.self_correction(input_prompt=_val, error_msg=_err, remote_url=env_url, client_key=env_key)
                                q_res, err = db_obj.execute_query(query=corr_sql)
                                count += 1
                            except Exception as e:
                                logger.error(f"Something went wrong, check the supplied credentials:\n{e}")
                                count += 1
                    if m:
                        _t = "\nWarning:\n".join([str(q_res), m])
                        q_res = _t
                elif option == "pandas":
                    tables = extract_table_names(_val)
                    tables_path = dict()
                    if Path(f"{path}/table_context.json").exists():
                        f = open(f"{path}/table_context.json", "r")
                        table_metadata = json.load(f)
                        for table in tables:
                            # Check if the local table_path exists in the cache
                            if table not in table_metadata["data_table_map"].keys():
                                val = enter_file_path(table)
                                if not os.path.isfile(val):
                                    click.echo("In-correct Path. Please enter again! Yes(y) or no(n)")
                                else:
                                    tables_path[table] = val
                                    table_metadata["data_table_map"][table] = val
                                    break
                            else:
                                tables_path[table] = table_metadata["data_table_map"][table]
                        assert len(tables) == len(tables_path)
                        with open(f"{path}/table_context.json", "w") as outfile:
                            json.dump(table_metadata, outfile, indent=4, sort_keys=False)
                    try:
                        q_res = execute_query_pd(query=_val, tables_path=tables_path, n_rows=100)
                        click.echo(f"The query results are:\n {q_res}")
                    except sqldf.PandaSQLException as e:
                        logger.error(f"Error in executing the query: {e}")
                        click.echo("Error in executing the query. Validate generated SQL and try again.")
                        click.echo("No result to display.")

                results.append("**Result:** \n")
                if q_res:
                    click.echo(f"The query results are:\n {q_res}")
                    results.extend([str(q_res), "\n"])
                else:
                    click.echo(f"While executing query:\n {err}")
                    results.extend([str(err), "\n"])

            save_sql = click.prompt("Would you like to save the generated SQL (y/n)?") if is_command else "n"
            if save_sql.lower() == "y" or save_sql.lower() == "yes":
                # Persist for future use
                _val = updated_sql if updated_sql else res
                save_query(base_path, query=question, response=_val)
            else:
                click.echo("Exiting...")
        else:
            results = ["I was not able to generate a response for the question. Please try re-phrasing."]
            alt_res, err = None, None
    except (MemoryError, RuntimeError, AttributeError) as e:
        logger.error(f"Something went wrong while generating response: {e}")
        if sql_g:
            del sql_g
        gc.collect()
        torch.cuda.empty_cache()
        alt_res, err = None, e
        results = ["Something went wrong while generating response. Please check the supplied API Keys and try again."]
    return results, alt_res, err


if __name__ == "__main__":
    cli()
