import json
import os
from pathlib import Path

import click
import openai
import toml
from colorama import Back as B
from colorama import Fore as F
from colorama import Style
from loguru import logger
from pandasql import sqldf
from sidekick.db_config import DBConfig
from sidekick.memory import EntityMemory
from sidekick.query import SQLGenerator
from sidekick.utils import (execute_query_pd, extract_table_names, save_query,
                            setup_dir)

# Load the config file and initialize required paths
base_path = (Path(__file__).parent / "../").resolve()
env_settings = toml.load(f"{base_path}/sidekick/configs/env.toml")
db_dialect = env_settings["DB-DIALECT"]["DB_TYPE"]
model_name = env_settings["MODEL_INFO"]["MODEL_NAME"]
os.environ["TOKENIZERS_PARALLELISM"] = "False"
__version__ = "0.0.4"


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
    setup_dir(base_path)


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
    f = open(f"{base_path}/sidekick/configs/env.toml", "w")
    toml.dump(env_settings, f)
    f.close()


def _get_table_info(cache_path: str):
    # Search for the file in the default current path, if not present ask user to enter the path
    if Path(f"{cache_path}/table_info.jsonl").exists():
        table_info_path = f"{cache_path}/table_info.jsonl"
    else:
        # Check in table cache before requesting
        if Path(f"{cache_path}/table_context.json").exists():
            f = open(f"{cache_path}/table_context.json", "r")
            table_metadata = json.load(f)
            if "schema_info_path" in table_metadata:
                table_info_path = table_metadata["schema_info_path"]
            else:
                table_info_path = click.prompt("Enter table info path")
                table_metadata["schema_info_path"] = table_info_path
                with open(f"{cache_path}/table_context.json", "w") as outfile:
                    json.dump(table_metadata, outfile, indent=4, sort_keys=False)
        else:
            table_info_path = click.prompt("Enter table info path")
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
def db_setup(db_name: str, hostname: str, user_name: str, password: str, port: int, table_info_path: str):
    db_setup_api(
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


def db_setup_api(
    db_name: str,
    hostname: str,
    user_name: str,
    password: str,
    port: int,
    table_info_path: str,
    table_samples_path: str,
    table_name: str,
    is_command: bool = False,
):
    """Creates context for the new Database"""
    click.echo(f" Information supplied:\n {db_name}, {hostname}, {user_name}, {password}, {port}")
    try:
        res = None
        err = None
        env_settings["LOCAL_DB_CONFIG"]["HOST_NAME"] = hostname
        env_settings["LOCAL_DB_CONFIG"]["USER_NAME"] = user_name
        env_settings["LOCAL_DB_CONFIG"]["PASSWORD"] = password
        env_settings["LOCAL_DB_CONFIG"]["PORT"] = port
        env_settings["LOCAL_DB_CONFIG"]["DB_NAME"] = db_name

        # To-DO
        # --- Need to remove the below keys from ENV toml --- #
        # env_settings["TABLE_INFO"]["TABLE_INFO_PATH"] = table_info_path
        # env_settings["TABLE_INFO"]["TABLE_SAMPLES_PATH"] = table_samples_path

        # Update settings file for future use.
        f = open(f"{base_path}/sidekick/configs/env.toml", "w")
        toml.dump(env_settings, f)
        f.close()
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

        if table_info_path is None:
            table_info_path = _get_table_info(path)

        if val.lower() == "y" or val.lower() == "yes":
            table_value = input("Enter table name: ") if is_command else table_name
            click.echo(f"Table name: {table_value}")
            # set table name
            db_obj.table_name = table_value.replace(" ", "_")
            res, err = db_obj.create_table(table_info_path)

        update_table_info(path, table_info_path, db_obj.table_name)
        # Check if table exists; pending --> and doesn't have any rows
        # Add rows to table
        if db_obj.has_table():
            click.echo(f"Checked table {db_obj.table_name} exists in the DB.")
            val = (
                input(color(F.GREEN, "", "Would you like to add few sample rows (at-least 3)? (y/n):"))
                if is_command
                else "y"
            )
            if val.lower().strip() == "y" or val.lower().strip() == "yes":
                val = input("Path to a CSV file to insert data from:") if is_command else table_samples_path
                res, err = db_obj.add_samples(val)
            else:
                click.echo("Exiting...")
                return
        else:
            echo_msg = "Job done. Ask a question now!"
            click.echo(echo_msg)

        if err is None:
            return f"Created a Database {db_name}. Inserted sample values from {table_samples_path} into table {table_name}, please ask questions!", None
        else:
            return None, err
    except Exception as e:
        echo_msg = f"Error creating database. Check configuration parameters.\n: {e}"
        click.echo(echo_msg)
        if not is_command:
            return echo_msg


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
    em = EntityMemory(k=5, path=base_path)
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
    setup_dir(base_path)

    context_dict = """{\n"<new_context_key>": "<new_context_value>"\n}
    """
    content_file_path = f"{base_path}/var/lib/tmp/data/context.json"
    context_str = context_dict
    if Path(f"{base_path}/var/lib/tmp/data/context.json").exists():
        context_dict = json.load(open(content_file_path, "r"))
        context_dict["<new_context_key>"] = "<new_context_value"
        context_str = json.dumps(context_dict, indent=4, sort_keys=True, default=str)

    updated_context = click.edit(context_str)
    if updated_context:
        context_dict = json.loads(updated_context)
        if "<new_context_key>" in context_dict:
            del context_dict["<new_context_key>"]
        path = f"{base_path}/var/lib/tmp/data/"
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
    query_api(question=question, table_info_path=table_info_path, sample_queries_path=sample_qna_path, table_name=None, is_command=True)


def query_api(question: str, table_info_path: str, sample_queries_path: str, table_name:str, is_command: bool = False):
    """Asks question and returns SQL."""
    results = []
    err = None  # TODO - Need to handle errors if occurred
    # Book-keeping
    setup_dir(base_path)

    # Check if table exists
    path = f"{base_path}/var/lib/tmp/data"
    table_context_file = f"{path}/table_context.json"
    table_context = json.load(open(table_context_file, "r")) if Path(table_context_file).exists() else {}
    table_names = []

    if table_name is not None:
        table_names = [table_name]
    elif table_context and "tables_in_use" in table_context:
        _tables = table_context["tables_in_use"]
        table_names = [_t.replace(" ", "_") for _t in _tables]
    else:
        # Ask for table name only when more than one table exists.
        table_names = [click.prompt("Which table to use?")]
        table_context["tables_in_use"] = [_t.replace(" ", "_") for _t in table_names]
        with open(f"{path}/table_context.json", "w") as outfile:
            json.dump(table_context, outfile, indent=4, sort_keys=False)
    logger.info(f"Table in use: {table_names}")
    # Check if env.toml file exists
    api_key = None
    if model_name != "h2ogpt-sql":
        api_key = env_settings["MODEL_INFO"]["OPENAI_API_KEY"]
        if api_key is None or api_key == "":
            if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
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

            os.environ["OPENAI_API_KEY"] = api_key
            env_settings["MODEL_INFO"]["OPENAI_API_KEY"] = api_key

            # Update settings file for future use.
            f = open(f"{base_path}/sidekick/configs/env.toml", "w")
            toml.dump(env_settings, f)
            f.close()
        openai.api_key = api_key

    # Set context
    logger.info("Setting context...")
    logger.info(f"Question: {question}")
    # Get updated info from env.toml
    host_name = env_settings["LOCAL_DB_CONFIG"]["HOST_NAME"]
    user_name = env_settings["LOCAL_DB_CONFIG"]["USER_NAME"]
    passwd = env_settings["LOCAL_DB_CONFIG"]["PASSWORD"]
    db_name = env_settings["LOCAL_DB_CONFIG"]["DB_NAME"]

    if db_dialect == "sqlite":
        db_url = f"sqlite:///{base_path}/db/sqlite/{db_name}.db"
    else:
        db_url = f"{db_dialect}+psycopg2://{user_name}:{passwd}@{host_name}/{db_name}".format(
            user_name, passwd, host_name, db_name
        )

    if table_info_path is None:
        table_info_path = _get_table_info(path)

    sql_g = SQLGenerator(
        db_url, api_key, job_path=base_path, data_input_path=table_info_path, sample_queries_path=sample_queries_path
    )
    if "h2ogpt-sql" not in model_name:
        sql_g._tasks = sql_g.generate_tasks(table_names, question)
        results.extend(["List of Actions Generated: \n", sql_g._tasks, "\n"])
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

    res = sql_g.generate_sql(table_names, question, model_name=model_name, _dialect=db_dialect)
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
                    res = sql_g.generate_sql(table_names, question, model_name=model_name, _dialect=db_dialect)
                    logger.info(f"Input query: {question}")
                    logger.info(f"Generated response:\n\n{res}")

        results.extend(["Generated Query:\n", res, "\n"])

        exe_sql = click.prompt("Would you like to execute the generated SQL (y/n)?") if is_command else "y"
        if exe_sql.lower() == "y" or exe_sql.lower() == "yes":
            # For the time being, the default option is Pandas, but the user can be asked to select Database or pandas DF later.
            q_res = None
            option = "DB"  # or DB
            _val = updated_sql if updated_sql else res
            if option == "DB":
                hostname = env_settings["LOCAL_DB_CONFIG"]["HOST_NAME"]
                user_name = env_settings["LOCAL_DB_CONFIG"]["USER_NAME"]
                password = env_settings["LOCAL_DB_CONFIG"]["PASSWORD"]
                port = env_settings["LOCAL_DB_CONFIG"]["PORT"]
                db_name = env_settings["LOCAL_DB_CONFIG"]["DB_NAME"]

                db_obj = DBConfig(db_name, hostname, user_name, password, port, base_path=base_path, dialect=db_dialect)

                q_res, err = db_obj.execute_query_db(query=_val)

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

            results.append("Query Results: \n")
            if q_res:
                click.echo(f"The query results are:\n {q_res}")
                results.extend([str(q_res), "\n"])
            else:
                click.echo(f"While executing query:\n {err}")
                results.extend([str(err), "\n"])
            # results.extend(["Query Results:", q_res])
        save_sql = click.prompt("Would you like to save the generated SQL (y/n)?") if is_command else "n"
        if save_sql.lower() == "y" or save_sql.lower() == "yes":
            # Persist for future use
            _val = updated_sql if updated_sql else res
            save_query(base_path, query=question, response=_val)
        else:
            click.echo("Exiting...")

    return results, err


if __name__ == "__main__":
    cli()
