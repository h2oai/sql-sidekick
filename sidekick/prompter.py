import json
import os
from pathlib import Path

import click
import openai
import toml
from colorama import Back as B
from colorama import Fore as F
from colorama import Style
from .db_config import DBConfig
from loguru import logger
from .memory import EntityMemory
from .query import SQLGenerator
from .utils import save_query, setup_dir

# Load the config file and initialize required paths
base_path = (Path(__file__).parent / "../").resolve()
env_settings = toml.load(f"{base_path}/sidekick/configs/.env.toml")


def color(fore="", back="", text=None):
    return f"{fore}{back}{text}{Style.RESET_ALL}"


@click.group()
@click.version_option()
def cli():
    click.echo(
        """Welcome to the SQL Sidekick!\nI am AI assistant that helps you with SQL queries.
I can help you with the following:
1. Configure a local database(for schema validation and syntax checking): `python sidekick/prompter.py configure db-setup`.
2. Learn contextual query/answer pairs: `python sidekick/prompter.py learn add-samples`.
3. Simply add context: `python sidekick/prompter.py learn update-context`.
4. Ask a question: `python sidekick/prompter.py query`.\n
"""
    )
    # Book-keeping
    setup_dir(base_path)


@cli.group("configure")
def configure():
    """Helps in configuring local database."""


def enter_table_name():
    val = input(color(F.GREEN, "", "Would you like to create a table for the database? (y/n): "))
    return val


@configure.command("db-setup", help="Enter information to configure postgres database locally")
@click.option("--db_name", "-n", default="querydb", help="Database name", prompt="Enter Database name")
@click.option("--hostname", "-h", default="localhost", help="Database hostname", prompt="Enter hostname name")
@click.option("--user_name", "-u", default="postgres", help="Database username", prompt="Enter username name")
@click.option(
    "--password",
    "-p",
    default="abc",
    hide_input=True,
    help="Database password",
    prompt="Enter password",
)
@click.option("--port", "-P", default=5432, help="Database port", prompt="Enter port (default 5432)")
def db_setup(db_name: str, hostname: str, user_name: str, password: str, port: int):
    """Creates context for the new Database"""
    click.echo(f" Information supplied:\n {db_name}, {hostname}, {user_name}, {password}, {port}")
    try:
        env_settings["LOCAL_DB_CONFIG"]["HOST_NAME"] = hostname
        env_settings["LOCAL_DB_CONFIG"]["USER_NAME"] = user_name
        env_settings["LOCAL_DB_CONFIG"]["PASSWORD"] = password
        env_settings["LOCAL_DB_CONFIG"]["PORT"] = port
        env_settings["LOCAL_DB_CONFIG"]["DB_NAME"] = db_name
        # Update settings file for future use.
        f = open(f"{base_path}/.env.toml", "w")
        toml.dump(env_settings, f)
        f.close()

        # For current session
        db_obj = DBConfig(db_name, hostname, user_name, password, port)
        if not db_obj.db_exists():
            db_obj.create_db()
            click.echo("Database created successfully!")
        else:
            click.echo("Database already exists!")

        val = enter_table_name()
        while True:
            if val.lower() != "y" and val.lower() != "n":
                click.echo("In-correct values. Enter Yes(y) or no(n)")
                val = enter_table_name()
            else:
                break

        if val.lower() == "y" or val.lower() == "yes":
            table_value = input("Enter table name: ")
            click.echo(f"Table name: {table_value}")
            # set table name
            db_obj.table_name = table_value.replace(" ", "_")
            db_obj.create_table()

        # Check if table exists; pending --> and doesn't have any rows
        if db_obj.has_table():
            click.echo(f"Checked table {db_obj.table_name} exists in the DB.")
            val = input(color(F.GREEN, "", "Would you like to add few sample rows (at-least 3)? (y/n): "))
            if val.lower() == "y":
                db_obj.add_samples()
            else:
                click.echo("Exiting...")
                return
        else:
            click.echo("Job done. Ask a question now!")
    except Exception as e:
        click.echo(f"Error creating database. Check configuration parameters.\n: {e}")


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
def query(question: str):
    """Asks question and returns SQL."""
    # Book-keeping
    setup_dir(base_path)

    # Check if table exists
    path = f"{base_path}/var/lib/tmp/data"
    table_context_file = f"{path}/table_context.json"
    table_context = json.load(open(table_context_file, "r")) if Path(table_context_file).exists() else {}
    table_names = []
    if table_context:
        table_name = table_context.get("tables_in_use", None)
        table_names = [_t.replace(" ", "_") for _t in table_name]
    else:
        table_names = [click.prompt("Which table to use?")]
        table_context["tables_in_use"] = [_t.replace(" ", "_") for _t in table_names]
        with open(f"{path}/table_context.json", "w") as outfile:
            json.dump(table_context, outfile, indent=4, sort_keys=False)
    logger.info(f"Table in use: {table_names}")
    # Check if .env.toml file exists
    api_key = env_settings["OPENAI"]["OPENAI_API_KEY"]
    if api_key is None or api_key == "":
        if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
            val = input(
                color(F.GREEN, "", "Looks like API key is not set, would you like to set OPENAI_API_KEY? (y/n):")
            )
            if val.lower() == "y":
                api_key = input(color(F.GREEN, "", "Enter OPENAI_API_KEY:"))
        os.environ["OPENAI_API_KEY"] = api_key
        env_settings["OPENAI"]["OPENAI_API_KEY"] = api_key
        # Update settings file for future use.
        f = open(f"{base_path}/.env.toml", "w")
        toml.dump(env_settings, f)
        f.close()
    openai.api_key = api_key

    # Set context
    logger.info("Setting context...")
    logger.info(f"Question: {question}")
    # Get updated info from .env.toml
    host_name = env_settings["LOCAL_DB_CONFIG"]["HOST_NAME"]
    user_name = env_settings["LOCAL_DB_CONFIG"]["USER_NAME"]
    passwd = env_settings["LOCAL_DB_CONFIG"]["PASSWORD"]
    db_name = env_settings["LOCAL_DB_CONFIG"]["DB_NAME"]

    db_url = f"postgresql+psycopg2://{user_name}:{passwd}@{host_name}/{db_name}".format(
        user_name, passwd, host_name, db_name
    )

    sql_g = SQLGenerator(db_url, api_key, path=base_path)
    sql_g._tasks = sql_g.generate_tasks(table_names, question)
    click.echo(sql_g._tasks)

    updated_tasks = None
    if sql_g._tasks is not None:
        edit_val = click.prompt("Would you like to edit the tasks? (y/n)")
        if edit_val.lower() == "y":
            updated_tasks = click.edit(sql_g._tasks)
            click.echo(f"Tasks:\n {updated_tasks}")
        else:
            click.echo("Skipping edit...")
    if updated_tasks is not None:
        sql_g._tasks = updated_tasks
    res = sql_g.generate_sql(table_name, question)
    logger.info(f"Generated response:\n\n{res}")

    if res is not None:
        edit_val = click.prompt("Would you like to edit the SQL? (y/n)")
        if edit_val.lower() == "y" or edit_val.lower() == "yes":
            updated_sql = click.edit(res)
            click.echo(f"Updated SQL:\n {updated_sql}")

        save_sql = click.prompt("Would you like to save the generated SQL?")
        if save_sql.lower() == "y" or save_sql.lower() == "yes":
            # Persist for future use
            save_query(base_path, query=question, response=res)


if __name__ == "__main__":
    cli()
