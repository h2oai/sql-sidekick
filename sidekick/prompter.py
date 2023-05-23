import json
import os
from pathlib import Path

import click
import openai
import toml
from colorama import Back as B
from colorama import Fore as F
from colorama import Style
from db_config import DBConfig
from loguru import logger
from query import SQLGenerator
from memory import EntityMemory

# Load the config file and initialize required paths
base_path = (Path(__file__).parent / "../").resolve()
env_settings = toml.load(f"{base_path}/.env.toml")


def color(fore="", back="", text=None):
    return f"{fore}{back}{text}{Style.RESET_ALL}"


@click.group()
@click.version_option()
def cli():
    """ """


@cli.group("configure")
def configure():
    """Helps in configuring local database."""


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
        db_obj = DBConfig(db_name, hostname, user_name, password, port)
        db_obj.create_db()
        click.echo("Database created successfully!")
    except Exception as e:
        click.echo(f"Error creating database\n: {e}")
    val = input(color(F.GREEN, "", "Would you like to create a table for the database? (y/n): "))
    if val.lower() == "y":
        table_value = input("Enter table name: ")
        click.echo(f"Table name: {table_value}")
        # set table name
        db_obj.table_name = table_value
        db_obj.create_table()

    # Check if table exists; pending --> and doesn't have any rows
    if db_obj.has_table():
        click.echo(f"Local table {db_obj.table_name} exists.")
        val = input(color(F.GREEN, "", "Would you like to add few sample rows (at-least 3)? (y/n): "))
        if val.lower() == "y":
            db_obj.add_samples()
        else:
            click.echo("Exiting...")
            return


@cli.group("learn")
def learn():
    """Helps in learning and building memory."""


def _add_context(entity_memory: EntityMemory):
    _FORMAT = '''# Add input Query and Response \n\n
"Query": "<any query>"
"Response": """<respective response>"""
'''
    res = click.edit(_FORMAT.replace("\t", ""))
    # Check if user has entered any value
    if res:
        try:
            entity_memory.save_context(res)
        except ValueError as ve:
            logger.info(f"Not a valid input. Try again")


@learn.command("add-context", help="Enter information to add more context")
def add_query_response():
    em = EntityMemory(k=5, path=base_path)
    _add_context(em)
    _more = "y"
    while _more.lower() != "n" or _more.lower() != "no":
        _more = click.prompt("Would you like to enter more information? (y/n)")
        if _more.lower() == "y":
            _add_context(em)
        else:
            break


@learn.command()
@click.option(
    "--edit_context",
    "-ec",
    help="Update context in memory for future use",
    prompt="Would you like to add/update additional context? (y/n)?",
)
def update_context(edit_context: str):
    """Helps learn context for generation."""
    context_dict = """{"<context_key>": "<context_value>"
    }
    """
    if edit_context.lower() == "y":
        updated_context = click.edit(context_dict)
        click.echo(f"Context:\n {updated_context}")
        if updated_context:
            context_dict = json.loads(updated_context)
            path = f"{base_path}/var/lib/tmp/data/"
            with open(f"{path}/context.json", "w") as outfile:
                json.dump(context_dict, outfile, indent=4, sort_keys=False)
    else:
        click.echo("Value not supported. Try Y/N ...")


@cli.command()
@click.option("--question", "-q", help="Database name", prompt="Ask a question")
def query(question: str):
    """Asks question and returns SQL."""

    # Check if table exists
    path = f"{base_path}/var/lib/tmp/data/"
    table_context_file = f"{path}/table_context.json"
    table_context = json.load(open(table_context_file, "r")) if Path(table_context_file).exists() else {}
    if table_context:
        table_name = table_context.get("tables_in_use", None)
    else:
        table_name = [click.prompt("Which table to use?")]
        table_context["tables_in_use"] = table_name
        with open(f"{path}/table_context.json", "w") as outfile:
            json.dump(table_context, outfile, indent=4, sort_keys=False)
    logger.info(f"Table in use: {table_name}")
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
    # Below re-definition is temporary
    host_name = "localhost"
    user_name = "postgres"
    passwd = "abc"
    db_name = "postgres"
    db_name = "querydb"
    db_url = f"postgresql+psycopg2://{user_name}:{passwd}@{host_name}/{db_name}".format(
        user_name, passwd, host_name, db_name
    )

    sql_g = SQLGenerator(db_url, api_key, path=base_path)
    sql_g._tasks = sql_g.generate_tasks(table_name, question)
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
        if edit_val.lower() == "y":
            updated_sql = click.edit(res)
            click.echo(f"Updated SQL:\n {updated_sql}")
        else:
            click.echo("Exiting...")


if __name__ == "__main__":
    click.echo("Welcome to the SQL Sidekick!")
    cli()
