import click
from db_utils import DBUtils
from colorama import init
from colorama import Fore as F
from colorama import Back as B
from colorama import Style


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
        db_obj = DBUtils(db_name, hostname, user_name, password, port)
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
            pass
        else:
            click.echo("Exiting...")
            return


@cli.group("learn")
def learn():
    """Learn context."""


if __name__ == "__main__":
    click.echo("Welcome to the SQL Sidekick!")
    cli()
