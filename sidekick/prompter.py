import click


@click.group()
@click.version_option()
def cli():
    """ """


@cli.group("configure")
def configure():
    """Helps in configuring local database."""


@configure.command("db-setup", help="Enter information to configure postgres database locally")
@click.option("--db_name", "-n", default="postgres", help="Database name", prompt="Enter Database name")
@click.option("--hostname", "-h", default="localhost", help="Database hostname", prompt="Enter hostname name")
@click.option("--user_name", "-u", default="postgres", help="Database username", prompt="Enter username name")
@click.option(
    "--password",
    "-p",
    default="postgres",
    hide_input=True,
    help="Database password",
    prompt="Enter password",
)
@click.option("--port", "-P", default=5432, help="Database port", prompt="Enter port (default 5432)")
def db_setup(db_name: str, hostname: str, user_name: str, password: str, port: int):
    """Creates context for the new Database"""
    click.echo(" Information supplied:\n {}, {}, {}, {}, {}".format(db_name, hostname, user_name, password, port))


@cli.group("learn")
def learn():
    """Learn context."""


if __name__ == "__main__":
    click.echo("Welcome to the SQL Sidekick!")
    cli()
