# create db with supplied info
import psycopg2 as pg
import sqlalchemy
from psycopg2.extras import Json
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists


class DBConfig:
    def __init__(self, db_name, hostname, user_name, password, port, dialect="postgresql") -> None:
        self.db_name = db_name
        self.hostname = hostname
        self.user_name = user_name
        self.password = password
        self.port = port
        self._table_name = None
        self.schema_info = None
        self._engine = None
        self.dialect = dialect
        self._url = f"{self.dialect}://{self.user_name}:{self.password}@{self.hostname}:{self.port}/"

    @property
    def table_name(self):
        return self._table_name

    @table_name.setter
    def table_name(self, val):
        self._table_name = val

    @property
    def engine(self):
        return self._engine

    def db_exists(self):
        engine = create_engine(f"{self._url}{self.db_name}", echo=True)
        return database_exists(f"{engine.url}")

    def create_db(self):
        engine = create_engine(self._url)
        self._engine = engine

        with engine.connect() as conn:
            conn.execute("commit")
            # Do not substitute user-supplied database names here.
            res = conn.execute(f"CREATE DATABASE {self.db_name}")
        return res

    def create_table(self):
        engine = create_engine(
            f"{self.dialect}://{self.user_name}:{self.password}@{self.hostname}:{self.port}/{self.db_name}"
        )
        self._engine = engine
        if self.schema_info is None:
            self.schema_info = """
                id uuid PRIMARY KEY,
                ts TIMESTAMP WITH TIME ZONE NOT NULL,
                kind TEXT NOT NULL, -- or int?,
                user_id TEXT,
                user_name TEXT,
                resource_type TEXT NOT NULL, -- or int?,
                resource_id  TEXT,
                stream TEXT NOT NULL,
                source TEXT NOT NULL,
                payload jsonb NOT NULL
            """
        create_syntax = f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    {self.schema_info}
                )
                """
        with engine.connect() as conn:
            conn.execute("commit")
            conn.execute(create_syntax)
        return

    def has_table(self):
        engine = create_engine(
            f"{self.dialect}://{self.user_name}:{self.password}@{self.hostname}:{self.port}/{self.db_name}"
        )
        return sqlalchemy.inspect(engine).has_table(self.table_name)

    def add_samples(self):
        # Non-functional for now.
        conn = pg.connect(
            database=self.db_name, user=self.user_name, password=self.password, host=self.hostname, port=self.port
        )
        # Creating a cursor object using the cursor() method
        conn.autocommit = True
        cursor = conn.cursor()

        cursor.execute()

        # Commit your changes in the database
        conn.commit()
