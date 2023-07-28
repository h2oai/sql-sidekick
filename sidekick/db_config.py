# create db with supplied info
import json
from pathlib import Path

import pandas as pd
import psycopg2 as pg
import sqlalchemy
from pandasql import sqldf
from psycopg2.extras import Json
from sidekick.logger import logger
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists


class DBConfig:
    def __init__(
        self,
        db_name,
        hostname,
        user_name,
        password,
        port,
        base_path,
        schema_info_path=None,
        schema_info=None,
        dialect="sqlite",
    ) -> None:
        self.db_name = db_name
        self.hostname = hostname
        self.user_name = user_name
        self.password = password
        self.port = port
        self._table_name = None
        self.schema_info_path = schema_info_path
        self.schema_info = schema_info
        self._engine = None
        self.dialect = dialect
        self.base_path = base_path
        if dialect == "sqlite":
            self._url = f"sqlite:///{base_path}/db/sqlite/{db_name}.db"
        else:
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
        if self.dialect == "sqlite":
            engine = create_engine(f"{self._url}", echo=True)
        else:
            engine = create_engine(f"{self._url}{self.db_name}", echo=True)
        return database_exists(f"{engine.url}")

    def create_db(self):
        engine = create_engine(self._url)
        self._engine = engine

        with engine.connect() as conn:
            # conn.execute("commit")
            # Do not substitute user-supplied database names here.
            if self.dialect != "sqlite":
                conn.execute("commit")
                res = conn.execute(f"CREATE DATABASE {self.db_name}")
                self._url = f"{self._url}{self.db_name}"
                return res
            else:
                logger.debug("SQLite DB is created when 'engine.connect()' is called")

    def _extract_schema_info(self, schema_info_path=None):
        # From jsonl format
        # E.g. {"Column Name": "id", "Column Type": "uuid PRIMARY KEY"}
        if schema_info_path is None:
            table_info_file = f"{self.base_path}/var/lib/tmp/data/table_context.json"
            if Path(table_info_file).exists():
                with open(table_info_file, "w") as outfile:
                    schema_info_path = json.load(outfile)["schema_info_path"]
        res = []
        try:
            if Path(schema_info_path).exists():
                with open(schema_info_path, "r") as in_file:
                    for line in in_file:
                        if line.strip():
                            data = json.loads(line)
                            if "Column Name" in data and "Column Type" in data:
                                col_name = data["Column Name"]
                                col_type = data["Column Type"]
                                _new_samples = f"{col_name} {col_type}"
                            res.append(_new_samples)
        except ValueError as ve:
            logger.error(f"Error in reading table context file: {ve}")
            pass
        return res

    def create_table(self, schema_info_path=None, schema_info=None):
        engine = create_engine(
            self._url, isolation_level="AUTOCOMMIT"
        )
        self._engine = engine
        if self.schema_info is None:
            if schema_info is not None:
                self.schema_info = schema_info
            else:
                # If schema information is not provided, extract from the template.
                self.schema_info = """,\n""".join(self._extract_schema_info(schema_info_path)).strip()
                logger.debug(f"Schema info used for creating table:\n {self.schema_info}")
            # self.schema_info = """
            #     id uuid PRIMARY KEY,
            #     ts TIMESTAMP WITH TIME ZONE NOT NULL,
            #     kind TEXT NOT NULL, -- or int?,
            #     user_id TEXT,
            #     user_name TEXT,
            #     resource_type TEXT NOT NULL, -- or int?,
            #     resource_id  TEXT,
            #     stream TEXT NOT NULL,
            #     source TEXT NOT NULL,
            #     payload jsonb NOT NULL
            # """
        create_syntax = f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    {self.schema_info}
                )
                """
        with engine.connect() as conn:
            if self.dialect != "sqlite":
                conn.execute("commit")
            conn.execute(create_syntax)
        return

    def has_table(self):
        engine = create_engine(
            self._url
        )

        return sqlalchemy.inspect(engine).has_table(self.table_name)

    def add_samples(self, data_csv_path=None):
        conn_str = self._url
        try:
            df = pd.read_csv(data_csv_path)
            engine = create_engine(conn_str, isolation_level="AUTOCOMMIT")

            sample_query = f"SELECT COUNT(*) AS ROWS FROM {self.table_name} LIMIT 1"
            num_rows_bef = pd.read_sql_query(sample_query, engine)

            # Write rows to database
            df.to_sql(self.table_name, engine, if_exists="append", index=False)

            # Fetch the number of rows from the table
            num_rows_aft = pd.read_sql_query(sample_query, engine)
            logger.info(f"Number of rows inserted: {num_rows_aft.iloc[0, 0] - num_rows_bef.iloc[0, 0]}")
            engine.dispose()

        except Exception as e:
            logger.info(f"Error occurred : {format(e)}")
        finally:
            engine.dispose()

    def execute_query_db(self, query=None, n_rows=100):
        output = []
        if self.dialect != "sqlite":
            conn_str = f"{self._url}{self.db_name}"
        else:
            conn_str = self._url

        try:
            if query:
                # Create an engine
                engine = create_engine(conn_str)

                # Create a connection
                connection = engine.connect()
                result = connection.execute(query)

                # Process the query results
                cnt = 0
                for row in result:
                    if cnt <= n_rows:
                        # Access row data using row[column_name]
                        output.append(row)
                        cnt += 1
                    else:
                        break
                # Close the connection
                connection.close()

                # Close the engine
                engine.dispose()
            else:
                logger.info("Query Empty or None!")
            return output, None
        except Exception as e:
            err = f"Error occurred : {format(e)}"
            logger.info(err)
            return None, err
        finally:
            connection.close()
            engine.dispose()
