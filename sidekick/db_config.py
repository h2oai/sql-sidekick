# create db with supplied info
import json
from pathlib import Path

import pandas as pd
import psycopg2 as pg
import sqlalchemy
from psycopg2.extras import Json
from sidekick.configs.data_template import data_samples_template
from sidekick.logger import logger
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
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
        self.column_names = []
        if dialect == "sqlite":
            logger.debug(f"Creating SQLite DB: sqlite:///{base_path}/db/sqlite/{db_name}.db")
            self._url = f"sqlite:///{base_path}/db/sqlite/{db_name}.db"
        else:
            self._url = f"{self.dialect}://{self.user_name}:{self.password}@{self.hostname}:{self.port}/"

    @property
    def table_name(self):
        return self._table_name

    @table_name.setter
    def table_name(self, val):
        self._table_name = val.lower().replace(" ", "_")

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
        try:
            with engine.connect() as conn:
                # conn.execute("commit")
                # Do not substitute user-supplied database names here.
                if self.dialect != "sqlite":
                    conn.execute("commit")
                    res = conn.execute(f"CREATE DATABASE {self.db_name}")
                    self._url = f"{self._url}{self.db_name}"
                    return res, None
                else:
                    logger.debug("SQLite DB is created when 'engine.connect()' is called")

            return True, None
        except SQLAlchemyError as sqla_error:
            logger.debug("SQLAlchemy error:", sqla_error)
            return None, sqla_error
        except Exception as error:
            logger.debug("Error Occurred:", error)
            return None, error

    def _extract_schema_info(self, schema_info_path=None):
        # From jsonl format
        # E.g. {"Column Name": "id", "Column Type": "uuid PRIMARY KEY"}
        if schema_info_path is None:
            table_info_file = f"{self.base_path}/var/lib/tmp/data/table_context.json"
            if Path(table_info_file).exists():
                with open(table_info_file, "w") as outfile:
                    schema_info_path = json.load(outfile)["schema_info_path"]
        res = []
        sample_values = []
        try:
            if Path(schema_info_path).exists():
                with open(schema_info_path, "r") as in_file:
                    for line in in_file:
                        if line.strip():
                            data = json.loads(line)
                            if "Column Name" in data and "Column Type" in data:
                                col_name = data["Column Name"]
                                self.column_names.append(col_name)
                                col_type = data["Column Type"]
                                if col_type.lower() == "text":
                                    col_type = col_type + " COLLATE NOCASE"
                                # if column has sample values, save in cache for future use.
                                if "Sample Values" in data:
                                    _sample_values = data["Sample Values"]
                                    _ds = data_samples_template.format(
                                        column_name=col_name,
                                        comma_separated_sample_values=",".join(
                                            str(_sample_val) for _sample_val in _sample_values
                                        ),
                                    )
                                    sample_values.append(_ds)
                                _new_samples = f"{col_name} {col_type}"
                            res.append(_new_samples)
                if len(sample_values) > 0:
                    # cache it for future use
                    with open(
                        f"{self.base_path}/var/lib/tmp/data/{self._table_name}_column_values.json", "w"
                    ) as outfile:
                        json.dump(sample_values, outfile, indent=2, sort_keys=False)
        except ValueError as ve:
            logger.error(f"Error in reading table context file: {ve}")
            pass
        return res

    def create_table(self, schema_info_path=None, schema_info=None):
        try:
            engine = create_engine(self._url, isolation_level="AUTOCOMMIT")
            self._engine = engine
            if self.schema_info is None:
                if schema_info is not None:
                    self.schema_info = schema_info
                else:
                    # If schema information is not provided, extract from the template.
                    self.schema_info = """,\n""".join(self._extract_schema_info(schema_info_path)).strip()
                    logger.debug(f"Schema info used for creating table:\n {self.schema_info}")
            create_syntax = f"""
                    CREATE TABLE IF NOT EXISTS {self.table_name} (
                        {self.schema_info}
                    )
                    """
            with engine.connect() as conn:
                if self.dialect != "sqlite":
                    conn.execute("commit")
                conn.execute(create_syntax)

            return self.table_name, None
        except SQLAlchemyError as sqla_error:
            logger.debug("SQLAlchemy error:", sqla_error)
            return None, sqla_error
        except Exception as error:
            logger.debug("Error Occurred:", error)
            return None, error

    def has_table(self):
        engine = create_engine(self._url)
        return sqlalchemy.inspect(engine).has_table(self.table_name)

    def data_preview(self, table_name):
        if table_name:
            query_str = f"SELECT * FROM {table_name} LIMIT 10"
            result = self.execute_query_db(query_str)
        else:
            result = "Table not found. Make sure uploaded dataset is registered."
        return result

    def add_samples(self, data_csv_path=None):
        conn_str = self._url
        try:
            logger.debug(f"Adding sample values to table: {data_csv_path}")
            df_chunks = pd.read_csv(data_csv_path, chunksize=5000)
            engine = create_engine(conn_str, isolation_level="AUTOCOMMIT")

            for idx, chunk in enumerate(df_chunks):
                # Write rows to database
                logger.debug(f"Inserting chunk: {idx}")
                chunk.columns = self.column_names
                # Make sure column names in the data-frame match the schema
                chunk.to_sql(self.table_name, engine, if_exists="append", index=False, method="multi")

            logger.info(f"Data inserted into table: {self.table_name}")
            # Fetch the number of rows from the table
            sample_query = f"SELECT COUNT(*) AS ROWS FROM {self.table_name} LIMIT 1"
            num_rows = pd.read_sql_query(sample_query, engine)
            res = num_rows.values[0][0]
            logger.info(f"Number of rows inserted: {res}")
            engine.dispose()
            return res, None
        except SQLAlchemyError as sqla_error:
            logger.debug("SQLAlchemy error:", sqla_error)
            return None, sqla_error
        except Exception as error:
            logger.debug("Error Occurred:", error)
            return None, error
        finally:
            if engine:
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
                logger.debug(f"Executing query:\n {query}")
                _query = text(query)
                result = connection.execute(_query)

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
