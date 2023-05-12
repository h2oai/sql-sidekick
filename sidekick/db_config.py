# create db with supplied info
import json
import psycopg2 as pg
import sqlalchemy
from sqlalchemy import create_engine
from psycopg2.extras import Json


class DBConfig:
    def __init__(self, db_name, hostname, user_name, password, port) -> None:
        self.db_name = db_name
        self.hostname = hostname
        self.user_name = user_name
        self.password = password
        self.port = port
        self._table_name = None
        self.schema_info = None
        self._engine = None

    @property
    def table_name(self):
        return self._table_name

    @table_name.setter
    def table_name(self, val):
        self._table_name = val

    @property
    def engine(self):
        return self._engine

    def create_db(self):
        engine = create_engine(f"postgresql://{self.user_name}:{self.password}@{self.hostname}:{self.port}/")
        self._engine = engine

        with engine.connect() as conn:
            conn.execute("commit")
            # Do not substitute user-supplied database names here.
            conn.execute(f"CREATE DATABASE {self.db_name}")
        return

    def create_table(self):
        engine = create_engine(
            f"postgresql://{self.user_name}:{self.password}@{self.hostname}:{self.port}/{self.db_name}"
        )

        schema_info = """
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
                    {schema_info}
                )
                """
        with engine.connect() as conn:
            conn.execute("commit")
            conn.execute(create_syntax)
        return

    def has_table(self):
        return sqlalchemy.inspect(self.engine).has_table(self.table_name)

    def add_samples(self):
        conn = pg.connect(
            database=self.db_name, user=self.user_name, password=self.password, host=self.hostname, port=self.port
        )
        # Creating a cursor object using the cursor() method
        conn.autocommit = True
        cursor = conn.cursor()

        cursor.execute(
            f"""INSERT into {self.table_name} (id, ts, kind, user_id, user_name, resource_type, resource_id, stream, source, payload)
        values ('12325678-1234-5678-1234-567812345678',
                '2023-04-11 17:39:56+0000', 'EVENT',
                '8cd4-476d-97e1-594b438eab7a', 'test_user',
                'resource_type_placeholder', '7de9-41b2-bd0f-854170640d6c',
                'running', 'localhost:9999',
                {Json({'engineEvent':
                    {'pausing': {'engine':
                    {'cpu': 4, 'gpu': 1, 'uid': 'dfba345a-449d-4a68-8b86-942a5a576716', 'name': 'workspaces/default/daiEngines/demo-1', 'type': 'TYPE_DRIVERLESS_AI', 'creator': 'NA', 'session': '9941a93c-375b-44b4-ad4e-17836ba72e5f', 'version': '1.10.4.1', 'createTime': '2023-02-16T10:34:13Z', 'resumeTime': '2023-02-28T15:42:02Z', 'displayName': 'Demo', 'memoryBytes': '64424509440', 'storageBytes': '282394099712', 'creatorDisplayName': 'a@x.ai'}
                    }}})}
                    )
        """
        )
        cursor.execute(
            f"""INSERT into {self.table_name} (id, ts, kind, user_id, user_name, resource_type, resource_id, stream, source, payload)
        values ('12335679-1234-5678-1234-567812345679',
                '2023-04-11 17:49:56+0000', 'EVENT',
                '8ce4-476d-97e1-594b438eab7a', 'test_user',
                'resource_type_placeholder', '7de9-41b2-bd0f-854170640d6c',
                'paused', 'localhost:9999',
                {Json({'engineEvent':
                    {'pausing': {'engine':
                    {'gpu': 2, 'uid': 'rfba345a-449d-4a68-8b86-942a5a576716', 'name': 'workspaces/default/daiEngines/demo-2', 'type': 'TYPE_DRIVERLESS_AI', 'creator': 'NA', 'session': '9941a93c-375b-44b4-ad4e-17836ba72e5f', 'version': '1.10.4.1', 'createTime': '2023-02-16T10:34:13Z', 'resumeTime': '2023-02-28T15:42:02Z', 'displayName': 'Demo', 'memoryBytes': '64424509440', 'storageBytes': '282394099712', 'creatorDisplayName': 'b@x.ai'}
                    }}})})
        """
        )

        cursor.execute(
            f"""insert into {self.table_name} (id, ts, kind, user_id, user_name, resource_type, resource_id, stream, source, payload)
        values ('12345379-1234-5678-1234-567812345778',
                '2023-04-11 18:49:56+0000', 'EVENT',
                '7ce4-476d-97e1-594b438eab9a', 'test_user',
                'resource_type_placeholder', '8de3-41b2-bd0f-854170640d6c',
                'running', 'localhost:9999',
                {Json({'engineEvent':
                    {'pausing': {'engine':
                    {'cpu': 10, 'uid': 'rfca345a-449d-4a68-8b86-942a5a576716', 'name': 'workspaces/default/daiEngines/demo-3', 'type': 'TYPE_DRIVERLESS_AI', 'creator': 'NA', 'session': '9941a93c-375b-44b4-ad4e-17836ba72e5f', 'version': '1.10.4.1', 'createTime': '2023-02-16T10:34:13Z', 'resumeTime': '2023-02-28T15:42:02Z', 'displayName': 'Demo', 'memoryBytes': '64424509440', 'storageBytes': '282394099712', 'creatorDisplayName': 'c@x.ai'}
                    }}})})
        """
        )

        # Commit your changes in the database
        conn.commit()
