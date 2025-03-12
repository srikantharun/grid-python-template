# --------------------------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------------------------

import enum
import logging
import sys
from typing import *

import mysql.connector
import mysql.connector.errorcode
import pipewatch_lib as lib
from package_lib.exceptions import HandledException
from typing_extensions import Self

from pipewatch_server import (
    config,
    model
)

# --------------------------------------------------------------------------------------------------
# Exports
# --------------------------------------------------------------------------------------------------

__all__ = [
    'spaces',
]

# --------------------------------------------------------------------------------------------------
# Module Variables
# --------------------------------------------------------------------------------------------------


_self = sys.modules[__name__]

spaces: dict[model.DatabaseSpaceName, 'MysqlDB']

_logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------------------
# Exceptions
# --------------------------------------------------------------------------------------------------

class DBConfigurationError(HandledException):
    pass


class DBAccessError(HandledException):
    pass


class DBTableDoesNotExistError(HandledException):
    pass


class DBInsertError(HandledException):
    pass


# --------------------------------------------------------------------------------------------------
# Classes
# --------------------------------------------------------------------------------------------------

class MysqlDB:
    _SQL_TYPE_MAP = {
        lib.DataTypeName.BOOLEAN:   "BOOLEAN",
        lib.DataTypeName.TIMESTAMP: "TIMESTAMP",
        lib.DataTypeName.INTEGER:   "INT",
        lib.DataTypeName.FLOAT:     "FLOAT",
        lib.DataTypeName.STRING:    "TEXT",
    }

    def __init__(
          self,
          host: str,
          database: str,
          user: str,
          password: str
    ) -> None:
        self._config = {
            'host':     host,
            'database': database,
            'user':     user,
            'password': password
        }

        self._connection = None
        self._cursor = None

    def __enter__(self) -> Self:
        self._connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._close()

    def _connect(self) -> None:
        _logger.info(f"Connecting to MySQL database '{self._config['database']}'...")
        try:
            self._connection = mysql.connector.connect(**self._config)
            self._cursor = self._connection.cursor()
        except mysql.connector.Error as err:
            if err.errno == mysql.connector.errorcode.ER_ACCESS_DENIED_ERROR:
                raise DBAccessError("Something is wrong with your user name or password") from err
            elif err.errno == mysql.connector.errorcode.ER_BAD_DB_ERROR:
                raise DBAccessError("Database does not exist") from err
            else:
                raise err

    def _close(self) -> None:
        _logger.info(f"Closing connection to MySQL database '{self._config['database']}'...")
        self._cursor.close()
        self._connection.close()

    def _get_columns(
          self,
          name: str
    ) -> list[str]:
        self._cursor.execute(f"SHOW COLUMNS FROM `{name}`")
        return [column[0] for column in self._cursor.fetchall()]

    def check_table_exists(self, table_name: str) -> bool:
        try:
            self._cursor.execute(f"SHOW TABLES LIKE '{table_name}'")
            return bool(self._cursor.fetchone())
        except mysql.connector.Error:
            return False

    def get_new_columns(
          self,
          name: str,
          columns: Iterable[str],
    ) -> set[str]:
        # Check if all columns in the data are in the table
        return set(columns) - set(self._get_columns(name))

    def get_dropped_columns(
          self,
          name: str,
          columns: Iterable[str],
    ) -> set[str]:
        # Check if all columns in the table are in the data
        return set(self._get_columns(name)) - set(columns)

    def get_column_diff(
          self,
          name: str,
          columns: Iterable[str],
    ) -> tuple[set[str], set[str]]:
        upstream_columns = set(self._get_columns(name))
        new_columns = set(columns) - set(upstream_columns)
        dropped_columns = set(upstream_columns) - set(columns)
        return new_columns, dropped_columns

    def create_table(
          self,
          name: str,
          data_types: dict[str, lib.DataTypeName]
    ) -> None:
        # Construct CREATE TABLE query
        columns_definitions = ', '.join([f"{col} {self._SQL_TYPE_MAP[data_types[col]]}" for col in data_types])

        query = f"CREATE TABLE `{name}` ({columns_definitions})"
        _logger.debug(
            f"Creating table '{name}' with SQL query:\n"
            f"{query}"
        )

        try:
            self._cursor.execute(query)
        except mysql.connector.Error as e:
            raise DBAccessError(f"Error in creating table {name}: {e}") from e

    def adapt_table(
          self,
          name: str,
          data_types: dict[str, lib.DataTypeName],
          allow_new_columns: bool = True,
          allow_dropped_columns: bool = False,
    ) -> None:
        assert allow_new_columns or allow_dropped_columns, \
            "At least one of 'allow_new_columns' and 'allow_dropped_columns' must be True"

        # Get the new and dropped columns
        new_columns, dropped_columns = self.get_column_diff(name, data_types)

        # Add new columns
        if allow_new_columns and new_columns:
            for column in new_columns:
                query = f"ALTER TABLE `{name}` ADD COLUMN {column} {self._SQL_TYPE_MAP[data_types[column]]}"
                _logger.debug(
                    f"Adding column '{column}' to table '{name}' with SQL query:\n"
                    f"{query}"
                )

                try:
                    self._cursor.execute(query)
                except mysql.connector.Error as e:
                    raise DBAccessError(f"Error in adding column '{column}' to table '{name}': {e}") from e

        # Drop dropped columns
        if allow_dropped_columns and dropped_columns:
            for column in dropped_columns:
                query = f"ALTER TABLE `{name}` DROP COLUMN {column}"
                _logger.debug(
                    f"Dropping column '{column}' from table '{name}' with SQL query:\n"
                    f"{query}"
                )

                try:
                    self._cursor.execute(query)
                except mysql.connector.Error as e:
                    raise DBAccessError(f"Error in dropping column '{column}' from table '{name}': {e}") from e

    def insert(
          self,
          name: str,
          data: dict[str, Any],
    ) -> None:
        if not self.check_table_exists(name):
            raise DBTableDoesNotExistError(f"Table '{name}' does not exist")

        # # Adjust the table schema if necessary
        # self.adjust_table_columns(name, data, data_types)

        # Prepare columns and values for the INSERT statement
        columns = ', '.join(data.keys())
        values = tuple(data.values())
        placeholders = ', '.join(['%s'] * len(data))

        # Construct the INSERT statement
        query = f"INSERT INTO `{name}` ({columns}) VALUES ({placeholders})"
        _logger.debug(
            f"Inserting into table '{name}' with SQL query:\n"
            f"{query}\n"
            f"Values:\n"
            f"{values}"
        )

        try:
            self._cursor.execute(query, values)
            self._connection.commit()
        except mysql.connector.Error as err:
            self._connection.rollback()
            raise DBInsertError(f"Error in inserting into table '{name}': {err}") from err

    def insert_many(
          self,
          name: str,
          data: dict[str, list[Any]],
    ) -> None:
        # Ensure all columns in rows are of equal length
        list_lengths = [len(lst) for lst in data.values()]
        if len(set(list_lengths)) != 1:
            raise DBInsertError("All columns in rows must be of equal length")

        # Prepare columns and values for the INSERT statement
        columns = ', '.join(data.keys())
        placeholders = ', '.join(['%s'] * len(data))

        # Construct the INSERT statement for multiple rows
        query = f"INSERT INTO `{name}` ({columns}) VALUES ({placeholders})"

        # Prepare the data for batch insert
        num_rows = list_lengths[0]
        values_to_insert = [tuple(data[column][i] for column in data) for i in range(num_rows)]

        _logger.debug(
            f"Inserting into table '{name}' with SQL query:\n"
            f"{query}\n"
            f"Values:\n"
            f"{values_to_insert}"
        )

        try:
            self._cursor.executemany(query, values_to_insert)
            self._connection.commit()
        except mysql.connector.Error as err:
            self._connection.rollback()
            raise DBInsertError(f"Error when inserting into table '{name}': {err}") from err


# --------------------------------------------------------------------------------------------------
# Lazy Loading
# --------------------------------------------------------------------------------------------------

def __getattr__(
      name: str
) -> Any:
    if name == 'spaces':
        global spaces

        # Strip the protocol from the database URL
        host_url = config.config.database.url.unicode_string().split('://')[1]

        spaces = {
            space: MysqlDB(
                database=database_name,
                host=host_url,
                user=config.config.database.user,
                password=config.config.database.password.get_secret_value(),
            ) for space, database_name in config.config.database.spaces.items()
        }

        return spaces

    raise AttributeError(name)
