# --------------------------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------------------------

import abc
import datetime
from enum import Enum
from typing import Annotated, Any

import pydantic
import pydantic.functional_serializers
import pydantic.functional_validators
from typing_extensions import Self

# --------------------------------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------------------------------

__all__ = [
    "DataTypeName",
    "NoneType",
    "TYPE_MAP",
    "ScalarType",
    "TableConfig",
    "VariableConfig",
    "Table",
    "PipewatchArtifact",
]


# --------------------------------------------------------------------------------------------------
# Enums
# --------------------------------------------------------------------------------------------------


class DataTypeName(str, Enum):
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    TIMESTAMP = "timestamp"


# --------------------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------------------

_NONE_STRINGS = (
    "None",
    "",
)


# --------------------------------------------------------------------------------------------------
# Functions
# --------------------------------------------------------------------------------------------------


def _validate_none_type(
    value: Any,
) -> Any:
    if isinstance(value, str) and value in _NONE_STRINGS:
        return None

    return value


def _strip_timezone(value: Any) -> Any:
    if isinstance(value, str):
        return " ".join(value.split(" ")[:2])
    return value


# --------------------------------------------------------------------------------------------------
# Types
# --------------------------------------------------------------------------------------------------

NoneType = Annotated[
    None,
    pydantic.functional_validators.BeforeValidator(_validate_none_type),
]

DateTime = Annotated[datetime.datetime, pydantic.functional_validators.BeforeValidator(_strip_timezone)]

ScalarType = Annotated[
    NoneType | int | float | bool | DateTime | str,
    pydantic.Field(union_mode="left_to_right"),
]

Table = Annotated[
    dict[str, ScalarType | Annotated[list[ScalarType], pydantic.Field(min_length=1)]],
    pydantic.Field(min_length=1),
]

TYPE_MAP = {
    DataTypeName.STRING: str,
    DataTypeName.INTEGER: int,
    DataTypeName.FLOAT: float,
    DataTypeName.BOOLEAN: bool,
    DataTypeName.TIMESTAMP: DateTime,
}


# --------------------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------------------


class _BaseModel(
    pydantic.BaseModel,
    abc.ABC,
    extra="forbid",
):
    pass


class VariableConfig(_BaseModel):
    type: DataTypeName
    regex: str | None = None


class TableConfig(_BaseModel):
    name: str
    variables: dict[str, VariableConfig]

    @pydantic.model_validator(mode="after")
    def _validate_regexes(self) -> Self:
        regex_set = any(variable.regex is not None for variable in self.variables.values())
        if regex_set and not all(variable.regex is not None for variable in self.variables.values()):
            msg = "All variables must have a regex if at least one variable has a regex."
            raise ValueError(msg)
        return self


class PipewatchArtifact(_BaseModel):
    configuration: TableConfig
    table: Table

    # TODO(schmuck): add computed table length here.

    @pydantic.model_validator(mode="after")
    def _validate_equal_length(self) -> Self:
        first_column = next(iter(self.table.values()))
        if isinstance(first_column, list):
            length = len(first_column)
            for column in self.table.values():
                if not isinstance(column, list):
                    msg = "All columns must be lists."
                    raise ValueError(msg)
                if len(column) != length:
                    msg = "All columns must have the same length."
                    raise ValueError(msg)
        return self

    @pydantic.model_validator(mode="after")
    def _validate_variables(self) -> Self:
        for key in self.table:
            if key not in self.configuration.variables:
                msg = f"Column '{key}' is not defined in the table configuration."
                raise ValueError(msg)
        for key in self.configuration.variables:
            if key not in self.table:
                msg = f"Column '{key}' is not defined in the table."
                raise ValueError(msg)
        return self

    @pydantic.model_validator(mode="after")
    def _validate_types(self) -> Self:
        for key, value in self.table.items():
            expected_type = TYPE_MAP[self.configuration.variables[key].type] | None

            if isinstance(value, list):
                for i, item in enumerate(value):
                    value[i] = pydantic.TypeAdapter(expected_type).validate_python(item)
            else:
                self.table[key] = [pydantic.TypeAdapter(expected_type).validate_python(value)]

        return self
