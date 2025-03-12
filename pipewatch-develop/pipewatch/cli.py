# The framework needs boolean values in functions
# ruff: noqa: FBT001
# ruff: noqa: FBT002
# ignore boolean arguments in typer commands
# ruff: noqa: FBT002
# ignore implicit optional arguments in typer commands
# ruff: noqa: RUF013

# TODO(schmuck): Fix this
# ignore f-strings in logging
# ruff: noqa: G004

# --------------------------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------------------------

import csv
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Annotated, Any

import package_lib.cli
import pipewatch_lib as lib
import pydantic
import typer
from package_lib import utils
from package_lib.cli import console, handle_exceptions

from pipewatch import context, queries

# --------------------------------------------------------------------------------------------------
# Module Variables
# --------------------------------------------------------------------------------------------------

__version__ = "0.3.2"

_logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------------------
# Command Declarations
# --------------------------------------------------------------------------------------------------

app = typer.Typer(
    name="Python Package Template CLI",
    help="A command-line interface for the PipeWatch monitoring system.",
    pretty_exceptions_show_locals=False,
    add_completion=False,
)

config_app = typer.Typer(
    help="Configuration commands.",
)
app.add_typer(config_app, name="config")

artifact_app = typer.Typer(
    help="Artifact commands.",
)
app.add_typer(artifact_app, name="artifact")

# Set up the main callback
package_lib.cli.create_main_callback(
    app,
    enable_logging=True,
    enable_version=True,
    version=__version__,
    package_name=__package__.split(".")[0],
)


# --------------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------------


def _load_data_from_file(
    data_file: Path,
) -> dict[str, list[Any]]:
    if data_file.suffix == ".csv":
        _logger.info(f"Loading data from CSV file '{data_file}'...")
        with data_file.open() as f:
            data_raw = list(csv.reader(f, delimiter=","))

        data = defaultdict(list)
        for row in data_raw[1:]:
            for key, value in zip(data_raw[0], row, strict=False):
                data[key].append(value)
    else:
        _logger.info(f"Loading data from JSON-like file '{data_file}'...")
        data = utils.load_data(data_file)

    return data


def _parse_data(
    data_file: Path,
    table_config: lib.TableConfig,
) -> dict[str, list[Any]]:
    content = data_file.read_text()

    _logger.info(f"Parsing data from file '{data_file}'...")

    column_length = None
    data = defaultdict(list)
    for name, variable in table_config.variables.items():
        matches = re.findall(variable.regex, content)
        _logger.debug(f"Variable '{name}' matches: {matches}")
        match len(matches):
            case 0:
                msg = f"Could not parse variable '{name}' in log file. Maybe the regex is wrong?"
                raise typer.BadParameter(msg)
            case 1:
                pass
            case n:
                if column_length is not None and column_length != n:
                    msg = f"Variable '{name}' has a different number of matches than the other variables."
                    raise typer.BadParameter(msg)
                column_length = n
        data[name] = matches

    # Broadcast data
    if column_length is not None:
        for name, values in data.items():
            if len(values) == 1:
                data[name] = values * column_length

    _logger.info("Data parsed successfully.")

    return data


def _get_artifact(
    config_file: Path,
    data_file: Path,
    parse: bool = False,
    allow_missing_data: bool = False,
) -> lib.PipewatchArtifact:
    data_file_exists = data_file.exists()
    if not allow_missing_data and not data_file_exists:
        msg = f"The data file '{data_file}' does not exist."
        raise typer.BadParameter(msg)

    _logger.info(f"Loading configuration from file '{config_file}'...")
    try:
        table_config = lib.TableConfig.model_validate(utils.load_data(config_file))
        _logger.debug(f"Database Configuration: {table_config}")
    except pydantic.ValidationError as e:
        msg = f"The database configuration file is invalid: {e}"
        raise typer.BadParameter(msg) from e
    _logger.info("Configuration loaded successfully.")

    if not data_file_exists:
        data = {name: [None] for name in table_config.variables}
    else:
        data = _parse_data(data_file=data_file, table_config=table_config) if parse else _load_data_from_file(data_file)

    # Validate data
    _logger.info(f"Validating data from file '{data_file}'...")
    try:
        table = pydantic.TypeAdapter(
            lib.Table,
        ).validate_python(data)
        _logger.debug(f"Table: {table}")
    except pydantic.ValidationError as e:
        msg = f"The data file is invalid: {e}"
        raise typer.BadParameter(msg) from e
    _logger.info("Data validated successfully.")

    # Validate artifact
    _logger.info("Validating artifact...")
    try:
        artifact = lib.PipewatchArtifact(
            configuration=table_config,
            table=table,
        )
    except pydantic.ValidationError as e:
        msg = f"The database configuration and the data are incompatible: {e}"
        raise typer.BadParameter(msg) from e
    _logger.debug(f"Artifact: {artifact}")

    _logger.info("Artifact validated successfully.")

    return artifact


# --------------------------------------------------------------------------------------------------
# Arguments
# --------------------------------------------------------------------------------------------------

ConfigFileArgument = Annotated[
    Path,
    typer.Argument(
        callback=package_lib.cli.create_path_callback(allow_absolute=True, check_exists=True),
        help="The database configuration file.",
    ),
]

DataFileArgument = Annotated[
    Path,
    typer.Argument(
        callback=package_lib.cli.create_path_callback(allow_absolute=True, check_exists=False),
        help="The data file.",
    ),
]
ExistingDataFileArgument = Annotated[
    Path,
    typer.Argument(
        callback=package_lib.cli.create_path_callback(allow_absolute=True, check_exists=True),
        help="The data file.",
    ),
]

OutputDirArgument = Annotated[
    Path,
    typer.Argument(
        # TODO(schmuck): enable when upgrading package-lib
        # callback=package_lib.cli.create_path_callback(allow_absolute=True, allow_none=True, check_exists=True),
        help="The output directory. Defaults to the current working directory.",
    ),
]

ParseOption = Annotated[
    bool,
    typer.Option(
        "--parse",
        help="Parse the data file instead of loading it directly.",
    ),
]

AllowMissingDataOption = Annotated[
    bool,
    typer.Option(
        "--allow-missing-data",
        help="Allow the data file to be missing.",
    ),
]


# --------------------------------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------------------------------

# Config
# --------------------------------------------------------------------------------------------------


@config_app.command("validate")
@handle_exceptions
def _config_validate() -> None:
    """Validate the configuration. Run this after making changes."""
    # Simply accessing these attributes will validate them
    _ = context.config
    console.print("Configuration and files are valid.")


# Artifact
# --------------------------------------------------------------------------------------------------


@artifact_app.command("validate")
@handle_exceptions
def _artifact_validate(
    config_file: ConfigFileArgument,
    data_file: ExistingDataFileArgument,
    parse: ParseOption = False,
) -> None:
    """Validate the artifact. Run this after making changes."""
    _get_artifact(
        config_file=config_file,
        data_file=data_file,
        parse=parse,
    )

    console.print("Configuration and files are valid.")


@artifact_app.command("create")
@handle_exceptions
def _artifact_create(
    config_file: ConfigFileArgument,
    data_file: DataFileArgument,
    output_dir: OutputDirArgument = None,
    parse: ParseOption = False,
    allow_missing_data: AllowMissingDataOption = False,
) -> None:
    """Create an artifact from a database configuration and a data file."""
    artifact = _get_artifact(
        config_file=config_file,
        data_file=data_file,
        parse=parse,
        allow_missing_data=allow_missing_data,
    )

    resolved_path = (output_dir or Path.cwd()) / context.config.artifact_path
    with resolved_path.open("w") as f:
        f.write(artifact.model_dump_json())

    console.print(f"Artifact created at '{resolved_path}'.")


@artifact_app.command("post")
@handle_exceptions
def _artifact_post(
    config_file: ConfigFileArgument,
    data_file: DataFileArgument,
    parse: ParseOption = False,
    allow_missing_data: AllowMissingDataOption = False,
) -> None:
    """Post data directly to the database from a database configuration and a data file."""
    artifact = _get_artifact(
        config_file=config_file,
        data_file=data_file,
        parse=parse,
        allow_missing_data=allow_missing_data,
    )

    queries.post_artifact(
        artifact=artifact,
    )

    console.print("Artifact posted successfully.")
