# --------------------------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------------------------

import datetime
import logging
from typing import *

import jinja2
import jinja2.sandbox
import pipewatch_lib as lib
import pydantic
import pydantic.functional_validators
from package_lib.exceptions import HandledException

from pipewatch_server import (
    config,
    db,
    gitlab,
    model
)

# --------------------------------------------------------------------------------------------------
# Exports
# --------------------------------------------------------------------------------------------------

__all__ = [
    'process_webhook_event',
]


# --------------------------------------------------------------------------------------------------
# Exceptions
# --------------------------------------------------------------------------------------------------

class ArtifactException(HandledException):
    pass


class TemplateException(HandledException):
    pass


# --------------------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------------------

_FINISHED_STATUS = (
    model.BuildStatus.SUCCESS,
    model.BuildStatus.FAILED,
    model.BuildStatus.CANCELED,
)

# --------------------------------------------------------------------------------------------------
# Module Variables
# --------------------------------------------------------------------------------------------------

_logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------------------
# Private Functions
# --------------------------------------------------------------------------------------------------

def _filter_event(
      event: model.PipelineEvent | model.JobEvent,
      webhook_name: model.WebhookName,
      table: config.PipelineTable | config.JobTable,
      env: jinja2.sandbox.ImmutableSandboxedEnvironment,
) -> bool:
    # Skip table if it is not for the current event type
    if webhook_name != table.webhook:
        _logger.info(f"Table '{table.name}' is not for this event type, skipping...")
        return False

    # Skip unfinished events if requested
    if table.skip_unfinished:
        status = (
            event.object_attributes.status if webhook_name == model.WebhookName.PIPELINE else
            event.build_status
        )
        if status not in _FINISHED_STATUS:
            _logger.info(f"Event not finished yet, skipping...")
            return False

    # Filter table according to configured filters
    for filter in table.filters:
        _logger.debug(f'Applying filter "{{{{ {filter} }}}}"...')
        try:
            rendered_filter = env.from_string(f"{{{{ {filter} }}}}").render()
        except jinja2.TemplateError as e:
            raise TemplateException(
                f"Error while rendering filter '{filter}' in table '{table.name}':\n"
                f"Filter: {filter}\n"
                f"Error: {e}"
            ) from e

        _logger.debug(f"Rendered filter: '{rendered_filter}'")
        if not rendered_filter.lower().capitalize() == 'True':
            _logger.debug(
                f"Event did not pass filter, skipping"
            )
            return False
        _logger.debug(f"Event passed filter")

    return True


def _attach_artifact(
      types: dict[str, lib.DataTypeName],
      data: dict[str, list[Any]],
      artifact: lib.PipewatchArtifact,
) -> tuple[dict[str, lib.DataTypeName], dict[str, list[Any]]]:
    _logger.info(f"Attaching artifact to data...")
    _logger.debug(f"Artifact:\n{artifact}")

    if set(artifact.configuration.variables) & set(types):
        raise ArtifactException(
            f"Artifact variables and table variables must not overlap!\n"
            f"Artifact variables: {artifact.configuration.variables}\n"
            f"Table variables: {types}"
        )

    table_length = len(list(artifact.table.values())[0])
    artifact_types = {name: variable.type for name, variable in artifact.configuration.variables.items()}

    new_types = {**types, **artifact_types}
    new_data = {
        **{
            # Broadcast the data to the length of the table
            name: variables * table_length
            for name, variables in data.items()
        },
        **artifact.table,
    }

    _logger.info("Merged artifact and table data")
    _logger.debug(
        f"New types:\n{new_types}\n"
        f"New data:\n{new_data}"
    )

    return new_types, new_data


def _store_data(
      table_name: str,
      types: dict[str, lib.DataTypeName],
      data: dict[str, list[Any]],
      space: model.DatabaseSpaceName = model.DatabaseSpaceName.PRODUCTION,
) -> None:
    with db.spaces[space] as conn:
        if not conn.check_table_exists(table_name):
            _logger.warning(f"Table '{table_name}' did not exist, creating...")
            conn.create_table(table_name, types)

        conn.adapt_table(
            table_name,
            data_types=types,
            allow_new_columns=True,  # TODO(schmuck): Make this configurable
            allow_dropped_columns=False,  # TODO(schmuck): Make this configurable
        )

        _logger.info(f"Inserting data into database...")
        conn.insert_many(table_name, data)


def _process_table(
      event: model.PipelineEvent | model.JobEvent,
      webhook_name: model.WebhookName,
      table: config.PipelineTable | config.JobTable,
      env: jinja2.sandbox.ImmutableSandboxedEnvironment,
) -> None:
    if not _filter_event(
          event=event,
          webhook_name=webhook_name,
          table=table,
          env=env,
    ):
        _logger.info(f"Event did not pass all filters, skipping")
        return
    _logger.info(f"Event passed all filters, continuing")

    # In case of a job event and requested, attach the artifact to the data. There might not be an artifact however.
    artifact = None
    if table.webhook == model.WebhookName.JOB and table.attach_artifact:
        _logger.info("Retrieving job artifact...")
        artifact = gitlab.get_pipewatch_artifact(
            event.project.id,
            event.build_id,
        )

        # If there is no artifact, this job should be skipped
        if artifact is None:
            _logger.warning(f"Artifact of job '{event.build_id}' not found, maybe there is no artifact?")
            _logger.info(f"Skipping job '{event.build_id}' because an artifact was requested but not found")
            return

        _logger.debug(f"Artifact:\n{artifact}")

    _logger.info(f"Event passed all checks, data will be stored...")

    # Create the data for the table
    _logger.info(f"Processing variables...")
    table_name = table.name
    types = {variable.name: variable.type for variable in table.variables}
    data = {}
    for variable in table.variables:
        _logger.debug(f"Processing variable '{variable.name}'...")
        try:
            rendered_string = env.from_string(f"{{{{ {variable.value} }}}}").render()
            rendered_value = pydantic.TypeAdapter(
                lib.NoneType | lib.TYPE_MAP[variable.type]
            ).validate_python(rendered_string)
        except (
              jinja2.TemplateError,
              pydantic.ValidationError,
        ) as e:
            _logger.error(
                f"Error while rendering template '{variable.name}' in table '{table.name}':\n"
                f"Variable config: {variable}\n"
                f"Error: {e}"
            )
            rendered_value = None

        _logger.debug(f"Rendered value: '{rendered_value}'")
        data[variable.name] = [rendered_value]
    _logger.info(f"Variables processed")

    # Attach artifact if requested
    if artifact is not None:
        table_name += f":{artifact.configuration.name}"
        types, data = _attach_artifact(
            types=types,
            data=data,
            artifact=artifact,
        )

    # Insert data into database
    _store_data(table_name, types, data)


# --------------------------------------------------------------------------------------------------
# Public Functions
# --------------------------------------------------------------------------------------------------

def process_webhook_event(
      event: model.PipelineEvent | model.JobEvent,
      webhook_name: model.WebhookName,
) -> None:
    _logger.info(f"Processing {webhook_name.value} webhook event...")
    _logger.debug(f"Event:\n{event}")

    # Create Jinja2 environment with global variables available to the templates
    env = jinja2.sandbox.ImmutableSandboxedEnvironment()
    env.globals = {
        'event': event.model_dump(mode='json'),
        'api':   (
            gitlab.PipelineAPI(
                project_id=event.project.id,
                pipeline_id=event.object_attributes.id,
            ) if webhook_name == model.WebhookName.PIPELINE else
            gitlab.JobAPI(
                project_id=event.project.id,
                job_id=event.build_id,
            )
        )
    }

    for table in config.config.tables:
        _logger.info(f"Processing table '{table.name}'...")
        _process_table(
            event=event,
            webhook_name=webhook_name,
            table=table,
            env=env,
        )


def process_test_artifact(
      artifact: lib.PipewatchArtifact,
) -> None:
    _logger.info(f"Processing test artifact...")
    _logger.debug(f"Artifact:\n{artifact}")

    _logger.info(f"Attaching timestamp info to test artifact...")
    artifact.configuration.variables['created_at'] = lib.VariableConfig(type=lib.DataTypeName.TIMESTAMP)

    table_length = len(list(artifact.table.values())[0])
    artifact.table['created_at'] = [datetime.datetime.now()] * table_length

    artifact_types = {name: variable.type for name, variable in artifact.configuration.variables.items()}

    _store_data(
        artifact.configuration.name,
        types=artifact_types,
        data=artifact.table,
        space=model.DatabaseSpaceName.TESTING,
    )
