# --------------------------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------------------------

import abc
import json
import logging
import sys
from functools import cached_property
from typing import Any

import gitlab
import gitlab.exceptions
import gitlab.v4.objects
import pipewatch_lib as lib
import requests.exceptions
from package_lib.exceptions import HandledException

from pipewatch_server import (
    config,
    model
)

# --------------------------------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------------------------------

__all__ = [
    'JobAPI',
    'PipelineAPI',
    'get_pipewatch_artifact',
]


# --------------------------------------------------------------------------------------------------
# Exceptions
# --------------------------------------------------------------------------------------------------

class GitlabError(HandledException):
    pass


class ArtifactError(HandledException):
    pass


# --------------------------------------------------------------------------------------------------
# Module Variables
# --------------------------------------------------------------------------------------------------


_self = sys.modules[__name__]
_self._gitlab_client: gitlab.Gitlab

_gitlab_client: gitlab.Gitlab

_logger = logging.getLogger(__name__)

_MAX_PAGE_LOOKBACK = 5


# --------------------------------------------------------------------------------------------------
# Classes
# --------------------------------------------------------------------------------------------------

class _GitlabAPI(abc.ABC):

    def __init__(
          self,
          project_id: int,
    ):
        self._project_id = project_id

    def _lookup_schedule(
          self,
          pipeline_id: int,
    ) -> dict[str, Any] | None:
        _logger.debug(
            f"Trying to get triggering schedule for pipeline '{pipeline_id}' "
            f"by reverse lookup from Gitlab's API..."
        )
        try:
            schedules = self._project.pipelineschedules.list(get_all=True)
            for schedule in schedules:
                last_page = int(schedule.pipelines.head().get('X-Total-Pages'))

                for page in range(last_page, last_page - _MAX_PAGE_LOOKBACK, -1):
                    pipelines = schedule.pipelines.list(get_all=False, page=page)

                    for pipeline in pipelines:
                        if pipeline.id == pipeline_id:
                            _logger.debug(
                                f"Found triggering schedule '{schedule.id}' for pipeline '{pipeline_id}'!"
                            )
                            return schedule.attributes
            _logger.debug(f"No triggering schedule found for pipeline '{pipeline_id}'!")
        except gitlab.exceptions.GitlabGetError:
            _logger.warning(
                f"Failed to get schedules for pipeline '{pipeline_id}'! "
                f"It is likely that the pipeline was not triggered by a schedule."
            )
            return None

    @cached_property
    def _project(self) -> gitlab.v4.objects.Project:
        """Get the project object from the Gitlab API."""
        _logger.debug(f"Getting project '{self._project_id}' from Gitlab's API...")
        return _self._gitlab_client.projects.get(self._project_id)

    @cached_property
    def project(self) -> dict[str, Any] | None:
        """Get the project attributes from the Gitlab API."""
        return self._project.attributes


class JobAPI(_GitlabAPI):

    def __init__(
          self,
          project_id: int,
          job_id: int,
    ):
        super().__init__(project_id)
        self._job_id = job_id

    @cached_property
    def job(self) -> dict[str, Any] | None:
        _logger.debug(f"Getting job '{self._job_id}' from Gitlab's API...")
        return self._project.jobs.get(self._job_id).attributes

    @cached_property
    def runner(self) -> dict[str, Any] | None:
        _logger.debug(f"Getting runner for job '{self._job_id}' from Gitlab's API...")
        return self.job['runner']

    @cached_property
    def pipeline(self) -> dict[str, Any] | None:
        _logger.debug(f"Getting pipeline for job '{self._job_id}' from Gitlab's API...")
        return self._project.pipelines.get(self.job['pipeline']['id']).attributes

    @cached_property
    def schedule(self) -> dict[str, Any] | None:
        return self._lookup_schedule(self.job['pipeline']['id'])


class PipelineAPI(_GitlabAPI):
    def __init__(
          self,
          project_id: int,
          pipeline_id: int,
    ):
        super().__init__(project_id)
        self._pipeline_id = pipeline_id

    @cached_property
    def _pipeline(self) -> gitlab.v4.objects.ProjectPipeline:
        _logger.debug(f"Getting pipeline '{self._pipeline_id}' from Gitlab's API...")
        return self._project.pipelines.get(self._pipeline_id)

    @cached_property
    def pipeline(self) -> dict[str, Any] | None:
        return self._pipeline.attributes

    @cached_property
    def children(self) -> list[dict[str, Any]]:
        _logger.debug(f"Getting child pipelines for pipeline '{self._pipeline_id}' from Gitlab's API...")

        children = []
        for bridge in self._pipeline.bridges.list(get_all=True):
            children.append(self._project.pipelines.get(bridge.attributes['downstream_pipeline']['id']))

        result = [child.attributes for child in children]

        _logger.debug(f"Found {len(result)} child pipelines for pipeline '{self._pipeline_id}':\n{result}!")

        return result

    @cached_property
    def schedule(self) -> dict[str, Any] | None:
        return self._lookup_schedule(self._pipeline_id)

    @cached_property
    def jobs(self) -> list[dict[str, Any]]:
        _logger.debug(f"Getting jobs for pipeline '{self._pipeline_id}' from Gitlab's API...")

        jobs = self._project.pipelines.get(self._pipeline_id).jobs.list(get_all=True)
        result = [job.attributes for job in jobs]

        _logger.debug(f"Found {len(result)} jobs for pipeline '{self._pipeline_id}':\n{result}!")

        return result

    @cached_property
    def downstream_jobs(self) -> list[dict[str, Any]]:
        _logger.debug(f"Getting jobs for child pipelines of pipeline '{self._pipeline_id}' from Gitlab's API...")

        downstream_jobs = []
        for child in self.children:
            downstream_jobs.extend(self._project.pipelines.get(child['id']).jobs.list(get_all=True))

        results = [job.attributes for job in downstream_jobs]

        _logger.debug(f"Found {len(results)} jobs for child pipelines of pipeline '{self._pipeline_id}':\n{results}!")

        return results


# --------------------------------------------------------------------------------------------------
# Public Functions
# --------------------------------------------------------------------------------------------------

# TODO(schmuck): Better error handling here

def _download_job_artifact(
      project_id: int,
      build_id: int,
      artifact_path: str,
) -> str | None:
    _logger.debug(f"Getting artifact '{artifact_path}' for build '{build_id}'...")

    try:
        artifact = (
            _self._gitlab_client
            .projects.get(project_id)
            .jobs.get(build_id)
            .artifact(artifact_path)
        )
    except gitlab.exceptions.GitlabGetError:
        _logger.warning(f"No artifacts found in build '{build_id}'!")
        return None
    except requests.exceptions.ChunkedEncodingError:
        _logger.warning(f"Artifacts found, but none with path '{artifact_path}'!")
        return None

    _logger.info(f"Got artifact '{artifact_path}' for build '{build_id}'!")

    return artifact.decode("utf-8")


def get_pipewatch_artifact(
      project_id: int,
      build_id: int,
) -> lib.PipewatchArtifact | None:
    artifact_raw = _download_job_artifact(
        project_id,
        build_id,
        config.config.artifact_path,
    )

    if artifact_raw is None:
        return None

    try:
        artifact = json.loads(artifact_raw)
    except json.JSONDecodeError:
        raise ArtifactError(f"Failed to decode artifact for build '{build_id}'!")

    return lib.PipewatchArtifact.model_validate(artifact)


# --------------------------------------------------------------------------------------------------
# Lazy Loading
# --------------------------------------------------------------------------------------------------

def __getattr__(
      name: str
) -> Any:
    if name == '_gitlab_client':
        global _gitlab_client

        _logger.info("Connecting to Gitlab...")

        if config.config.gitlab is None:
            raise GitlabError("Gitlab config has to be set!")

        try:
            _gitlab_client = gitlab.Gitlab(
                url=config.config.gitlab.url.unicode_string(),
                private_token=config.config.gitlab.access_token.get_secret_value(),
            )

            _logger.info("Authenticating with Gitlab...")
            _gitlab_client.auth()
        except gitlab.exceptions.GitlabConnectionError:
            raise GitlabError("Failed to connect to Gitlab!")
        except gitlab.exceptions.GitlabAuthenticationError:
            raise GitlabError("Failed to authenticate with Gitlab!")

        return _gitlab_client

    raise AttributeError(name)
