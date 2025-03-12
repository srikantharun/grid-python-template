# --------------------------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------------------------

import logging
from typing import *

import fastapi
import package_lib.api
import pipewatch_lib as lib
import pydantic
from package_lib.exceptions import HandledException

from pipewatch_server import (
    config,
    core,
    model
)

# --------------------------------------------------------------------------------------------------
# Exceptions
# --------------------------------------------------------------------------------------------------

class APIError(HandledException):
    """Raised when an API error occurs."""
    pass


# --------------------------------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------------------------------

# TODO(schmuck): Make sure that only non-null data is sent! (on all server interfaces)

__version__ = "0.3.8"

app = fastapi.FastAPI(
    title="Pipewatch Server API",
    version=__version__,
)

# --------------------------------------------------------------------------------------------------
# Module Variables
# --------------------------------------------------------------------------------------------------

_logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------------------
# Exception Handlers
# --------------------------------------------------------------------------------------------------

# Create the default exception handler
package_lib.api.create_exception_handler(
    app,
    exception_type=HandledException,
)

# Create the exception handler for pydantic validation errors
package_lib.api.create_exception_handler(
    app,
    exception_type=pydantic.ValidationError,
)

# --------------------------------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------------------------------

if config.config.api.webhook.enabled:
    @app.post("/webhook/pipeline")
    async def pipeline_webhook(
          event: Annotated[
              model.PipelineEvent, fastapi.Body()
          ],
          auth_token: Annotated[
              str, fastapi.Header(
                  alias="X-Gitlab-Token",
              )
          ] = None,
    ) -> None:
        if config.config.auth_enabled:
            if auth_token is None:
                raise APIError("Missing API token!")

            if auth_token != config.config.api.webhook.auth_token.get_secret_value():
                raise APIError("Invalid API token!")

        core.process_webhook_event(event, model.WebhookName.PIPELINE)


    @app.post("/webhook/job")
    async def job_webhook(
          event: Annotated[
              model.JobEvent, fastapi.Body()
          ],
          auth_token: Annotated[
              str, fastapi.Header(
                  alias="X-Gitlab-Token",
              )
          ] = None,
    ) -> None:
        if config.config.auth_enabled:
            if auth_token is None:
                raise APIError("Missing API token!")

            if auth_token != config.config.api.webhook.auth_token.get_secret_value():
                raise APIError("Invalid API token!")

        core.process_webhook_event(event, model.WebhookName.JOB)

if config.config.api.testing.enabled:
    @app.post("/testing/artifact")
    async def test_artifact_webhook(
          artifact: Annotated[
              lib.PipewatchArtifact, fastapi.Body()
          ],
          auth_token: Annotated[
              str, fastapi.Header(
                  alias="X-Auth-Token",
              )
          ] = None,
    ) -> None:
        if config.config.auth_enabled:
            if auth_token is None:
                raise APIError("Missing API token!")

            if auth_token != config.config.api.testing.auth_token.get_secret_value():
                raise APIError("Invalid API token!")

        core.process_test_artifact(artifact)
