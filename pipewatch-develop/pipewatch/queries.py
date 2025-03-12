# --------------------------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------------------------

import logging
from typing import Any

import package_lib.exceptions
import pipewatch_lib as lib
import requests

from pipewatch import context

# --------------------------------------------------------------------------------------------------
# Exceptions
# --------------------------------------------------------------------------------------------------


class RequestError(package_lib.exceptions.HandledException):
    """Raised when a request fails."""


class ServerError(package_lib.exceptions.HandledException):
    """Raised when a request fails."""


# --------------------------------------------------------------------------------------------------
# Module Variables
# --------------------------------------------------------------------------------------------------

_logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------------


def _post_data(
    endpoint: str,
    body: dict[str, Any],
) -> Any:
    """
    Send a REST API request to the backend server.

    Args:
    ----
        endpoint: The endpoint to send the request to.
        body: The request body.

    Returns:
    -------
        The response body.

    """
    if context.config.server is None:
        msg = "Server configuration is missing!"
        raise RequestError(msg)

    _logger.info(
        "Sending request to backend server '%s'...",
        context.config.server.url.unicode_string() + endpoint,
    )
    _logger.debug("Request body:\n%s", body)

    try:
        response = requests.post(
            context.config.server.url.unicode_string() + endpoint,
            headers={
                "X-Auth-Token": context.config.server.auth_token.get_secret_value(),
            },
            json=body,
            timeout=context.config.server.timeout,
        )
    except requests.RequestException as e:
        msg = f"Request to backend failed with exception: {e}"
        raise RequestError(msg) from e

    _logger.debug("Response:\n%s", response.text)

    try:
        response.raise_for_status()
    except requests.HTTPError as e:
        msg = f"Server returned status code {response.status_code}:\n{response.text}"
        raise ServerError(msg) from e

    return response.json()


# --------------------------------------------------------------------------------------------------
# Public Functions
# --------------------------------------------------------------------------------------------------


def post_artifact(
    artifact: lib.PipewatchArtifact,
) -> None:
    """
    Post an artifact to the backend server's `testing/artifact` endpoint.

    Args:
    ----
        artifact: The artifact to post.

    """
    _post_data(
        "testing/artifact",
        body=artifact.model_dump(mode="json"),
    )
