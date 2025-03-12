# --------------------------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------------------------

import abc
import logging
import sys
from typing import *

import package_lib.config
import pipewatch_lib as lib
import pydantic
from typing_extensions import Self

from pipewatch_server import model

# --------------------------------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------------------------------

__all__ = [
    'config',
]

# --------------------------------------------------------------------------------------------------
# Module Variables
# --------------------------------------------------------------------------------------------------

_self = sys.modules[__name__]

config: 'Config'

_logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------------------
# Base Classes
# --------------------------------------------------------------------------------------------------

class _CustomBaseSettings(
    package_lib.config.BaseSettings,
    extra='forbid',
    env_prefix=f'{__package__}__',
):
    pass


# --------------------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------------------

Template = Annotated[
    str,
    pydantic.Field(
        description="A Jinja2 template string.",
        example="event.name",
    ),
]


class Variable(_CustomBaseSettings):
    name: str
    type: lib.DataTypeName
    value: Template


class _TableBase(_CustomBaseSettings, abc.ABC):
    webhook: model.WebhookName
    name: str
    filters: list[Template]
    variables: list[Variable]


class JobTable(_TableBase):
    webhook: Literal[model.WebhookName.JOB]
    attach_artifact: bool = False
    skip_unfinished: bool = True


class PipelineTable(_TableBase):
    webhook: Literal[model.WebhookName.PIPELINE]
    skip_unfinished: bool = True


class Endpoint(_CustomBaseSettings):
    enabled: bool
    auth_token: pydantic.SecretStr | None = None


class Gitlab(_CustomBaseSettings):
    url: pydantic.HttpUrl
    access_token: pydantic.SecretStr


class Database(_CustomBaseSettings):
    url: Annotated[pydantic.AnyUrl, pydantic.UrlConstraints(allowed_schemes=['mysql'])]

    user: str | None = None
    password: pydantic.SecretStr

    spaces: Annotated[dict[model.DatabaseSpaceName, str], pydantic.Field(min_items=1)]


class API(_CustomBaseSettings):
    webhook: Endpoint = Endpoint(enabled=True)
    testing: Endpoint = Endpoint(enabled=False)


class Config(_CustomBaseSettings):
    auth_enabled: bool = True

    api: API = API()

    database: Database
    gitlab: Gitlab | None = None

    tables: list[
        Annotated[
            PipelineTable | JobTable,
            pydantic.Field(discriminator='webhook'),
        ]
    ]

    artifact_path: str = 'pipewatch_data.json'

    @pydantic.model_validator(mode='after')
    def validate_authentication(self) -> Self:
        if self.auth_enabled:
            assert self.api.webhook.auth_token is not None, \
                "Authentication is enabled, but no auth token is set for the 'webhook' endpoint"

            if self.api.testing.enabled:
                assert self.api.testing.auth_token is not None, \
                    "Authentication is enabled, but no auth token is set for the 'testing' endpoint"

        return self


# --------------------------------------------------------------------------------------------------
# Lazy Loading
# --------------------------------------------------------------------------------------------------

def __getattr__(
      name: str
) -> Any:
    if name == 'config':
        global config

        config = package_lib.config.load_config(
            file_name=f'.{__package__}'.replace('_', '-'),
            config_model=Config,
            try_extensions=True,
        )
        _logger.debug(f"Config: {config}")

        return config

    raise AttributeError(name)


# Trigger lazy loading immediately
_ = _self.config
