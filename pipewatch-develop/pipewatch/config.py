# --------------------------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------------------------

import package_lib.config
import pydantic

# --------------------------------------------------------------------------------------------------
# Base Classes
# --------------------------------------------------------------------------------------------------


class _CustomBaseSettings(
    package_lib.config.BaseSettings,
    env_prefix=f"{__package__}__",
):
    pass


# --------------------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------------------


class Server(_CustomBaseSettings):
    url: pydantic.HttpUrl
    auth_token: pydantic.SecretStr

    timeout: float = 10.0


class Config(_CustomBaseSettings):
    artifact_path: str = "pipewatch_data.json"
    server: Server | None = None
