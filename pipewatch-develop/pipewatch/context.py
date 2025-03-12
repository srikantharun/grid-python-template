# --------------------------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------------------------

from pathlib import Path
from typing import Any

import package_lib.context

from pipewatch import config as _config

# --------------------------------------------------------------------------------------------------
# Lazy Loading
# --------------------------------------------------------------------------------------------------


_context = package_lib.context.LazyContext(
    config_class=_config.Config,
    config_file_name=f".{__package__}.toml",
)
config: _config.Config
root_path: Path
config_path: Path


def __getattr__(name: str) -> Any:
    return getattr(_context, name)
