from pathlib import Path

from mkdocs_macros_helper.decorators import *

# --------------------------------------------------------------------------------------------------
# Macros
# --------------------------------------------------------------------------------------------------

@macro
def html_image(
      path: str | Path,
      alt: str = '',
      width: int = None,
      align: str = 'center',
) -> str:
    return f'<div style="text-align:{align};">' \
           f'<img src="{path}" alt="{alt}" width="{width}">' \
           f'</div>'
