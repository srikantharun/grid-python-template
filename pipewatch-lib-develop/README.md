# pipewatch-lib

![release](https://git.axelera.ai/ai-hw-team/pipewatch/pipewatch-lib/-/badges/release.svg)
![pipeline](https://git.axelera.ai/ai-hw-team/pipewatch/pipewatch-lib/badges/develop/pipeline.svg?ignore_skipped=true)
![coverage](https://git.axelera.ai/ai-hw-team/pipewatch/pipewatch-lib/badges/develop/coverage.svg?job=test:pytest)

<a href="http://ai-hw-team.doc.axelera.ai/pipewatch/pipewatch-lib"><img alt="Static Badge" src="https://img.shields.io/badge/Documentation-orange?logo=readthedocs&logoColor=white"></a>

## Documentation

The main documentation for this project can be found [here](http://ai-hw-team.doc.axelera.ai/pipewatch/pipewatch-lib).

## Installation

If you want to use this library, add this package to your `pyproject.toml` file:

```toml
[tool.poetry.dependencies]
pipewatch-lib = "<replace with your version requirement>"

[[tool.poetry.source]]
name = "nexus"
url = "http://10.1.5.124/repository/pypi-axe/simple"
priority = "supplemental"
```

When you use `pip` to install this package, you need to add the following option:

```bash
pip install pipewatch-lib --extra-index-url http://10.1.5.124/repository/pypi-axe/simple --trusted-host 10.1.5.124
```

## Development

The development environment is managed with [package-tools](http://tools.doc.axelera.ai/py/dev/package-tools). It can be installed with make:

```bash
make install.package-tools
```

In the background, this will install `package-tools` via `pipx` and make it available to you as a command line tool. Afterwards, you can run it with:

```bash
package-tools --help
```

For more information on how to use `package-tools`, please refer to the [documentation](http://tools.doc.axelera.ai/py/dev/package-tools).
