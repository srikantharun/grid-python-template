# pipewatch

![release](https://git.axelera.ai/ai-hw-team/pipewatch/pipewatch/-/badges/release.svg)
![pipeline](https://git.axelera.ai/ai-hw-team/pipewatch/pipewatch/badges/develop/pipeline.svg?ignore_skipped=true)
![coverage](https://git.axelera.ai/ai-hw-team/pipewatch/pipewatch/badges/develop/coverage.svg?job=test:pytest)

<a href="http://ai-hw-team.doc.axelera.ai/pipewatch/pipewatch"><img alt="Static Badge" src="https://img.shields.io/badge/Documentation-orange?logo=readthedocs&logoColor=white"></a>

## Documentation

The main documentation for this project can be found [here](http://ai-hw-team.doc.axelera.ai/pipewatch/pipewatch).

## Installation

It is recommended to install this CLI tool via `pipx`.

On a `silverlight` or `hetzner` machine, you can install it with the following command:

```bash
pipx install pipewatch
```

If you're on a different machine, you can install it with the following command:

```bash
pipx install pipewatch --pip-args "--extra-index-url http://10.1.5.124/repository/pypi-axe/simple --trusted-host 10.1.5.124"
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
