It is recommended to install this CLI tool via [`pipx`](https://pipx.pypa.io/stable/).

On a `silverlight` or `hetzner` machine, you can install it with the following command:

```bash
pipx install pipewatch
```

If you're on a different machine, you can install it with the following command:

```bash
pipx install pipewatch --pip-args "--extra-index-url http://10.1.5.124/repository/pypi-axe/simple --trusted-host 10.1.5.124"
```

Of course, you can also install it in a virtual environment with `pip`.
It won't be as convenient to use, however.
