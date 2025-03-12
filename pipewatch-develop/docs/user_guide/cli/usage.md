The `pipewatch` CLI is the main entry point for the `pipewatch` ecosystem.
It serves as a helper to validate & create the _artifact files_.
You will most likely only call it from within your CI/CD pipelines.

An _artifact file_ will contain the configuration and data that describe a table and its entries.

!!! Info
    A table can contain one or more rows and will be

    - created if it doesn't exist yet
    - appended if there already is a table with the same name
    - column-expanded if the new data contains more columns than the existing table
    - none-filled if the new data contains fewer columns than the existing table

## Commands

The easiest way to get an overview of the available commands is to run `pipewatch --help`.

??? abstract "`pipewatch --help`"

    ```bash
     Usage: pipewatch [OPTIONS] COMMAND [ARGS]...

     A command-line interface for the Python Package Template.

    ╭─ Options ─────────────────────────────────────────────────────────────────────────────────────────────────────╮
    │ --version                                                       Print the version and exit.                   │
    │ --log-level                [debug|info|warning|error|critical]  Set the log level. If not set, no logs will   │
    │                                                                 be printed.                                   │
    │ --log-filters              TEXT                                 Set the log filters.                          │
    │ --log-file                 TEXT                                 Set the log file. If not set, logs will be    │
    │                                                                 printed to stdout.                            │
    │ --log-config-file          TEXT                                 Set the log config file. If this is set, no   │
    │                                                                 other logging options can be set.             │
    │ --help             -h                                           Show this message and exit.                   │
    ╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
    ╭─ Commands ────────────────────────────────────────────────────────────────────────────────────────────────────╮
    │ artifact                       Artifact commands.                                                             │
    │ config                         Configuration commands.                                                        │
    ╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
    ```

You can run `pipewatch <command> --help` to get more information about a specific command and sub-commands.

You will likely only use the `artifact` command and its subcommands:

```bash
pipewatch artifact validate [--parse] <config_file> <data_file>
pipewatch artifact create [--parse] <config_file> <data_file> [--output-dir <output_file>]

# The following command is only for testing purposes and requires an API token
pipewatch artifact post [--parse] <config_file> <data_file>
```

Use `validate` to validate if your configuration and data files match and if they can be posted to the server.

!!! warning
    `pipewatch` can currently not validate whether your configuration has been overwritten, which can lead to an error in the backend server, if, for example, you changed the data type of a variable.

## Files

The following sections will show you the expected input file formats.

### Configuration File

The configuration file holds the information about the table that you want to store in the database.
It can be written in `toml`, `yaml` or `json` format.
For now, we don't support nested data structures.

```yaml
name: <table_name>
variables:
  <variable_name_a>:
    type: <data_type_a>
    regex: <regex_a>
  <variable_name_b>:
    type: <data_type_b>
    regex: <regex_b>
```

The `type` field is required and can be one of the following:

- `string`
- `integer`
- `float`
- `boolean`
- `timestamp` (format: `YYYY-MM-DD HH:MM:SS` or [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601))

It is used to validate the data in the data file and defines the format of the data in the database.

#### Regex Expressions

The `regex` field is optional and can be provided if you want to use `pipewatch`'s parsing functionality.
If you don't provide `regex` expressions, you will have to construct the data file yourself.

!!! tip
    Most of the time, you will only want a part of the regex match to become the variable value.
    For this, use a single [capturing group](https://www.regular-expressions.info/refcapture.html) in your regex.
    If you don't use a capturing group, the whole match will become the variable value.

    _Example_:<br>
    `regex: ".*?([0-9]+).*"` will match the first number in a string

You either have to leave out the `regex` field or provide it for all variables.

_See the section on [raw text files](#raw-text-file) for more information._

### Data File

For data files, you have two options:

- Provide a raw text file and let `pipewatch` parse it with the regex expressions from the configuration file.
- Provide a parsed data file that already contains the data in the correct format. Supported formats are `csv`, `json`, `yaml` and `toml`.

#### Raw Text File

If your data don't need special preprocessing, they can often be parsed with regex expressions.
In that case, you can use the `--parse` flag to let `pipewatch` parse the data for you (see [commands](#commands)).

!!! info
    The text file can contain **either one or multiple occurrences of each variable**.
    Each of the variables that occur multiple times, have to occur the same number of times.
    The variables that occur only once, will be broadcast.
    This way, the data can always be stored in a table.

#### CSV

If you're using the CSV format, the data file should have the following format:

```csv
<variable_name_a>,<variable_b>
<value_a>,<value_b>
```

You can also provide multiple data entries in one file:

```csv
<variable_name_a>,<variable_b>
<value_a_0>,<value_b_0>
<value_a_1>,<value_b_1>
```

#### JSON-like

If you're using JSON, YAML or TOML for your data file, it should have the following format:

```yaml
<variable_name_a>: <value_a>
<variable_b>: <value_b>
```

You can also provide  multiple data entries in one file:

```yaml
<variable_name_a>: [<item_a_0>, <item_a_1>]
<variable_b>: [<item_b_0>, <item_b_1>]
```
