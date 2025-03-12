## Server Setup

For simplified deployment, the `pipewatch-server` is published as a Docker image hosted on our [Gitlab](https://git.axelera.ai/ai-hw-team/pipewatch/pipewatch-server/container_registry) server.
The image is tagged with the version number of the `pipewatch-server` release.
The `latest` tag always points to the latest release.

The guide below focuses on the deployment of the `pipewatch-server` via Docker.

!!! abstract "Container Overview"

    | Property     | Description                                                                            |
    |--------------|----------------------------------------------------------------------------------------|
    | Image        | `git.axelera.ai:5050/ai-hw-team/pipewatch/pipewatch-server/pipewatch-server:<version>` |
    | Architecture | `amd64`                                                                                |
    | Volumes      | `/workdir`, `/logs`                                                                    |
    | Exposed Port | 80                                                                                     |

### Configuration

The `pipewatch-server` allows very fine-grained configuration (see [configuration](../configuration) for details).
Each configuration option can be set via a configuration file or an environment variable.

The configuration file should be named `.pipewatch-server.[toml|yaml|json]` and be placed in a directory that is mounted to `/workdir` in the container.

### Passwords & Authentication

The `pipewatch-server` requires some sensitive information to be set in order to work properly.

These are:

- `api.<endpoint>.auth_token`: The authentication tokens for the endpoints, used to authenticate incoming requests.
- `gitlab.access_token`: The access token for the Gitlab instance, used to query the Gitlab API.
- `database.password`: The password for the database user, used to connect to the database.

The `gitlab.access_token` has to be created on the Gitlab instance.
The other variables can be set arbitrarily.
Use long, random strings for the passwords and tokens.

!!! warning "Attention"
    Although the `pipewatch-server` supports setting these options via the configuration file, it is **strongly recommended** to set them via environment variables.

### Logging

The `pipewatch-server` produces very detailed logs, that can be easily parsed e.g. by `grep`ping for specific sub-loggers, process IDs, etc.
The logs are written in a rotating fashion to `/logs/pipewatch-server.log`.
Each day, a new log file is created and the old ones are kept for 30 days.

!!! tip
    In order to store the logs on the host machine, mount the directory `/logs` to a directory on the host machine.

## Webhook Setup

The `pipewatch-server` exposes two webhook endpoints that it listens to for incoming data (see [API](../api) for details).
If authentication is enabled (which is strongly recommended), the endpoints expect an authentication token in the header.
These tokens then have to be registered on the CI servers.

### Gitlab

The webhooks have to be enabled for each project separately.
The webhook settings for your project can be found under `Settings > Webhooks` in the Gitlab UI of your project or using the URL `https://<gitlab_url>/<group>/<project>/-/hooks`.
For each event type, a separate webhook has to be created.

Follow these steps to create a webhook:

1. Click on `Add new webhook`.
2. Enter the webhook URL:

    ```
    https://<pipewatch_server_url>:<pipewatch_server_port>/webhook/<event_type>
    ```

3. Enter the webhook secret (aka authentication token) in the `Secret oken` field.
4. Select the event type that you chose before.
5. If you don't support SSL encryption, disable the `Enable SSL verification` option.

### Github

Currently, Github is not supported.
It is planned to add support for Github in the future.
