The `pipewatch-server` can be configured via a configuration file and/or with environment variables.
The application expects the configuration to be stored in a file named `.pipewatch-server.[toml|yaml|json]` in the current working directory.
If your using Docker, the configuration file should be mounted to `/workdir` in the container.

!!! tip
    Every configuration option can also be set/overridden via an environment variable.
    The environment variable has to be prefixed with `PIPEWATCH_SERVER__` and the configuration option has to be written in uppercase.
    For example, the configuration option `gitlab.url` can be set via the environment variable `PIPWATCH_SERVER__GITLAB__URL`.

    _**Note:** Environment variables take precedence over the configuration file!_

## Options

The following configuration options are available:

| Configuration option         | Description                                                          | Default value         |
|------------------------------|----------------------------------------------------------------------|-----------------------|
| `auth_enabled`               | Enable endpoint authentication via header.                           | true                  |
| `api.webhook.enabled`        | Enable the webhook endpoints.                                        | true                  |
| `api.webhook.auth_token`     | The authentication token for the webhook endpoints.                  | -                     |
| `api.testing.enabled`        | Enable the testing endpoints.                                        | false                 |
| `api.testing.auth_token`     | The authentication token for the testing endpoint.                   | -                     |
| `gitlab.url`                 | The URL of the GitLab instance.                                      | -                     |
| `gitlab.access_token`        | The access token for the GitLab instance.                            | -                     |
| `database.url`               | The URL of the database (has to start with `mysql://`)               | -                     |
| `database.user`              | The user of the database.                                            | -                     |
| `database.password`          | The user password of the database.                                   | -                     |
| `database.spaces.production` | The name of the production database space.                           | -                     |
| `database.spaces.testing`    | The name of the testing database space.                              | -                     |
| `artifact_path`              | The path to the file where `pipewatch` artifacts are to be expected. | `pipewatch_data.json` |
| `tables`                     | See [static table configuration](#static-table-configuration)        | -                     |
| -                            | The number of workers for the `uvicorn` server.                      | 1                     |

## Static Table Configuration

The `pipewatch-server` allows very fine-grained control over the information that go into the database tables.
The configuration is stored in the configuration file under the key `tables` a list of table configurations.

A table configuration has the following options:

| Configuration option | Description                                                                             | Default value |
|----------------------|-----------------------------------------------------------------------------------------|---------------|
| `name`               | The name of the table as stored in the database.                                        | -             |
| `webhook`            | The webhook to use for this table. One of `job` or `pipeline`.                          | -             |
| `skip_unfinished`    | Whether to skip unfinished jobs/pipelines.                                              | true          |
| `attach_artifact`    | Whether to attach a job artifact to this table. Only available when `webhook == 'job'`. | false         |
| `filters`            | A list of template expressions to apply as filters before processing the table.         | -             |
| `variables`          | A list of variables to store in the table                                               | -             |

The variable objects have the following options:

| Variable option | Description                                                                            |
|-----------------|----------------------------------------------------------------------------------------|
| `name`          | The name of the variable as stored in the database.                                    |
| `type`          | The type of the variable. One of `string`, `integer`, `float`, `boolean`, `timestamp`. |
| `value`         | The template expression used to evaluate the variable value.                           |

### Filters & Variables

Filters and variables enable the user to control if and which data is stored in the database.
They are extremely powerful and versatile due to the use of an underlying [Jinja2](https://jinja.palletsprojects.com/en/3.1.x/) templating engine.

For **variables**, the `value` configuration option is used to specify the template expression.
The **filters** are directly specified as a list of template expressions.
Both are evaluated as Jinja2 expressions (check out their [documentation](https://jinja.palletsprojects.com/en/3.1.x/templates/#expressions) for a syntax guide) with the following variables available in the context:

- `event`: The webhook event object as received from the GitLab webhook (see [GitLab Webhook Events](https://docs.gitlab.com/ee/user/project/integrations/webhook_events.html)).
- `api`: An interface to various [Gitlab API objects](https://docs.gitlab.com/ee/api/api_resources.html) related to the event.

!!! tip
    The **`event`** object is readily available upon evaluation of the template expression.
    It is therefore evaluated _very fast_.

    The **`api`** object on the other hand, makes requests to the GitLab API and is therefore _slower_, but also more powerful.
    It's attributes are lazily evaluated — meaning that they only make requests to the GitLab API if necessary — and cached.

The following attributes are available on the `api` object (depending on the webhook):

| Attribute         | Webhook           | Object Reference                                                                                            | Description                                                          |
|-------------------|-------------------|-------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------|
| `project`         | `job`, `pipeline` | _see [Project API](https://docs.gitlab.com/ee/api/projects.html#get-single-project)_                        | The project the event belongs to.                                    |
| `job`             | `job`             | _see [Job API](https://docs.gitlab.com/ee/api/jobs.html#get-a-single-job)_                                  | The job the event belongs to.                                        |
| `runner`          | `job`             | _see [Runner API](https://docs.gitlab.com/ee/api/runners.html#get-runners-details)_                         | The runner the job was executed on.                                  |
| `pipeline`        | `job`, `pipeline` | _see [Pipeline API](https://docs.gitlab.com/ee/api/pipelines.html#get-a-single-pipeline)_                   | The pipeline the event belongs to.                                   |
| `schedule`        | `job`, `pipeline` | _see [Schedule API](https://docs.gitlab.com/ee/api/pipeline_schedules.html#get-a-single-pipeline-schedule)_ | The schedule the pipeline was triggered by.                          |
| `children`        | `pipeline`        | _see [Pipeline API](https://docs.gitlab.com/ee/api/pipelines.html#get-a-single-pipeline)_                   | A list of the child pipelines, if the pipeline is a parent pipeline. |
| `jobs`            | `pipeline`        | _see [Job API](https://docs.gitlab.com/ee/api/jobs.html#get-a-single-job)_                                  | A list of the jobs of the pipeline.                                  |
| `downstream_jobs` | `pipeline`        | _see [Job API](https://docs.gitlab.com/ee/api/jobs.html#get-a-single-job)_                                  | A list of the jobs of the child pipelines.                           |

For an example, check out the [example server setup](./deployment/example_setup.md).
