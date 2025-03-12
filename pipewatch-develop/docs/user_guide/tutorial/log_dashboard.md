This tutorial will guide you through the necessary steps to create a visual dashboard to monitor the data produced in your CI jobs' logs.
The tutorial continuously refers to the rest of the documentation for more details on the different steps.

The following prerequisites need to be fulfilled:

- the `pipewatch` CLI is available in your CI environment
- the webhooks are set up correctly for your project
- you have a running instance of the `pipewatch-server`
- you have a Grafana instance running
- you have a MySQL database running

### 1. Configure a job monitor on the server

This step is usually done by the administrator of the server, and only needs to be done upon [deployment](../../backend_server/deployment) of the server.
The goal of this step is to configure

- _which of the jobs should be monitored_ and
- _what job information should be appended_ to your table.

!!! note
    We speak of a **_static_** and a **_dynamic_** table configuration, where the _static_ configuration is done on the server and the _dynamic_ configuration is done by the user and provided to the CLI via a [configuration file](../../cli/usage/#configuration-file).

    The variables of both configurations are then stored in the table of interest.

In order to create the static table configuration, we add an entry to the `tables` list in the [config file](../../backend_server/configuration/#static-table-configuration) of the server:

???+ example

    ```yaml
    tables:
      # ... other tables
      - name: job_monitor
        webhook: job
        skip_unfinished: true
        attach_artifact: true
        filters:
          - event.project.name == 'triton'
          - event.ref == 'main'
        variables:
          - name: build_status
            type: string
            value: event.build_status
          - name: build_duration
            type: float
            value: event.build_duration
          - name: build_failure_reason
            type: string
            value: event.build_failure_reason
          - name: build_started_at
            type: timestamp
            value: event.build_started_at
          - name: runner_description
            type: string
            value: api.runner.description
    ```

    The filter setup above will set up a monitor which selects only jobs that

    - are part of the `triton` project
    - are run on the `main` branch
    - are finished
    - contain a `pipewatch` artifact

These variables will be automatically attached to all tables that are created with `pipewatch`.
This means that if you set up the server correctly, you will always have your data entries tagged with e.g. a **timestamp**, the **build status**, and many more.

!!! tip "Good to know"
    The variable value configurations in the above example are a very simple example of Jinja2 expressions.
    You could go bonkers and do complex inline math if you want to.
    Which means you get **configurable post-processing** capabilities out-of-the-box.

If you change the configuration of the server, you will need to restart the server (or respin the container) for the changes to take effect.

### 2. Define your (dynamic) table variables

The dynamic table configuration is done by the user and provided to the CLI via a [configuration file](../../cli/usage/#configuration-file).

First, tdentify the variables that you want to monitor in your logs.
For this tutorial, we will use the following fictional log output:

???+ example "example_log.log"

    ```text
    [2021-08-31 12:00:00] Network INFO: YOLO performance: 50000 FPS
    [2021-08-31 12:00:05] Network INFO: YOLO accuracy: 0.95
    [2021-08-31 12:00:04] DEBUG: some other log message
    [2021-08-31 12:00:04] WARNING: some warning
    [2021-08-31 12:05:00] Network INFO: Resnet performance: 1000000 FPS
    [2021-08-31 12:05:05] Network INFO: Resnet accuracy: 0.99
    [2021-08-31 12:10:00] Network INFO: EfficientNet performance: 4000 FPS
    [2021-08-31 12:10:00] Network DEBUG: some other log message
    [2021-08-31 12:10:05] Network INFO: EfficientNet accuracy: 0.84
    [2021-08-31 12:15:00] Network INFO: MobileNet performance: 50 FPS
    [2021-08-31 12:15:05] Network INFO: MobileNet accuracy: 0.75
    ```

In this example, we want to monitor the **performance** and **accuracy** of the different networks.

We therefore create the following [configuration file](../../cli/usage/#configuration-file):

???+ example "example_config.yaml"

    ```yaml
    name: network_monitor
    variables:
      name:
        type: string
        regex: 'Network INFO: (\w+) performance:'
      performance:
        type: float
        regex: 'Network INFO: \w+ performance: (\d+) FPS'
      accuracy:
        type: float
        regex: 'Network INFO: \w+ accuracy: (\d+\.\d+)'
    ```

If we now ran the `pipewatch` CLI:

```bash
pipewatch artifact create example_config.yaml example_log.log --parse
```

we would get a table with the following entries:

???+ example "Resulting Table"

    | name          | performance | accuracy |
    |---------------|-------------|----------|
    | YOLO          | 50000       | 0.95     |
    | Resnet        | 1000000     | 0.99     |
    | EfficientNet  | 4000        | 0.84     |
    | MobileNet     | 50          | 0.75     |

!!! info
    The `name` field of the static table configuration is combined with the `name` field of the dynamic table configuration to uniquely identify the table as **`<static_name>:<dynamic_name>`**.
    This is how you will find the table as a data source in Grafana.

The next step is to call the `pipewatch` CLI in your CI job to create the table and forward it to the server.

### 3. Set up the CI job

In your `.gitlab-ci.yml` file, you need to call the `pipewatch` CLI from the job that produces the log output:

???+ example ".gitlab-ci.yml"

    ```yaml
    stages:
      - test

    test:
      stage: test
      script:
        - run_simulation.sh > build.log
        - pipewatch artifact create example_config.yaml build.log --parse
      artifacts:
        paths:
          - pipewatch_data.json
        expire_in: 1 day
    ```

The `pipewatch` CLI will create a `pipewatch_data.json` file in the current working directory.
This file contains the data that will be forwarded to the server.
It is a combination of the configuration and the data that was parsed from the log file.

### 4. Create a Grafana dashboard

The last step is to create a Grafana dashboard that visualizes the data.
Check the [Grafana documentation](https://grafana.com/docs/grafana/latest/panels-visualizations/visualizations/) for more information on how to do this.

The MySQL database should be available as a data source in Grafana.
While your in the panel editor, you can select the table you want to visualize in the _Query_ tab.
It will be available as `<static_name>:<dynamic_name>`.
By now, the table will **contain both the static and the dynamic variables**!

The following screenshot (although not matching this example) shows how to select the table:

![Grafana Query Tab](../../../assets/images/grafana_query_tab.png)

A possible dashboard could look like this (again, not matching this example):

![Grafana Example Dashboard](../../../assets/images/grafana_example_dashboard.png)
