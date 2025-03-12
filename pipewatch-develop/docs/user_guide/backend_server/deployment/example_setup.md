On your host, create a directory for the configuration and the logs:

```bash
mkdir -p /opt/pipewatch-server/logs
mkdir -p /opt/pipewatch-server/config
```

Create a configuration file, where you store all non-sensitive configuration options:

```bash
touch /opt/pipewatch-server/config/.pipewatch-server.toml
```

The following configuration file can serve as an inspiration:

???+ example "Example Configuration"
    ```yaml
    auth_enabled: true
    api:
      webhook:
        enabled: true
      testing:
        enabled: false
    gitlab:
      url: https://git.axelera.ai
    database:
      url: mysql://localhost:3306
      user: axelera
      spaces:
        production: pipewatch
    tables:
      - name: pipelines_info
        webhook: pipeline
        skip_unfinished: true
        filters:
          - event.project.name == 'triton'
          - event.object_attributes.source == 'schedule'
        variables:
          - name: schedule_description
            type: string
            value: api.schedule.description
          - name: created_at
            type: timestamp
            value: event.object_attributes.created_at
          - name: status
            type: string
            value: event.object_attributes.status
      - name: artifact_table
        webhook: job
        skip_unfinished: true
        attach_artifact: true
        filters:
          - event.project.name == 'triton'
        variables:
          - name: ref
            type: string
            value: event.ref
          - name: pipeline_id
            type: integer
            value: event.pipeline_id
          - name: build_id
            type: integer
            value: event.build_id
          - name: build_status
            type: string
            value: event.build_status
    ```

With docker, log onto the Gitlab registry with your user account:

```bash
docker login git.axelera.ai:5050
```

Pull the latest version of the `pipewatch-server` Docker image:

```bash
docker pull git.axelera.ai:5050/ai-hw-team/pipewatch/pipewatch-server/pipewatch-server:latest
```

Start the `pipewatch-server` container:

```bash
docker run -d \
    --name pipewatch-server \
    -v /opt/pipewatch-server/logs:/logs \
    -v /opt/pipewatch-server/config:/workdir \
    -p <your-port>:80 \
    --env UVICORN_WORKERS=4 \
    --env PIPEWATCH_SERVER__GITLAB__ACCESS_TOKEN=<your-gitlab-access-token> \
    --env PIPEWATCH_SERVER__DATABASE__PASSWORD=<your-database-password> \
    --env PIPEWATCH_SERVER__API__WEBHOOKS__AUTH_TOKEN=<your-webhook-api-token> \
    git.axelera.ai:5050/ai-hw-team/pipewatch/pipewatch-server/pipewatch-server:latest
```
