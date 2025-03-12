In order to set up a pipeline monitor, only a subset of the steps described in the [CI data dashboard tutorial](../log_dashboard) are required.
The same prerequisites apply, and only steps 1 and 4 need to be followed.
Step 4 is identical to the one described in the [CI data dashboard tutorial](../log_dashboard), and step 1 is described below.

### 1. Configure a pipeline monitor on the server

Since the pipeline monitor does not depend on dynamic data created in the CI jobs, the static configuration is sufficient.

We add an entry to the `tables` list in the [config file](../../backend_server/configuration/#static-table-configuration) of the server:

???+ example

    ```yaml
    tables:
      # ... other tables
      - name: pipeline_monitor
        webhook: pipeline
        skip_unfinished: true
        filters:
          - event.project.name == 'triton'
          - event.ref == 'main'
          - event.object_attributes.source == 'schedule'
        variables:
          - name: schedule_description
            type: string
            value: api.schedule.description
          - name: schedule_id
            type: integer
            value: api.schedule.id
          - name: pipeline_id
            type: integer
            value: event.object_attributes.id
          - name: created_at
            type: timestamp
            value: event.object_attributes.created_at
          - name: duration
            type: float
            value: event.object_attributes.duration
          - name: status
            type: string
            value: event.object_attributes.status
          - name: successful_job_ratio
            type: float
            value: >
              event.builds | selectattr('status', '==', 'success') | list | length /
              event.builds | length
    ```

    The filter setup above will set up a monitor which selects only pipelines that

    - are part of the `triton` project
    - are run on the `main` branch
    - are triggered by a pipeline schedule

The above example also shows how to utilise complex Jinja2 [expressions](https://jinja.palletsprojects.com/en/3.1.x/templates/#expressions) and [filters](https://jinja.palletsprojects.com/en/3.1.x/templates/#list-of-builtin-filters) to create new variables from the data provided by the webhook.
