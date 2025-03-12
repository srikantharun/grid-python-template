The `pipewatch-server`'s API is rather simple.
Depending on the [configuration](../configuration), it exposes the following [HTTP `POST`](https://developer.mozilla.org/en-US/docs/Web/HTTP/Methods/POST) endpoints:

- **Webhook Endpoints**: The webhook endpoints receive data from [Gitlab Webhooks](https://docs.gitlab.com/ee/user/project/integrations/webhooks.html).
- **Testing Endpoint**: The testing endpoint can be used to directly `POST` data to the server for testing purposes.

In the future, more direct endpoints might be added to the server, e.g. to directly query the database, or create dashboards from data produced outside of a CI pipeline.

#### Authentication

The `pipewatch-server` supports authentication via a header token.
If authentication is enabled, the endpoints expect an authentication token in the header.
The token can be set on the server side via a configuration option (see [configuration](../configuration) for details).

!!! danger
    If you're sending authentication tokens in the header, make sure to use SSL encryption (HTTPS)!
    Alternatively, keep the communication within a trusted network.

    When compromised, the authentication token can be used to corrupt the database.

## Endpoints

### Webhook Endpoints

The webhook endpoints are meant to receive data from [Gitlab Webhooks](https://docs.gitlab.com/ee/user/project/integrations/webhooks.html).
For authentication, both endpoints expect the header `X-Gitlab-Token` to be present.

#### `/webhook/pipeline`

##### Headers

| Header           | Description                                        |
|------------------|----------------------------------------------------|
| `X-Gitlab-Token` | The authentication token for the webhook endpoint. |

##### Body

The body of the request is expected to be a [pipeline event](https://docs.gitlab.com/ee/user/project/integrations/webhook_events.html#pipeline-events) JSON object.

#### `/webhook/job`

##### Headers

| Header           | Description                                        |
|------------------|----------------------------------------------------|
| `X-Gitlab-Token` | The authentication token for the webhook endpoint. |

##### Body

The body of the request is expected to be a [job event](https://docs.gitlab.com/ee/user/project/integrations/webhook_events.html#job-events) JSON object.

### Testing Endpoint

#### `/testing`

##### Headers

| Header         | Description                                        |
|----------------|----------------------------------------------------|
| `X-Auth-Token` | The authentication token for the testing endpoint. |

##### Body

The body of the request is expected to be a _`pipewatch` artifact_ (see) TODO.

### CURL Examples

The webhook endpoints can be tested with `curl` as follows:

```bash
curl -X POST \
    -H "X-Gitlab-Token: <webhook_token>" \
    -H "Content-Type: application/json" \
    -d @<webhook_event>.json \
    http://<host>:<port>/webhook/<event>
```

The testing endpoint can be tested with `curl` as follows:

```bash
curl -X POST \
    -H "X-Auth-Token: <testing_token>" \
    -H "Content-Type: application/json" \
    -d @<artifact>.json \
    http://<host>:<port>/testing
```
