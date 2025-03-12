# Build the install image
FROM python:3.10-slim AS install

# Get the pip configuration
COPY pip.conf /etc/pip.conf

# Get the application
COPY . /app

# Install the app
RUN \
    pip install --user --no-cache-dir --upgrade \
    pip \
    concurrent-log-handler \
    /app

# Build the runtime image
FROM python:3.10-slim AS run

# Get the python files
COPY --from=install /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Install all required packages
RUN apt-get update &&  \
    apt-get install -y \
    git

# Get the log configuration
RUN mkdir -p /config
COPY log_conf_docker.yaml /config/log_conf.yaml

# Set up the volumes
VOLUME /logs
VOLUME /workdir

# Work in the project directory
WORKDIR /workdir

EXPOSE 80

# Run the server
ENTRYPOINT ["uvicorn", "pipewatch_server.api:app"]
CMD ["--host", "0.0.0.0", "--port", "80", "--log-config", "/config/log_conf.yaml"]
