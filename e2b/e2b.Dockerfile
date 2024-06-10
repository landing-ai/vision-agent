FROM e2bdev/code-interpreter:latest
ENV POETRY_HOME="/opt/poetry"
ENV PATH="$POETRY_HOME/bin:$PATH"
# Disable virtualenv creation for convenience
ENV POETRY_VIRTUALENVS_CREATE=false
# Check if startup_script.sh exists and modify it
RUN if [ -f /root/.jupyter/start-up.sh ]; then \
      sed -i '2i export PYTHONPATH=/home/user/app' /root/.jupyter/start-up.sh; \
    else \
      echo "Error: startup_script.sh does not exist" >&2; \
      exit 1; \
    fi
# Install system dependencies
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
  curl ffmpeg vim build-essential \
  && rm -rf /var/lib/apt/lists/*

# Install Poetry and Python dependencies
RUN curl -sSL https://install.python-poetry.org | python -
WORKDIR /home/user/app
# Copy only the dependencies definition to leverage Docker cache
COPY pyproject.toml poetry.lock /home/user/app/
RUN poetry install --no-root --no-interaction --without dev
COPY . /home/user/app
