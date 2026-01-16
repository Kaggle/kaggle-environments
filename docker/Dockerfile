# This is set in cloudbuild.yaml, but provide a default
ARG BASE_IMAGE=gcr.io/kaggle-images/python:v163

# ===== Base Stages =====
FROM node:22-slim AS node_builder
FROM ${BASE_IMAGE} AS base

COPY --from=node_builder /usr/local/ /usr/local/
# Confirm the installation was successful
RUN node -v && npm -v

WORKDIR /usr/src/app/kaggle_environments

# Copy everything required for the build.
# This includes the dependency list, the source code, and manifest.
COPY ./pyproject.toml ./pyproject.toml
COPY ./MANIFEST.in ./MANIFEST.in
COPY ./kaggle_environments ./kaggle_environments

# Now that the source code is present, run the installation.
# This layer is now correctly cached and will only rebuild if the source or pyproject.toml changes.
RUN uv pip install --system . && \
    uv cache clean

# 3. Copy any remaining files that aren't part of the build.
COPY ./README.md ./README.md

# ===== CPU Final Stage =====
FROM base AS cpu
CMD ["kaggle-environments"]

# ===== GPU Final Stage =====
FROM base AS gpu
CMD ["kaggle-environments"]
