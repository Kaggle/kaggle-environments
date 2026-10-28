# This is set in cloudbuild.yaml, but provide a default
ARG BASE_IMAGE=gcr.io/kaggle-images/python:v163

# ===== Base Stages =====
FROM node:22.23.4-slim AS node_builder
FROM ${BASE_IMAGE} AS base

# Distrust the compromised emSign ECC Root CA - G3 (http://b/575927270). The CPU base is
# Ubuntu jammy and the GPU base is noble; both are patched in 20260816ubuntu0.{22,24}.04.1.
RUN apt-get update && \
    apt-get install --no-install-recommends -y --only-upgrade ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    dpkg --compare-versions "$(dpkg-query -W -f='${Version}' ca-certificates)" ge 20260816ubuntu0
# Make grpcio use the patched OS bundle instead of its compiled-in roots.
ENV GRPC_DEFAULT_SSL_ROOTS_FILE_PATH=/etc/ssl/certs/ca-certificates.crt

COPY --from=node_builder /usr/local/ /usr/local/
# Confirm the installation was successful
RUN node -v && npm -v

WORKDIR /usr/src/app/kaggle_environments

# Copy everything required for the build.
# This includes the dependency list, the source code, and manifest.
COPY ./pyproject.toml ./pyproject.toml
COPY ./MANIFEST.in ./MANIFEST.in
COPY ./kaggle_environments ./kaggle_environments

# Pin the version reported by the installed package.
ARG REPORTED_VERSION=1.32.7
RUN sed -i "s/^version = \".*\"$/version = \"${REPORTED_VERSION}\"/" pyproject.toml && \
    grep -q "^version = \"${REPORTED_VERSION}\"$" pyproject.toml

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
