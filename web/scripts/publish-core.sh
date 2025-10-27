#!/bin/bash
set -e

echo "Installing dependencies..."
npm install -g pnpm
pnpm install

echo "Building core library..."
pnpm --filter @kaggle-environments/core build

PACKAGE_NAME="@kaggle-environments/core"
CORE_PACKAGE_JSON_PATH="web/core/package.json"
VERSION_FROM_PACKAGE_JSON=$(grep -E '"version":\s*".*"' "$CORE_PACKAGE_JSON_PATH" | sed -E 's/.*"version":\s*"(.*)".*/\1/')

if [ -z "$VERSION_FROM_PACKAGE_JSON" ]; then
  echo "❌ Could not find version in $CORE_PACKAGE_JSON_PATH. Exiting."
  exit 1
fi

echo "Found version $VERSION_FROM_PACKAGE_JSON. Proceeding to publish/verify loop."


MAX_RETRIES=3
RETRY_DELAY_SECONDS=20
SUCCESS=false

echo "//registry.npmjs.org/:_authToken=${NPM_TOKEN}" > web/core/.npmrc

set +e

for i in $(seq 1 $MAX_RETRIES)
do
  echo "--- Attempt $i/$MAX_RETRIES ---"

  # 1. Check if the version now exists on the registry.
  echo "Checking NPM registry for $PACKAGE_NAME@$VERSION_FROM_PACKAGE_JSON..."
  if npm view "$PACKAGE_NAME@$VERSION_FROM_PACKAGE_JSON" version > /dev/null; then
    echo "✅ Version found on NPM - Skipping Publish"
    SUCCESS=true
    break
  fi

  # 2. If not found, attempt to publish.
  echo "Version not found. Attempting to publish..."
  if pnpm --filter @kaggle-environments/core publish --no-git-checks; then
    echo "Publish command succeeded. Will verify on the next attempt after a delay."
  else
    echo "Publish command failed."
  fi

  # 3. If this wasn't the last attempt, wait before retrying.
  if [ "$i" -lt "$MAX_RETRIES" ]; then
    echo "Waiting ${RETRY_DELAY_SECONDS}s before next attempt..."
    sleep $RETRY_DELAY_SECONDS
  fi
done

# Cleanup the .npmrc file regardless of outcome
rm web/core/.npmrc

# Re-enable failing on error
set -e

if [ "$SUCCESS" != "true" ]; then
  echo "❌ Failed to publish or verify package after $MAX_RETRIES attempts. Aborting."
  exit 1
fi

echo "🚀 Process complete."
