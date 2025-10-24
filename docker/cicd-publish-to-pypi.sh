#!/bin/bash
set -e

echo "Installing git..."
apt-get update -y && apt-get install -y git --no-install-recommends

pip install requests --quiet

PACKAGE_NAME="kaggle-environments"
VERSION_FROM_PYPROJECT=$(grep -E "^version\s*=\s*\"[0-9]+\.[0-9]+\.[0-9]+\"$" pyproject.toml | cut -d '"' -f 2)
CLOUD_RUN_REQS_PATH="docker/cloud-run-source/requirements.txt"

if [ -z "$VERSION_FROM_PYPROJECT" ]; then
  echo "❌ Could not find version in pyproject.toml. Exiting."
  exit 1
fi

echo "Found version $VERSION_FROM_PYPROJECT in pyproject.toml. Checking against PyPI."

cat > /tmp/check_pypi.py << EOL
import sys
import requests
package_name = '$PACKAGE_NAME'
version_to_check = sys.argv[1]
response = requests.get(f'https://pypi.org/pypi/{package_name}/json')
if response.status_code == 200:
    data = response.json()
    if version_to_check in data.get('releases', {}):
        print('true')
    else:
        print('false')
elif response.status_code == 404:
    print('false')
else:
    print(f'Error checking PyPI: {response.status_code}', file=sys.stderr)
    sys.exit(1)
EOL

VERSION_EXISTS=$(python3 /tmp/check_pypi.py "$VERSION_FROM_PYPROJECT")

if [ "$VERSION_EXISTS" = "true" ]; then
  echo "⏩ Version $VERSION_FROM_PYPROJECT already exists on PyPI. Skipping publish."
else
  echo "🚀 Version $VERSION_FROM_PYPROJECT not found on PyPI. Publishing..."
  pip install flit --quiet
  export FLIT_USERNAME=__token__
  export FLIT_PASSWORD=$PYPI_TOKEN

  flit publish
  
  # It takes a bit for the package to show up on PyPI. Make sure it's visible through Pip before proceeding.
  # --- RETRY LOGIC START ---
  echo "Verifying that package '$PACKAGE_NAME==$VERSION_FROM_PYPROJECT' is available from PyPI..."
  MAX_RETRIES=10
  RETRY_DELAY_SECONDS=30
  SUCCESS=false

  for i in $(seq 1 $MAX_RETRIES)
  do
    if python -m pip install --no-cache-dir "$PACKAGE_NAME==$VERSION_FROM_PYPROJECT" --quiet; then
      echo "✅ Successfully found and installed package from PyPI."
      SUCCESS=true
      break
    else
      echo "Attempt $i/$MAX_RETRIES: Package not yet available. Waiting ${RETRY_DELAY_SECONDS}s..."
      sleep $RETRY_DELAY_SECONDS
    fi
  done

  if [ "$SUCCESS" != "true" ]; then
    echo "❌ Failed to find package on PyPI after $MAX_RETRIES attempts. Aborting."
    exit 1
  fi
  # --- SAFETY BUFFER ---
  # The package is available, but let's wait a bit longer for CDN propagation.
  echo "Waiting 45 seconds as a safety buffer for CDN propagation..."
  sleep 45
  # --- RETRY LOGIC END ---

  echo "Successfully published and verified. Updating Cloud Run requirements at $CLOUD_RUN_REQS_PATH..."
  
  sed -i "s/VERSION_FROM_CICD_DEPLOY/${VERSION_FROM_PYPROJECT}/" "$CLOUD_RUN_REQS_PATH"
  
  echo "Cloud Run requirements.txt updated."

  # Create a flag file to signal success
  echo "true" > /workspace/published_new_version.flag
fi
