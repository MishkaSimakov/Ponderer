#!/bin/sh
# Pull the policy the cloud host trained into policy/. Run from anywhere.
set -e

HOST=${CLOUD_HOST:-cloud}
DEST=${CLOUD_DEST:-Ponderer}

cd "$(dirname "$0")"

mkdir -p policy

# No --delete: the cloud is a source of new weights, not the owner of the archive.
rsync -rlptvz "$HOST:$DEST/policy/" policy/
