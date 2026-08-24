#!/bin/sh
# Pull the policy and the tensorboard logs the cloud host produced. Run from anywhere.
set -e

HOST=${CLOUD_HOST:-cloud}
DEST=${CLOUD_DEST:-Ponderer}

cd "$(dirname "$0")"

mkdir -p policy logs/tb

# No --delete: the cloud is a source of new weights and runs, not the owner of
# the archive.
rsync -rlptvz "$HOST:$DEST/policy/" policy/
rsync -rlptvz "$HOST:$DEST/logs/tb/" logs/tb/
