#!/bin/sh
# Pull the runs the brick recorded into logs/brick. Run from anywhere.
set -e

HOST=${EV3_HOST:-ev3dev}
DEST=${EV3_DEST:-/home/robot/ponderer}

cd "$(dirname "$0")"

mkdir -p logs/brick
ssh "$HOST" "mkdir -p $DEST/logs/brick"

# No --delete: the brick is a source of new runs, not the owner of the archive.
rsync -rlptv "$HOST:$DEST/logs/brick/" logs/brick/
