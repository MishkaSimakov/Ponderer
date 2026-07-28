#!/bin/sh
# Copy the brick-side code to ev3dev. Run from anywhere.
set -e

HOST=${EV3_HOST:-ev3dev}
DEST=${EV3_DEST:-/home/robot/ponderer}

cd "$(dirname "$0")"

# No -z: the ssh config disables compression, the brick cpu is the bottleneck.
# No -a: owner/group/devices cannot be preserved as the robot user.
rsync -rlptv --delete \
    --exclude='.venv/' \
    --exclude='.idea/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='*.csv' \
    --exclude='*.log' \
    server shared experiments \
    "$HOST:$DEST/"
