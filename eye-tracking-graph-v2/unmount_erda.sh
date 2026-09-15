#!/bin/bash
set -euo pipefail

mnt="./erda2/"

echo "Unmounting ERDA if mounted..."

# Try clean unmount
if mountpoint -q "$mnt"; then
  chmod +rx $HOME
  fusermount -uz "$mnt" || umount -l "$mnt" || true
  sleep 2
  chmod -rx $HOME
  chmod u+rx $HOME
fi

# Double-check: only remove if not mounted
if mountpoint -q "$mnt"; then
  echo "⚠️ Still mounted — skipping rm for safety."
else
  echo "✅ Not mounted. Removing directory."
  rm -rf "$mnt"
fi
