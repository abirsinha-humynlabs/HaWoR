#!/bin/bash
# Mount the EC2 instance-store NVMe at /opt/dlami/nvme for HaWoR scratch.
# Instance store is wiped on every stop/start, so on boot it comes back blank: format it ONLY
# when it carries no filesystem signature at all, never otherwise. Matched by device model, not
# by nvmeXn1 name, because NVMe enumeration order is not stable across boots.
set -euo pipefail
MNT=/opt/dlami/nvme
if ! mountpoint -q "$MNT"; then
  DEV=""
  for b in /sys/block/nvme*n1; do
    [ "$(tr -d ' ' < "$b/device/model")" = "AmazonEC2NVMeInstanceStorage" ] && { DEV=/dev/$(basename "$b"); break; }
  done
  [ -n "$DEV" ] || { echo "no instance-store NVMe found" >&2; exit 0; }
  if ! blkid -p "$DEV" >/dev/null 2>&1; then mkfs.xfs -q "$DEV"; fi
  mkdir -p "$MNT"
  mount -o noatime "$DEV" "$MNT"
fi
mkdir -p "$MNT"/cache/{conda_pkgs,pip} "$MNT"/hawor_runs/{clips,work,scratch,render} "$MNT"/logs "$MNT"/dl
chown -R ec2-user:ec2-user "$MNT"
