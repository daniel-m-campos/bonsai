#!/bin/bash
# RunPod pod entrypoint: PUBLIC_KEY -> authorized_keys, then foreground sshd.
set -e
mkdir -p /root/.ssh /run/sshd
chmod 700 /root/.ssh
if [ -n "$PUBLIC_KEY" ]; then
    echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
fi
ssh-keygen -A
echo "bonsai-ci pod ready"
exec /usr/sbin/sshd -D
