#!/bin/bash
key=/home/jojuo/.ssh/id_ed25519
user=tran.duy.vu@di.ku.dk
erdadir=' '
mnt=erda2/
if [ -f "$key" ]
then
    mkdir -p ${mnt}
    sshfs ${user}@io.erda.dk:${erdadir} ${mnt} -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3 -o IdentityFile=${key}
else
    echo "'${key}' is not an ssh key"
fi
