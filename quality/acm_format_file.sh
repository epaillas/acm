#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <file>"
    exit 1
fi

if [ ! -f "$1" ]; then
    echo "Error : '$1' isn't a file."
    exit 1
fi

ruff check --select I --fix $1
ruff format --config $ACM_root/quality/ruff.toml  $1


