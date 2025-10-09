#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! source "$SCRIPT_DIR/oneapi-tbb-2022.2.0/env/vars.sh"; then
    echo "vars.sh 执行失败，开始解压 oneapi-tbb-2022.2.0-lin.tgz..."
    tar zxvf "$SCRIPT_DIR/oneapi-tbb-2022.2.0-lin.tgz" -C "$SCRIPT_DIR"
    echo "重新执行 vars.sh..."
    source "$SCRIPT_DIR/oneapi-tbb-2022.2.0/env/vars.sh"
fi

