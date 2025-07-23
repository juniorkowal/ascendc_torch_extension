#!/bin/bash

# `bisheng` CLI: https://www.hiascend.com/document/detail/zh/canncommercial/800/developmentguide/opdevg/BishengCompiler/atlas_bisheng_10_0003.html
# Adapted from: https://gitee.com/ascend/mstt/tree/master/sample/pytorch_adapter

folder_path="src/gather_custom/lib"

if [ -d "$folder_path" ]; then
    if [ "$(ls -A "$folder_path")" ]; then
        echo "Deleting old lib ..."
        rm -rf "$folder_path"/*
    fi
else
    echo "Creating lib folder ..."
    mkdir -p "$folder_path"
fi

bisheng -fPIC -shared -xcce -O2 -std=c++17 \
    --cce-soc-version=Ascend910B3 --cce-soc-core-type=VecCore \
    -I${ASCEND_TOOLKIT_HOME}/compiler/tikcpp/tikcfw \
    -I${ASCEND_TOOLKIT_HOME}/compiler/tikcpp/tikcfw/impl \
    -I${ASCEND_TOOLKIT_HOME}/compiler/tikcpp/tikcfw/interface \
    -I${ASCEND_TOOLKIT_HOME}/include \
    -o src/gather_custom/lib/libgather_custom_ascendc.so src/gather_custom/gather_custom.cpp
