#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -d upstream/.git ]; then
 git clone https://github.com/monatis/ggmlc.git upstream
 git -C upstream checkout 680dd84584dd1cb4de670b1d5b50450f39c805ff
fi
cmake -S upstream -B reference-build -DCMAKE_BUILD_TYPE=Release
cmake --build reference-build --target laya -j2
g++ -O2 -std=c++17 validation/reference.cpp \
 -Iupstream/examples/laya/include -Iupstream/runtime/include -Iupstream/third_party/ggml/include \
 reference-build/examples/laya/CMakeFiles/laya.dir/src/{questions,sequence,recipe,presets,language}.cpp.o \
 reference-build/runtime/libggmlc_runtime.a reference-build/libggml_lib.a \
 -lpthread -ldl -o reference-build/reference
