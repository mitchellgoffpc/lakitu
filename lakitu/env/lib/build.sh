#!/usr/bin/env bash

set -e
cd $(dirname "$0")

echo "Building mupen64plus-core"
cd mupen64plus-core/projects/unix
make clean

if [[ "$OSTYPE" == "darwin"* ]]; then
    sed -i.bak 's/^  LDLIBS += -lopcodes -lbfd/  # LDLIBS += -lopcodes -lbfd/' Makefile  # libopcodes and libbfd aren't needed on macOS
    make all DEBUGGER=1
    mv Makefile.bak Makefile
    mv libmupen64plus.dylib ../../../libmupen64plus.dylib
else
    make all DEBUGGER=1
    mv libmupen64plus.so.2.0.0 ../../../libmupen64plus.so
fi

cp ../../data/font.ttf ../../../
make clean
cd ../../../

echo "Building mupen64plus-audio-sdl"
cd mupen64plus-audio-sdl/projects/unix
make clean
make all
mv mupen64plus-audio-sdl.* ../../../
make clean
cd ../../../

echo "Building mupen64plus-rsp-hle"
cd mupen64plus-rsp-hle/projects/unix
make clean
make all
mv mupen64plus-rsp-hle.* ../../../
make clean
cd ../../../

echo "Building mupen64plus-input-ext"
cd mupen64plus-input-ext/projects/unix
make clean
make all
mv mupen64plus-input-ext.* ../../../
make clean
cd ../../../

echo "Building mupen64plus-video-GLideN64"
cd GLideN64
rm -rf build
mkdir build
cd build
cmake -DMUPENPLUSAPI=On ../src/
make all
mv plugin/Release/mupen64plus-video-GLideN64.* ../../
cd ../
rm -rf build

echo "Success!"
