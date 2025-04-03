#!/usr/bin/env bash

set -e
cd $(dirname "$0")

echo "Building mupen64plus-core"
cd mupen64plus-core/projects/unix
make all
mv libmupen64plus.* ../../../
make clean
cd ../../../

echo "Building mupen64plus-audio-sdl"
cd mupen64plus-audio-sdl/projects/unix
make all
mv mupen64plus-audio-sdl.* ../../../
make clean
cd ../../../

echo "Building mupen64plus-rsp-hle"
cd mupen64plus-rsp-hle/projects/unix
make all
mv mupen64plus-rsp-hle.* ../../../
make clean
cd ../../../

echo "Building mupen64plus-input-ext"
cd mupen64plus-input-ext/projects/unix
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
