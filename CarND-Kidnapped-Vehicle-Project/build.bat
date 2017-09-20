echo off
REM !/bin/bash
REM  Script to build all components from scratch, using the maximum available CPU power
REM 
REM  Given parameters are passed over to CMake.
REM  Examples:
REM     * ./build_all.sh -DCMAKE_BUILD_TYPE=Debug
REM     * ./build_all.sh VERBOSE=1
REM 
REM  Written by Tiffany Huang, 12/14/2016
REM 

REM  Compile code.
if not exist "build" mkdir build
cd build
cmake .. -G "Unix Makefiles" && make
if exist "particle_filter.exe" move "particle_filter.exe" ..
cd ..
