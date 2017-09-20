echo off
REM !/bin/bash
REM  Script to clean the tree from all compiled files.
REM  You can rebuild them afterwards using "build.sh".
REM 
REM  Written by Tiffany Huang, 12/14/2016
REM 

REM  Remove the dedicated output directories
if exist "build" rmdir /s /q build

REM  We're done!
echo Cleaned up the project!
