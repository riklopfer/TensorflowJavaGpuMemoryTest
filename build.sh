#!/usr/bin/env bash

# build
[ -d out ] || mkdir out
javac -cp libtensorflow.jar -d out/ test/MemoryTest.java 
