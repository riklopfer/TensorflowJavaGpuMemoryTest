#!/usr/bin/env bash
java -cp libtensorflow.jar:./out/ -Djava.library.path=./jni/ test.MemoryTest
