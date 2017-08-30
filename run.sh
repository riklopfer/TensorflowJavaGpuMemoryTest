#!/usr/bin/env bash
java -Xmx128m -cp libtensorflow.jar:./out/ -Djava.library.path=./jni/ test.MemoryTest "$@"
