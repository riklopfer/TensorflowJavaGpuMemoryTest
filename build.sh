#!/usr/bin/env bash

# tensorflow revision
if ! [ $TF_REV ]; then
    TF_REV=1.3.0-rc1
    echo "Defaulting to TF_REV=${TF_REV}"
fi

# native type
if ! [ $TF_TYPE ]; then
    TF_TYPE="gpu"
    echo "Defaulting to TF_TYPE=${TF_TYPE}"
fi

# OS
OS=$(uname -s | tr '[:upper:]' '[:lower:]')

# clean
rm -rvf jni/
rm -vf libtensorflow.jar

mkdir -p ./jni
curl -L "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-${TF_TYPE}-${OS}-x86_64-${TF_REV}.tar.gz" |
tar --warning=no-timestamp -xz -C ./jni

# Java jar
curl -o libtensorflow.jar https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-${TF_REV}.jar

# build
[ -d out ] || mkdir out
javac -cp libtensorflow.jar -d out/ test/MemoryTest.java 
