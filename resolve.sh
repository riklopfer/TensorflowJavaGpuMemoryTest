#!/usr/bin/env bash
if ! [ $2 ]; then
	echo "USAGE:   resolve.sh TF_REV TF_TYPE"
	echo "TF_REV  -> tensorflow revision"
	echo "TF_TYPE -> gpu or cpu"
	echo 
	echo "example: "
	echo "         resolve.sh 1.3.0 gpu"
	exit
fi

TF_REV=$1
TF_TYPE=$2

# tensorflow revision
if ! [ $TF_REV ]; then
    TF_REV=1.3.0
    echo "Defaulting to TF_REV=${TF_REV}"
fi

# native type
if ! [ $TF_TYPE ]; then
    TF_TYPE="gpu"
    echo "Defaulting to TF_TYPE=${TF_TYPE}"
fi

# OS
OS=$(uname -s | tr '[:upper:]' '[:lower:]')

JNI_URL="https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-${TF_TYPE}-${OS}-x86_64-${TF_REV}.tar.gz"
JAR_URL="https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-${TF_REV}.jar"

# clean
rm -rvf jni/
rm -vf libtensorflow.jar
mkdir ./jni

echo "Downloading: ${JNI_URL}"
curl -L ${JNI_URL} |
tar --warning=no-timestamp -xz -C ./jni || exit

echo "Downloading: ${JAR_URL}"
curl -o libtensorflow.jar ${JAR_URL} 
