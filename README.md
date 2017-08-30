# TensorflowJavaGpuMemoryTest

Borrowed heavily from https://github.com/jpangburn/tensorflowmemorytest

## Download Dependencies

|ENV Variable|Purpose|Default|
|---|---|---|
|TF_REV|Revision to pull|`1.3.0`|
|TF_TYPE|`cpu` or `gpu`|`gpu`|

GPU

```shell
export TF_REV=1.3.0
export TF_TYPE=gpu
./resolve.sh
```

CPU

```shell
export TF_REV=1.3.0
export TF_TYPE=cpu
./resolve.sh
```


## Build and Run

```shell
./build.sh

export CUDA_VISIBLE_DEVICES=0
./run.sh
```

Open system monitor and watch the memory.

## [TF 11948](https://github.com/tensorflow/tensorflow/issues/11948)

To reproduce this issue, run the following:

```shell
# resolve 
export TF_REV=1.3.0
export TF_TYPE=gpu
./resolve.sh

# build
./build.sh

# run
export CUDA_VISIBLE_DEVICES=0
./run.sh
```

### Sample Log

```
Processing 1 floats.
2017-08-29 14:30:27.963729: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
2017-08-29 14:30:27.963779: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2017-08-29 14:30:27.963788: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
2017-08-29 14:30:27.963795: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
2017-08-29 14:30:27.963802: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use FMA instructions, but these are available on your machine and could speed up CPU computations.
2017-08-29 14:30:29.569904: I tensorflow/core/common_runtime/gpu/gpu_device.cc:955] Found device 0 with properties: 
name: GeForce GTX 1080
major: 6 minor: 1 memoryClockRate (GHz) 1.7335
pciBusID 0000:01:00.0
Total memory: 7.92GiB
Free memory: 7.81GiB
2017-08-29 14:30:29.569957: I tensorflow/core/common_runtime/gpu/gpu_device.cc:976] DMA: 0 
2017-08-29 14:30:29.569965: I tensorflow/core/common_runtime/gpu/gpu_device.cc:986] 0:   Y 
2017-08-29 14:30:29.569981: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1045] Creating TensorFlow device (/gpu:0) -> (device: 0, name: GeForce GTX 1080, pci bus id: 0000:01:00.0)
```

### Valgrind output

[valgrind output](valgrind.out) was generated with the following command:

```shell
valgrind --leak-check=yes java -Djava.compiler=NONE -cp libtensorflow.jar:./out/ -Djava.library.path=./jni/ test.MemoryTest 1000 2>&1 | tee valgrind.out

```