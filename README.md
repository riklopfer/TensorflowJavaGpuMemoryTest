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

