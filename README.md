# TensorflowJavaGpuMemoryTest

Borrowed heavily from https://github.com/jpangburn/tensorflowmemorytest

## Build

|ENV Variable|Purpose|Default|
|---|---|---|
|TF_REV|Revision to pull|`1.3.0-rc1`|
|TF_TYPE|`cpu` or `gpu`|`gpu`|

Run on the GPU

```shell
export TF_REV=1.3.0-rc1
export TF_TYPE=gpu
bash ./build.sh
```

Run on the CPU

```shell
export TF_REV=1.3.0-rc1
export TF_TYPE=cpu
bash ./build.sh
```


## Run

```shell
bash ./run.sh
```

Open system monitor and watch the memory.

