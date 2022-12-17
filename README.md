# ror

A rust ONNX runtime!


## Install onnxruntime library

You have to install onnxruntime library on your system:

```
wget https://github.com/microsoft/onnxruntime/releases/download/v1.13.1/onnxruntime-linux-x64-1.13.1.tgz -O /tmp/onnx.tgz
mkdir -p /opt
tar xzf /tmp/onnx.tgz -C /opt
```

Then, set `LD_LIBRARY_PATH` to include onnxruntime:

```
export LD_LIBRARY_PATH=/opt/onnxruntime-linux-x64-1.13.1/lib:$LD_LIBRARY_PATH
```

## Run tests

To run test, you also need to download the example onnx file:

```
wget https://raw.githubusercontent.com/microsoft/Windows-Machine-Learning/master/Samples/FNSCandyStyleTransfer/UWP/cs/Assets/candy.onnx
```


```
cargo test
```
