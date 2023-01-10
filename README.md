# ror

A rust ONNX runtime!


## Install onnxruntime library

You have to install onnxruntime library on your system.

### Linux-x64
```
wget https://github.com/microsoft/onnxruntime/releases/download/v1.13.1/onnxruntime-linux-x64-1.13.1.tgz -O /tmp/onnx.tgz
mkdir -p /tmp/onnx
tar xzf /tmp/onnx.tgz -C /tmp/onnx
export ONNX_DIR=/tmp/onnx/onnxruntime-linux-x64-1.13.1
export LD_LIBRARY_PATH=/tmp/onnx/onnxruntime-linux-x64-1.13.1/lib:$LD_LIBRARY_PATH
```

### OSX-arm64
```
wget https://github.com/microsoft/onnxruntime/releases/download/v1.13.1/onnxruntime-osx-arm64-1.13.1.tgz -O /tmp/onnx.tgz
mkdir -p /tmp/onnx
tar xzf /tmp/onnx.tgz -C /tmp/onnx
export ONNX_DIR=/tmp/onnx/onnxruntime-osx-arm64-1.13.1
export DYLD_LIBRARY_PATH=/tmp/onnx/onnxruntime-osx-arm64-1.13.1/lib:$DYLD_LIBRARY_PATH
```

## Run tests
```
cargo test
```
