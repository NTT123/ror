Install libraries

```bash
sudo apt install llvm-dev libclang-dev clang -y
```

Download onnxruntime

```
wget https://github.com/microsoft/onnxruntime/releases/download/v1.13.1/onnxruntime-linux-x64-1.13.1.tgz
tar xzf onnxruntime-linux-x64-1.13.1.tgz
```

```bash
export LIBCLANG_PATH=/usr/local/lib
```
