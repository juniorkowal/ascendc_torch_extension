# Ascend operator pytorch extension

Pytorch extension/package with operator for torch_npu written in AscendC.

## Usage

Common environment setup

```bash
sudo docker pull quay.io/ascend/cann:8.2.rc1.alpha003-910-ubuntu22.04-py3.11

sudo docker run --rm -it --ipc=host --privileged \
    --device=/dev/davinci0 --device=/dev/davinci1 \
    --device=/dev/davinci2 --device=/dev/davinci3 \
    --device=/dev/davinci_manager \
    --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc  \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
    -v $(pwd):/workdir \
    -w /workdir \
    --name custom_ops \
    quay.io/ascend/cann:8.2.rc1.alpha003-910-ubuntu22.04-py3.11 \
    /bin/bash

source /usr/local/Ascend/ascend-toolkit/set_env.sh
# export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH

apt update && apt install -y gcc g++
pip install --upgrade pip setuptools wheel
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
pip install torch_npu==2.6.0
pip install pytest
```

