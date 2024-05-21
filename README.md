# DepthMamba

## 目录

- [安装](#安装)
- [使用说明](#使用说明)


## 安装

请按照以下步骤安装和配置项目：

1. 克隆仓库：
    ```bash
    git clone https://github.com/swagger-coder/DepthMamba.git
    cd DepthMamba
    ```

2. 安装依赖：
    
- Python 3.10.13
  ```bash
  conda create -n DepthMamba python=3.10.13
  ```

- torch 2.0.1 
  ```bash
  # CUDA 11.7
  pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
  ```

- 安装其他Requirements: requirements.txt
  ```bash
  pip install -r requirements.txt
  ```

- 安装 ``causal_conv1d`` 和 ``mamba``
  ```bash
  pip install -e causal_conv1d>=1.1.0
  pip install -e mamba-1p1p1
  ```
  
  

## 使用说明

训练
```bash
bash pretrain_deptha_0516.sh
```


