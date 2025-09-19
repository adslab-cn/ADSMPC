## Introduction
本项目是ADSLab整理和构建的MPC训练和推理框架

## Architecture
This repository has the following components:
- **crypto**
计算库核心部分
    - FSS
    函数秘密分享（Function Secret Share）的代码实现
        - primitives
        密码原语部分，包括DCF，更多FSS原语待添加，如DPF, DIF等
        - protocol
        方法协议部分，以FSS为基础实现的协议
        - aux_parameter
        辅助操作，包括密钥定义等
        - api
        协议评估以及与neural_networks之间的接口，方便网络对FSS后端进行调用（未来看情况可能会作为公共操作放在外面）
    - ASS
    加性秘密分享，待添加

- **neural_networks**
上层神经网络的训练和推理结构，调用底层


- **dataset**
用于存放隐私保护神经网络推理的明文模型结构代码，以及应用所需的相关数据也可放在该文件夹下


- **tests**
本项目的测试代码


## SetUp
项目本体使用以下指令即可安装

可选参数: quick: 直接使用默认安装

```bash
sudo ./1-base.sh quick
```

# Running Tests & Networks

**编译**
```bash
mkdir build && cd build
cmake ..
make
```

编译完成后，将dataset文件夹中需要的数据集复制到创建的build文件夹中

**运行**

如果在本机测试，即tests下代码中ip地址设置为127.0.0.1，使用两个terminal模拟两台服务器

Server：
```bash
./CNN 1
```

Client：
```bash
./CNN 2
```

**Reference:** 

[EzPC](https://github.com/mpc-msri/EzPC/)  
[NssMPClib](https://github.com/XidianNSS/NssMPClib)  
