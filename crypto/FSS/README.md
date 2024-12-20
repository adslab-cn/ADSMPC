## Introduction
函数秘密分享（Function Secret Share）的代码实现

## Architecture

- **FSS**

- primitives
    密码原语部分，包括DCF，更多FSS原语待添加，如DPF, DIF等
- protocol
    方法协议部分，以FSS为基础实现的协议
- aux_parameter
    辅助操作，包括密钥定义等
- api
    协议评估以及与neural_networks之间的接口，方便网络对FSS后端进行调用（未来看情况可能会作为公共操作放在外面）

## The process of adding protocols

添加协议的完整过程

1. 在 `include/aux_parameter/keypack.h` 设计对应协议需要的密钥

2. 在 `include/protocol`中定义新协议，在 `src/protocol`中添加新协议细节实现

    包括密钥生成和密钥评估两个函数

3. 在 `include/api/api.h`中定义新协议api，在 `src/api/api.cpp` 中添加新协议api的细节实现

    api是为了连接底层FSS和上层神经网络，并实现操作的评估