/*    全局配置    */
#pragma once

class Peer; // 参与方
class Dealer; // 密钥发送方
namespace FSSConfig {

    extern int bitlength; // 加密算法中使用的比特长度
    extern int num_threads; // 并行计算线程数
    // 参与方角色
    extern int party;
    extern Peer *client;
    extern Peer *server;
    extern Peer *peer;
    extern Dealer *dealer;
    extern int port; // 网络通信端口号
    // 启用随机化技术
    extern bool stochasticRT;
    extern bool stochasticT;
}
