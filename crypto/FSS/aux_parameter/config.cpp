#include "config.h"

namespace FSSConfig {
    int bitlength = 64;
    int num_threads = 4;
    int party = 0;
    Peer *client = nullptr;
    Peer *server = nullptr;
    Peer *peer = nullptr; // 在线通信
    Dealer *dealer = nullptr;
    int port = 42069; // 默认端口
    bool stochasticRT = false;
    bool stochasticT  = false;
}
