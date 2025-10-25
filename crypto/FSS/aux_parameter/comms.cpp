// MPC框架的通信层和密钥管理模块
// 用于接收和发送对应密钥

#include "comms.h"
#include "assert.h"
#include <bitpack/bitpack.h> 
#include "stats.h"
#include <chrono>

using namespace FSSConfig;

void print_socket_info(const std::string& name, int sockfd) {
    if (sockfd < 0) {
        std::cerr << "  Socket '" << name << "': Invalid" << std::endl;
        return;
    }

    struct sockaddr_in local_addr, peer_addr;
    socklen_t addr_len = sizeof(struct sockaddr_in);

    // 获取本地地址
    if (getsockname(sockfd, (struct sockaddr*)&local_addr, &addr_len) == 0) {
        char local_ip[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &local_addr.sin_addr, local_ip, sizeof(local_ip));
        int local_port = ntohs(local_addr.sin_port);
        std::cerr << "  Socket '" << name << "' (fd=" << sockfd << "):" << std::endl;
        std::cerr << "    Local Address : " << local_ip << ":" << local_port << std::endl;
    }

    // 获取远端地址
    if (getpeername(sockfd, (struct sockaddr*)&peer_addr, &addr_len) == 0) {
        char peer_ip[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &peer_addr.sin_addr, peer_ip, sizeof(peer_ip));
        int peer_port = ntohs(peer_addr.sin_port);
        std::cerr << "    Peer Address  : " << peer_ip << ":" << peer_port << std::endl;
    }
}
SocketBuf::SocketBuf(std::string ip, int port, bool onlyRecv = false)
{
    this->t = BUF_SOCKET;
    std::cerr << "[CLIENT] Trying to connect to server at " << ip << ":" << port << " and " << port + 3 << "..." << std::endl;
    std::cerr << "trying to connect with server...";
    {
        struct sockaddr_in addr;
        addr.sin_family = AF_INET; // IPv4地址族 
        addr.sin_port = htons(port); // 端口号转换为网络字节序
        addr.sin_addr.s_addr = inet_addr(ip.c_str()); // 将IP字符串转换为二进制格式
        std::cerr << "[CLIENT] Attempting to connect recv socket to " << ip << ":" << port << "..." << std::endl;
        while (1)// 无限循环直到连接成功
        {
            recvsocket = socket(AF_INET, SOCK_STREAM, 0);
            if (recvsocket < 0)
            {
                perror("socket");
                exit(1);
            }
            if (connect(recvsocket, (struct sockaddr *)&addr, sizeof(addr)) == 0)
            {
                std::cerr << "[CLIENT] Recv socket connected to " << ip << ":" << port << "." << std::endl;
                break;
            }
            ::close(recvsocket);
            usleep(1000);
        }
        // 设置TCP_NODELAY禁用Nagle算法，减少延迟
        const int one = 1;
        setsockopt(recvsocket, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    }
    sleep(1);
    if (!onlyRecv)
    {
        struct sockaddr_in addr;
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port + 3); // 发送端口为接收端口+3
        addr.sin_addr.s_addr = inet_addr(ip.c_str());
        std::cerr << "[CLIENT] Attempting to connect send socket to " << ip << ":" << port + 3 << "..." << std::endl;
        while (1)
        {
            sendsocket = socket(AF_INET, SOCK_STREAM, 0);
            if (sendsocket < 0)
            {
                perror("socket");
                exit(1);
            }
            if (connect(sendsocket, (struct sockaddr *)&addr, sizeof(addr)) == 0)
            {
                std::cerr << "[CLIENT] Send socket connected to " << ip << ":" << port + 3 << "." << std::endl;
                break;
            }
            ::close(sendsocket);
            usleep(1000);
        }
        const int one = 1;
        setsockopt(sendsocket, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    }
    std::cerr << "[CLIENT] Both sockets connected. Peer established." << std::endl; // 修改原来的 "connected"
}




// 进行简单的握手验证
void SocketBuf::sync()
{
    print_socket_info("sendsocket", this->sendsocket);
    print_socket_info("recvsocket", this->recvsocket);
    char buf[1] = {1}; 
    send(sendsocket, buf, 1, 0); 
    recv(recvsocket, buf, 1, MSG_WAITALL);
    bytesReceived += 1; 
    bytesSent += 1; 
    always_assert(buf[0] == 1); 
}

void SocketBuf::read(char *buf, int bytes)
{
    always_assert(bytes == recv(recvsocket, (char *)buf, bytes, MSG_WAITALL));
    bytesReceived += bytes;
}

// 数据读取方法
char *SocketBuf::read(int bytes)
{
    char *tmpBuf = new char[bytes]; // 动态分配缓冲区
    always_assert(bytes == recv(recvsocket, (char *)tmpBuf, bytes, MSG_WAITALL));
    bytesReceived += bytes; // 累加接收字节数
    return tmpBuf; // 返回缓冲区指针（需调用者释放）
}

// 动态分配缓冲区的读取方法
void SocketBuf::write(char *buf, int bytes)
{
    always_assert(bytes == send(sendsocket, buf, bytes, 0)); // 发送数据并验证发送字节数
    bytesSent += bytes; // 累加发送字节数
}

// 资源释放方法
void SocketBuf::close()
{
    ::close(sendsocket);
    ::close(recvsocket);
}

void Peer::close()
{
    keyBuf->close();
}

/* 服务器端等待客户端连接 */ 
Peer* waitForPeer(int port)
{
    int sendsocket, recvsocket;
    std::cerr << "[SERVER] Starting to wait for client on port " << port << " and " << port + 3 << "..." << std::endl;
    std::cerr << "waiting for connection from client...";
    {
        struct sockaddr_in dest;
        struct sockaddr_in serv;
        socklen_t socksize = sizeof(struct sockaddr_in);
        memset(&serv, 0, sizeof(serv));
        serv.sin_family = AF_INET;
        serv.sin_addr.s_addr = htonl(INADDR_ANY); /* set our address to any interface */
        serv.sin_port = htons(port);              /* set the server port number */
        int mysocket = socket(AF_INET, SOCK_STREAM, 0);
        int reuse = 1;
        setsockopt(mysocket, SOL_SOCKET, SO_REUSEADDR, (const char *)&reuse,
                   sizeof(reuse));
        if (::bind(mysocket, (struct sockaddr *)&serv, sizeof(struct sockaddr)) < 0)
        {
            perror("error: bind");
            exit(1);
        }
        if (listen(mysocket, 1) < 0)
        {
            perror("error: listen");
            exit(1);
        }
        std::cerr << "[SERVER] Listening on port " << port << " for send socket..." << std::endl;
        sendsocket = accept(mysocket, (struct sockaddr *)&dest, &socksize);
        std::cerr << "[SERVER] Accepted connection on port " << port << " for send socket." << std::endl;
        const int one = 1;
        setsockopt(sendsocket, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        close(mysocket);
    }

    {
        struct sockaddr_in dest;
        struct sockaddr_in serv;
        socklen_t socksize = sizeof(struct sockaddr_in);
        memset(&serv, 0, sizeof(serv));
        serv.sin_family = AF_INET;
        serv.sin_addr.s_addr = htonl(INADDR_ANY); /* set our address to any interface */
        // 双端口设计：发送/接收分离，避免单端口双向通信的竞争问题。
        serv.sin_port = htons(port + 3);          /* set the server port number */
        int mysocket = socket(AF_INET, SOCK_STREAM, 0);
        int reuse = 1;
        setsockopt(mysocket, SOL_SOCKET, SO_REUSEADDR, (const char *)&reuse,
                   sizeof(reuse));
        if (::bind(mysocket, (struct sockaddr *)&serv, sizeof(struct sockaddr)) < 0)
        {
            perror("error: bind");
            exit(1);
        }
        if (listen(mysocket, 1) < 0)
        {
            perror("error: listen");
            exit(1);
        }
        std::cerr << "[SERVER] Listening on port " << port + 3 << " for recv socket..." << std::endl;
        recvsocket = accept(mysocket, (struct sockaddr *)&dest, &socksize);
        std::cerr << "[SERVER] Accepted connection on port " << port + 3 << " for recv socket." << std::endl;
        const int one = 1;
        setsockopt(recvsocket, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        close(mysocket);
    }

    std::cerr << "[SERVER] Both sockets connected. Peer established." ;
    return new Peer(sendsocket, recvsocket);
}

void Peer::send_ge(const GroupElement &g, int bw)
{
    if (bw > 32)
    {
        char *buf = (char *)(&g);
        this->keyBuf->write(buf, 8);
    }
    else if (bw > 16)
    {
        char *buf = (char *)(&g);
        this->keyBuf->write(buf, 4);
    }
    else if (bw > 8)
    {
        char *buf = (char *)(&g);
        this->keyBuf->write(buf, 2);
    }
    else
    {
        char *buf = (char *)(&g);
        this->keyBuf->write(buf, 1);
    }
}

void Peer::send_ge_array(const GroupElement *g, int size)
{
    char *buf = (char *)(g);
    this->keyBuf->write(buf, 8 * size);
}

void Peer::send_block(const osuCrypto::block &b)
{
    char *buf = (char *)(&b);
    this->keyBuf->write(buf, sizeof(osuCrypto::block));
}

void Peer::send_mask(const GroupElement &g)
{
    send_ge(g, 64);
}

void Peer::send_input(const GroupElement &g)
{
    send_ge(g, 64);
}

void Peer::send_batched_input(GroupElement *g, int size, int bw)
{
    if (bw > 32)
    {
        uint64_t *temp = new uint64_t[size];
        for (int i = 0; i < size; i++)
        {
            temp[i] = g[i];
        }
        char *buf = (char *)(temp);
        this->keyBuf->write(buf, 8 * size);
        delete[] temp;
    }
    else if (bw > 16)
    {
        uint32_t *temp = new uint32_t[size];
        for (int i = 0; i < size; i++)
        {
            temp[i] = (uint32_t)g[i];
        }
        char *buf = (char *)(temp);
        this->keyBuf->write(buf, 4 * size);
        delete[] temp;
    }
    else if (bw > 8)
    {
        uint16_t *temp = new uint16_t[size];
        for (int i = 0; i < size; i++)
        {
            temp[i] = (uint16_t)g[i];
        }
        char *buf = (char *)(temp);
        this->keyBuf->write(buf, 2 * size);
        delete[] temp;
    }
    else
    {
        uint8_t *temp = new uint8_t[size];
        for (int i = 0; i < size; i++)
        {
            temp[i] = (uint8_t)g[i];
        }
        char *buf = (char *)(temp);
        this->keyBuf->write(buf, size);
        delete[] temp;
    }
}

void Peer::recv_batched_input(uint64_t *g, int size, int bw)
{
    if (bw > 32)
    {
        this->keyBuf->read((char *)g, 8 * size);
    }
    else if (bw > 16)
    {
        uint32_t *tmp = new uint32_t[size];
        this->keyBuf->read((char *)tmp, 4 * size);
        for (int i = 0; i < size; i++)
        {
            g[i] = tmp[i];
        }
        delete[] tmp;
    }
    else if (bw > 8)
    {
        uint16_t *tmp = new uint16_t[size];
        this->keyBuf->read((char *)tmp, 2 * size);
        for (int i = 0; i < size; i++)
        {
            g[i] = tmp[i];
        }
        delete[] tmp;
    }
    else
    {
        uint8_t *tmp = new uint8_t[size];
        this->keyBuf->read((char *)tmp, size);
        for (int i = 0; i < size; i++)
        {
            g[i] = tmp[i];
        }
        delete[] tmp;
    }
}

void Peer::send_mult_key(const MultKey &k)
{
    char *buf = (char *)(&k);
    this->keyBuf->write(buf, sizeof(MultKey));
}

void Peer::send_square_key(const SquareKey &k)
{
    send_ge(k.b, 64);
    send_ge(k.c, 64);
}

void Peer::send_matmul_key(const MatMulKey &k)
{
    int s1 = k.s1;
    int s2 = k.s2;
    int s3 = k.s3;

    for (int i = 0; i < s1; i++)
    {
        for (int j = 0; j < s2; j++)
        {
            send_ge(Arr2DIdx(k.a, s1, s2, i, j), k.Bin);
        }
    }

    for (int i = 0; i < s2; i++)
    {
        for (int j = 0; j < s3; j++)
        {
            send_ge(Arr2DIdx(k.b, s2, s3, i, j), k.Bin);
        }
    }

    for (int i = 0; i < s1; i++)
    {
        for (int j = 0; j < s3; j++)
        {
            send_ge(Arr2DIdx(k.c, s1, s3, i, j), k.Bout);
        }
    }
}

void Peer::send_conv2d_key(const Conv2DKey &k)
{
    int N = k.N;
    int H = k.H;
    int W = k.W;
    int CO = k.CO;
    int CI = k.CI;
    int FH = k.FH;
    int FW = k.FW;
    int zPadHLeft = k.zPadHLeft;
    int zPadHRight = k.zPadHRight;
    int zPadWLeft = k.zPadWLeft;
    int zPadWRight = k.zPadWRight;
    ;
    int strideH = k.strideH;
    int strideW = k.strideW;

    int d0 = N;
    int d1 = ((H - FH + (zPadHLeft + zPadHRight)) / strideH) + 1;
    int d2 = ((W - FW + (zPadWLeft + zPadWRight)) / strideW) + 1;
    int d3 = CO;

    for (int n = 0; n < N; ++n)
    {
        for (int h = 0; h < H; ++h)
        {
            for (int w = 0; w < W; ++w)
            {
                for (int ci = 0; ci < CI; ++ci)
                {
                    send_ge(Arr4DIdx(k.a, N, H, W, CI, n, h, w, ci), k.Bin);
                }
            }
        }
    }

    for (int fh = 0; fh < FH; ++fh)
    {
        for (int fw = 0; fw < FW; ++fw)
        {
            for (int ci = 0; ci < CI; ++ci)
            {
                for (int co = 0; co < CO; ++co)
                {
                    send_ge(Arr4DIdx(k.b, FH, FW, CI, CO, fh, fw, ci, co), k.Bin);
                }
            }
        }
    }

    GroupElement *c = k.c;
    int Bout = k.Bout;
    for (int i = 0; i < d0; ++i)
    {
        for (int j = 0; j < d1; ++j)
        {
            for (int k = 0; k < d2; ++k)
            {
                for (int l = 0; l < d3; ++l)
                {
                    send_ge(Arr4DIdx(c, d0, d1, d2, d3, i, j, k, l), Bout);
                }
            }
        }
    }
}

void Peer::send_conv3d_key(const Conv3DKey &k)
{
    int N = k.N;
    int D = k.D;
    int H = k.H;
    int W = k.W;
    int CO = k.CO;
    int CI = k.CI;
    int FD = k.FD;
    int FH = k.FH;
    int FW = k.FW;
    int zPadDLeft = k.zPadDLeft;
    int zPadDRight = k.zPadDRight;
    int zPadHLeft = k.zPadHLeft;
    int zPadHRight = k.zPadHRight;
    int zPadWLeft = k.zPadWLeft;
    int zPadWRight = k.zPadWRight;
    ;
    int strideH = k.strideH;
    int strideW = k.strideW;
    int strideD = k.strideD;

    int d0 = N;
    int d1 = ((D - FD + (zPadDLeft + zPadDRight)) / strideD) + 1;
    int d2 = ((H - FH + (zPadHLeft + zPadHRight)) / strideH) + 1;
    int d3 = ((W - FW + (zPadWLeft + zPadWRight)) / strideW) + 1;
    int d4 = CO;

    for (size_t i = 0; i < N * D * H * W * CI; ++i)
    {
        send_ge(k.a[i], k.Bin);
    }

    for (size_t i = 0; i < FD * FH * FW * CI * CO; ++i)
    {
        send_ge(k.b[i], k.Bin);
    }

    for (size_t i = 0; i < d0 * d1 * d2 * d3 * d4; ++i)
    {
        send_ge(k.c[i], k.Bout);
    }
}

void Peer::send_dcf_keypack(const DCFKeyPack &kp)
{
    for (int i = 0; i < kp.Bin + 1; ++i)
    {
        send_block(kp.k[i]);
    }
    for (int i = 0; i < kp.groupSize; ++i)
    {
        send_ge(kp.g[i], kp.Bout);
    }
    for (int i = 0; i < kp.groupSize * kp.Bin; ++i)
    {
        send_ge(kp.v[i], kp.Bout);
    }
    send_ge(42, 64);
}

void Peer::send_dpf_keypack(const DPFKeyPack &kp)
{
    for (int i = 0; i < kp.bin + 1; ++i)
    {
        send_block(kp.s[i]);
    }
    send_ge(kp.tLcw, kp.bin);
    send_ge(kp.tRcw, kp.bin);
    send_ge(kp.payload, kp.bout);
    send_ge(42, 64);
}

void Peer::send_dpfet_keypack(const DPFETKeyPack &kp)
{
    for (int i = 0; i < kp.bin + 1 - 7; ++i)
    {
        send_block(kp.s[i]);
    }
    send_ge(kp.tLcw, kp.bin);
    send_ge(kp.tRcw, kp.bin);
    send_block(kp.leaf);
    send_ge(42, 64);
}

void Peer::send_ddcf_keypack(const DualDCFKeyPack &kp)
{
    send_dcf_keypack(kp.dcfKey);
    for (int i = 0; i < kp.groupSize; ++i)
    {
        send_ge(kp.sb[i], kp.Bout);
    }
}

void Peer::send_new_mult_key(const MultKeyNew &k, int bw1, int bw2)
{
    send_ge(k.a, bw1 + bw2);
    send_ge(k.b, bw1 + bw2);
    send_ge(k.c, bw1 + bw2);
    send_dcf_keypack(k.k1);
    send_dcf_keypack(k.k2);
    send_dcf_keypack(k.k3);
    send_dcf_keypack(k.k4);
}

void Peer::send_relu_key(const ReluKeyPack &kp)
{
    int Bin = kp.Bin;
    int groupSize = 2;
    for (int i = 0; i < Bin + 1; ++i)
    {
        send_block(kp.k[i]);
    }
    for (int i = 0; i < groupSize; ++i)
    {
        send_ge(kp.g[i], kp.Bout);
    }
    for (int i = 0; i < Bin * groupSize; ++i)
    {
        send_ge(kp.v[i], kp.Bout);
    }
    send_ge(kp.e_b0, kp.Bout);
    send_ge(kp.e_b1, kp.Bout);
    send_ge(kp.beta_b0, kp.Bout);
    send_ge(kp.beta_b1, kp.Bout);
    send_ge(kp.r_b, kp.Bout);
    send_ge(kp.drelu, 1);
}

void Peer::send_maxpool_key(const MaxpoolKeyPack &kp)
{
    send_relu_key(kp.reluKey);
    send_ge(kp.rb, kp.Bout);
}

void Peer::send_maxpool_double_key(const MaxpoolDoubleKeyPack &kp)
{
    send_relu_2round_key(kp.reluKey);
    send_ge(kp.rb, kp.Bout);
}

void Peer::send_scmp_keypack(const ScmpKeyPack &kp)
{
    send_ddcf_keypack(kp.dualDcfKey);
    send_ge(kp.rb, kp.Bout);
}

void Peer::send_pubdiv_key(const PublicDivKeyPack &kp)
{
    send_ddcf_keypack(kp.dualDcfKey);
    send_scmp_keypack(kp.scmpKey);
    send_ge(kp.zb, kp.Bout);
}

void Peer::send_ars_key(const ARSKeyPack &kp)
{
    if (!FSSConfig::stochasticT)
        send_dcf_keypack(kp.dcfKey);
    if (kp.Bout > kp.Bin - kp.shift)
    {
        send_ddcf_keypack(kp.dualDcfKey);
    }
    send_ge(kp.rb, kp.Bout);
}

void Peer::send_spline_key(const SplineKeyPack &kp)
{
    send_dcf_keypack(kp.dcfKey);
    for (auto &pi : kp.p)
    {
        send_ge(pi, kp.Bin);
    }

    for (auto &row : kp.e_b)
    {
        for (auto &e : row)
        {
            send_ge(e, kp.Bout);
        }
    }

    for (auto &b : kp.beta_b)
    {
        send_ge(b, kp.Bout);
    }

    send_ge(kp.r_b, kp.Bout);
}

void Peer::send_signedpubdiv_key(const SignedPublicDivKeyPack &kp)
{
    send_ge(kp.d, kp.Bin);
    send_dcf_keypack(kp.dcfKey);
    send_publicIC_key(kp.publicICkey);
    send_scmp_keypack(kp.scmpKey);
    send_ge(kp.A_share, kp.Bin);
    send_ge(kp.corr_share, kp.Bout);
    send_ge(kp.B_share, kp.Bout);
    send_ge(kp.rdiv_share, kp.Bout);
    send_ge(kp.rout_temp_share, kp.Bout);
    send_ge(kp.rout_share, kp.Bout);
}

void Peer::send_publicIC_key(const PublicICKeyPack &kp)
{
    send_dcf_keypack(kp.dcfKey);
    send_ge(kp.zb, kp.Bout);
}

void Peer::send_relu_truncate_key(const ReluTruncateKeyPack &kp)
{
    send_dcf_keypack(kp.dcfKeyN);
    if (!FSSConfig::stochasticRT)
        send_dcf_keypack(kp.dcfKeyS);
    send_ge(kp.zTruncate, kp.Bin);
    send_ge(kp.a, kp.Bin);
    send_ge(kp.b, kp.Bin);
    send_ge(kp.c, kp.Bin);
    send_ge(kp.d1, kp.Bin);
    send_ge(kp.d2, kp.Bin);
}

void Peer::send_relu_2round_key(const Relu2RoundKeyPack &kp)
{
    send_dcf_keypack(kp.dcfKey);
    send_ge(kp.a, kp.Bin);
    send_ge(kp.b, kp.Bin);
    send_ge(kp.c, kp.Bin);
    send_ge(kp.d1, kp.Bin);
    send_ge(kp.d2, kp.Bin);
}

void Peer::send_select_key(const SelectKeyPack &kp)
{
    send_ge(kp.a, kp.Bin);
    send_ge(kp.b, kp.Bin);
    send_ge(kp.c, kp.Bin);
    send_ge(kp.d1, kp.Bin);
    send_ge(kp.d2, kp.Bin);
}

void Peer::send_bitwise_and_key(const BitwiseAndKeyPack &kp)
{
    send_ge(kp.t[0], 64);
    send_ge(kp.t[1], 64);
    send_ge(kp.t[2], 64);
    send_ge(kp.t[3], 64);
}

void Peer::send_mic_key(const MICKeyPack &kp, int bin, int bout, int m)
{
    send_dcf_keypack(kp.dcfKey);
    for (int i = 0; i < m; ++i)
    {
        send_ge(kp.z[i], bout);
    }
}

void Peer::send_fix_to_float_key(const FixToFloatKeyPack &kp, int bl)
{
    send_mic_key(kp.micKey, bl, bl, 2 * bl);
    send_ge(kp.rs, 1);
    send_ge(kp.rpow, bl);
    send_ge(kp.ry, bl);
    send_select_key(kp.selectKey);
    send_ge(kp.rm, bl);
}

void Peer::send_float_to_fix_key(const FloatToFixKeyPack &kp, int bl)
{
    send_dcf_keypack(kp.dcfKey);
    send_select_key(kp.selectKey);
    send_ge(kp.rm, 24);
    send_ge(kp.re, 10);
    send_ge(kp.rw, 1);
    // send_ge(kp.rt, bl);
    send_ge(kp.rh, bl);
    for (int i = 0; i < 1024; ++i)
    {
        send_ge(kp.p[i], bl);
    }
    for (int i = 0; i < 1024; ++i)
    {
        send_ge(kp.q[i], bl);
    }
    send_ars_key(kp.arsKey);
}

void Peer::send_relu_extend_key(const ReluExtendKeyPack &kp, int bin, int bout)
{
    send_dcf_keypack(kp.dcfKey);
    send_ge(kp.rd, 2);
    send_ge(kp.rw, 2);
    for (int i = 0; i < 4; ++i)
        send_ge(kp.p[i], bout);
    for (int i = 0; i < 2; ++i)
        send_ge(kp.q[i], bout);
}

void Peer::send_sign_extend2_key(const SignExtend2KeyPack &kp, int bin, int bout)
{
    send_dcf_keypack(kp.dcfKey);
    send_ge(kp.rw, 1);
    send_ge(kp.p[0], bout);
    send_ge(kp.p[1], bout);
}

void Peer::send_triple_key(const TripleKeyPack &kp)
{
    for (size_t i = 0; i < kp.na; ++i)
    {
        send_ge(kp.a[i], kp.bw);
    }

    for (size_t i = 0; i < kp.nb; ++i)
    {
        send_ge(kp.b[i], kp.bw);
    }

    for (size_t i = 0; i < kp.nc; ++i)
    {
        send_ge(kp.c[i], kp.bw);
    }
}

void Peer::send_edabits_prtrunc_key(const EdabitsPrTruncKeyPack &kp, int bw)
{
    send_ge(kp.a, bw);
    send_ge(kp.b, bw);
}

void Peer::send_pubcmp_key(const PubCmpKeyPack &kp)
{
    send_dcf_keypack(kp.dcfKey);
    send_ge(kp.rout, 1);
}

void Peer::send_clip_key(const ClipKeyPack &kp)
{
    send_pubcmp_key(kp.cmpKey);
    send_ge(kp.a, kp.bin);
    send_ge(kp.b, kp.bin);
    send_ge(kp.c, kp.bin);
    send_ge(kp.d1, kp.bin);
    send_ge(kp.d2, kp.bin);
}

void Peer::send_lut_key(const LUTKeyPack &kp)
{
    send_dpf_keypack(kp.dpfKey);
    send_ge(kp.rout, kp.bout);
}

void Peer::send_f2bf16_key(const F2BF16KeyPack &kp)
{
    int bin = kp.bin;
    send_dcf_keypack(kp.dcfKey);
    send_dcf_keypack(kp.dcfTruncate);
    send_ge(kp.rout_k, bin);
    send_ge(kp.rout_m, bin);
    send_ge(kp.rin, bin);
    send_ge(kp.prod, bin);
    send_ge(kp.rout, 13);
    send_ge(kp.rProd, bin);
}

void Peer::send_truncate_reduce_key(const TruncateReduceKeyPack &kp)
{
    send_dcf_keypack(kp.dcfKey);
    send_ge(kp.rout, kp.bin - kp.shift);
}

void Peer::send_lutss_key(const LUTSSKeyPack &kp)
{
    send_ge(kp.b0, 64);
    send_ge(kp.b1, 64);
    send_ge(kp.b2, 64);
    send_ge(kp.b3, 64);
    send_ge(kp.routRes, kp.bout);
    send_ge(kp.routCorr, kp.bout);
    send_ge(kp.rout, kp.bout);
}

void Peer::send_lutdpfet_key(const LUTDPFETKeyPack &kp)
{
    send_dpfet_keypack(kp.dpfKey);
    send_ge(kp.routRes, kp.bout);
    send_ge(kp.routCorr, 1);
}

void Peer::send_sloth_drelu_key(const SlothDreluKeyPack &kp)
{
    send_dpfet_keypack(kp.dpfKey);
    send_ge(kp.r, 1);
}

void Peer::send_wrap_dpf_key(const WrapDPFKeyPack &kp)
{
    send_dpfet_keypack(kp.dpfKey);
    send_ge(kp.r, 1);
}

void Peer::send_wrap_ss_key(const WrapSSKeyPack &kp)
{
    send_ge(kp.b0, 64);
    if (kp.bin == 7)
        send_ge(kp.b1, 64);
}

void Peer::send_sloth_lrs_key(const SlothLRSKeyPack &kp)
{
    send_ge(kp.rout, kp.bin);
    send_ge(kp.msb, kp.bin);
    send_ge(kp.select, kp.bin);
}

GroupElement Peer::recv_input()
{
    char buf[8];
    this->keyBuf->read(buf, 8);
    GroupElement g = *(uint64_t *)buf;
    return g;
}

void Peer::send_uint8_array(const uint8_t *data, int size)
{
    this->keyBuf->write((char *)data, size);
    // always_assert(size == send(sendsocket, data, size, 0));
    // bytesSent += size;
}

void Peer::recv_uint8_array(uint8_t *data, int size)
{
    this->keyBuf->read((char *)data, size);
    // always_assert(size == recv(recvsocket, data, size, MSG_WAITALL));
    // bytesReceived += size;
}

Dealer::Dealer(std::string ip, int port)
{
    this->keyBuf = new SocketBuf(ip, port, true);
}

void Dealer::close()
{
    keyBuf->close();
}

GroupElement Dealer::recv_mask()
{
    char buf[8];
    this->keyBuf->read(buf, 8);
    GroupElement g = *(uint64_t *)buf;
    return g;
}

MultKey Dealer::recv_mult_key()
{
    char buf[sizeof(MultKey)];
    this->keyBuf->read(buf, sizeof(MultKey));
    MultKey k(*(MultKey *)buf);
    return k;
}

SquareKey Dealer::recv_square_key()
{
    SquareKey k;
    k.b = recv_ge(64);
    k.c = recv_ge(64);
    return k;
}

osuCrypto::block Dealer::recv_block()
{
    char buf[sizeof(osuCrypto::block)];
    this->keyBuf->read(buf, sizeof(osuCrypto::block));
    osuCrypto::block b = *(osuCrypto::block *)buf;
    return b;
}

osuCrypto::block *Dealer::recv_block_array(int numBlocks)
{
    return (osuCrypto::block *)this->keyBuf->read(numBlocks * sizeof(osuCrypto::block));
}

GroupElement Dealer::recv_ge(int bl)
{
    if (bl > 32)
    {
        char buf[8];
        this->keyBuf->read(buf, 8);
        GroupElement g(*(uint64_t *)buf);
        mod(g, bl);
        return g;
    }
    else if (bl > 16)
    {
        char buf[4];
        this->keyBuf->read(buf, 4);
        GroupElement g(*(uint32_t *)buf);
        mod(g, bl);
        return g;
    }
    else if (bl > 8)
    {
        char buf[2];
        this->keyBuf->read(buf, 2);
        GroupElement g(*(uint16_t *)buf);
        mod(g, bl);
        return g;
    }
    else
    {
        char buf[1];
        this->keyBuf->read(buf, 1);
        GroupElement g(*(uint8_t *)buf);
        mod(g, bl);
        return g;
    }
}

GroupElement *Dealer::recv_ge_array(int bw, int size)
{
    always_assert(bw > 32);
    return (GroupElement *)this->keyBuf->read(8 * size);
}

void Dealer::recv_ge_array(const GroupElement *g, int size)
{
    char *buf = (char *)g;
    this->keyBuf->read(buf, 8 * size);
}

DCFKeyPack Dealer::recv_dcf_keypack(int Bin, int Bout, int groupSize)
{
    DCFKeyPack kp;
    kp.Bin = Bin;
    kp.Bout = Bout;
    kp.groupSize = groupSize;
    kp.k = recv_block_array(Bin + 1);
    kp.g = new GroupElement[groupSize];
    for (int i = 0; i < groupSize; ++i)
    {
        kp.g[i] = recv_ge(Bout);
    }
    kp.v = new GroupElement[Bin * groupSize];
    for (int i = 0; i < Bin * groupSize; ++i)
    {
        kp.v[i] = recv_ge(Bout);
    }
    GroupElement t = recv_ge(64);
    always_assert(t == 42);
    return kp;
}

DPFKeyPack Dealer::recv_dpf_keypack(int bin, int bout)
{
    DPFKeyPack kp;
    kp.bin = bin;
    kp.bout = bout;
    kp.s = recv_block_array(bin + 1);
    kp.tLcw = recv_ge(bin);
    kp.tRcw = recv_ge(bin);
    kp.payload = recv_ge(bout);

    GroupElement t = recv_ge(64);
    always_assert(t == 42);
    return kp;
}

DPFETKeyPack Dealer::recv_dpfet_keypack(int bin)
{
    DPFETKeyPack kp;
    kp.bin = bin;
    kp.s = recv_block_array(bin - 6);
    kp.tLcw = recv_ge(bin);
    kp.tRcw = recv_ge(bin);
    kp.leaf = recv_block();

    GroupElement t = recv_ge(64);
    always_assert(t == 42);
    return kp;
}

DualDCFKeyPack Dealer::recv_ddcf_keypack(int Bin, int Bout, int groupSize)
{
    DualDCFKeyPack kp;
    kp.Bin = Bin;
    kp.Bout = Bout;
    kp.groupSize = groupSize;
    kp.dcfKey = recv_dcf_keypack(Bin, Bout, groupSize);
    kp.sb = new GroupElement[groupSize];
    for (int i = 0; i < groupSize; ++i)
    {
        kp.sb[i] = recv_ge(Bout);
    }
    return kp;
}

MultKeyNew Dealer::recv_new_mult_key(int bw1, int bw2)
{
    MultKeyNew kp;
    kp.a = recv_ge(bw1 + bw2);
    kp.b = recv_ge(bw1 + bw2);
    kp.c = recv_ge(bw1 + bw2);
    kp.k1 = recv_dcf_keypack(bw2, bw1, 1);
    kp.k2 = recv_dcf_keypack(bw1, bw2, 1);
    kp.k3 = recv_dcf_keypack(bw2, bw1, 1);
    kp.k4 = recv_dcf_keypack(bw1, bw2, 1);

    return kp;
}

MatMulKey Dealer::recv_matmul_key(int Bin, int Bout, int s1, int s2, int s3)
{
    MatMulKey k;
    k.Bin = Bin;
    k.Bout = Bout;
    k.s1 = s1;
    k.s2 = s2;
    k.s3 = s3;

    k.a = make_array<GroupElement>(s1, s2);
    k.b = make_array<GroupElement>(s2, s3);
    k.c = make_array<GroupElement>(s1, s3);

    for (int i = 0; i < s1; ++i)
    {
        for (int j = 0; j < s2; ++j)
        {
            Arr2DIdx(k.a, s1, s2, i, j) = (party == SERVER ? GroupElement(prngShared.get<uint64_t>()) : recv_ge(Bin));
            mod(Arr2DIdx(k.a, s1, s2, i, j), Bin);
        }
    }

    for (int i = 0; i < s2; ++i)
    {
        for (int j = 0; j < s3; ++j)
        {
            Arr2DIdx(k.b, s2, s3, i, j) = (party == SERVER ? GroupElement(prngShared.get<uint64_t>()) : recv_ge(Bin));
            mod(Arr2DIdx(k.b, s2, s3, i, j), Bin);
        }
    }

    for (int i = 0; i < s1; ++i)
    {
        for (int j = 0; j < s3; ++j)
        {
            Arr2DIdx(k.c, s1, s3, i, j) = (party == SERVER ? GroupElement(prngShared.get<uint64_t>()) : recv_ge(Bout));
            mod(Arr2DIdx(k.c, s1, s3, i, j), Bout);
        }
    }

    return k;
}

Conv2DKey Dealer::recv_conv2d_key(int Bin, int Bout, int64_t N, int64_t H, int64_t W,
                                  int64_t CI, int64_t FH, int64_t FW,
                                  int64_t CO, int64_t zPadHLeft,
                                  int64_t zPadHRight, int64_t zPadWLeft,
                                  int64_t zPadWRight, int64_t strideH,
                                  int64_t strideW)
{
    Conv2DKey k;
    k.Bin = Bin;
    k.Bout = Bout;
    k.N = N;
    k.H = H;
    k.W = W;
    k.CO = CO;
    k.CI = CI;
    k.FH = FH;
    k.FW = FW;
    k.zPadHLeft = zPadHLeft;
    k.zPadHRight = zPadHRight;
    k.zPadWLeft = zPadWLeft;
    k.zPadWRight = zPadWRight;
    ;
    k.strideH = strideH;
    k.strideW = strideW;

    int d0 = N;
    int d1 = ((H - FH + (zPadHLeft + zPadHRight)) / strideH) + 1;
    int d2 = ((W - FW + (zPadWLeft + zPadWRight)) / strideW) + 1;
    int d3 = CO;

    k.a = make_array<GroupElement>(N, H, W, CI);
    k.b = make_array<GroupElement>(FH, FW, CI, CO);
    k.c = make_array<GroupElement>(d0, d1, d2, d3);

    for (int n = 0; n < N; ++n)
    {
        for (int h = 0; h < H; ++h)
        {
            for (int w = 0; w < W; ++w)
            {
                for (int ci = 0; ci < CI; ++ci)
                {
                    Arr4DIdx(k.a, N, H, W, CI, n, h, w, ci) = (party == SERVER ? GroupElement(prngShared.get<uint64_t>()) : recv_ge(Bin));
                    mod(Arr4DIdx(k.a, N, H, W, CI, n, h, w, ci), Bin);
                }
            }
        }
    }

    for (int fh = 0; fh < FH; ++fh)
    {
        for (int fw = 0; fw < FW; ++fw)
        {
            for (int ci = 0; ci < CI; ++ci)
            {
                for (int co = 0; co < CO; ++co)
                {
                    Arr4DIdx(k.b, FH, FW, CI, CO, fh, fw, ci, co) = (party == SERVER ? GroupElement(prngShared.get<uint64_t>()) : recv_ge(Bin));
                    mod(Arr4DIdx(k.b, FH, FW, CI, CO, fh, fw, ci, co), Bin);
                }
            }
        }
    }

    GroupElement *c = k.c;
    for (int i = 0; i < d0; ++i)
    {
        for (int j = 0; j < d1; ++j)
        {
            for (int k = 0; k < d2; ++k)
            {
                for (int l = 0; l < d3; ++l)
                {
                    Arr4DIdx(c, d0, d1, d2, d3, i, j, k, l) = (party == SERVER ? GroupElement(prngShared.get<uint64_t>()) : recv_ge(Bout));
                    mod(Arr4DIdx(c, d0, d1, d2, d3, i, j, k, l), Bout);
                }
            }
        }
    }
    return k;
}

Conv3DKey Dealer::recv_conv3d_key(int Bin, int Bout, int64_t N, int64_t D, int64_t H, int64_t W,
                                  int64_t CI, int64_t FD, int64_t FH, int64_t FW, int64_t CO,
                                  int64_t zPadDLeft, int64_t zPadDRight,
                                  int64_t zPadHLeft, int64_t zPadHRight,
                                  int64_t zPadWLeft, int64_t zPadWRight,
                                  int64_t strideD, int64_t strideH, int64_t strideW)
{
    Conv3DKey k;
    k.Bin = Bin;
    k.Bout = Bout;
    k.N = N;
    k.D = D;
    k.H = H;
    k.W = W;
    k.CO = CO;
    k.CI = CI;
    k.FD = FD;
    k.FH = FH;
    k.FW = FW;
    k.zPadDLeft = zPadDLeft;
    k.zPadDRight = zPadDRight;
    k.zPadHLeft = zPadHLeft;
    k.zPadHRight = zPadHRight;
    k.zPadWLeft = zPadWLeft;
    k.zPadWRight = zPadWRight;
    k.strideD = strideD;
    k.strideH = strideH;
    k.strideW = strideW;

    int d0 = N;
    int d1 = ((D - FD + (zPadDLeft + zPadDRight)) / strideD) + 1;
    int d2 = ((H - FH + (zPadHLeft + zPadHRight)) / strideH) + 1;
    int d3 = ((W - FW + (zPadWLeft + zPadWRight)) / strideW) + 1;
    int d4 = CO;

    k.a = make_array<GroupElement>(N, D, H, W, CI);
    k.b = make_array<GroupElement>(FD, FH, FW, CI, CO);
    k.c = make_array<GroupElement>(d0, d1, d2, d3, d4);

    for (size_t i = 0; i < N * D * H * W * CI; ++i)
    {
        k.a[i] = (party == SERVER ? GroupElement(prngShared.get<uint64_t>()) : recv_ge(Bin));
        mod(k.a[i], Bin);
    }

    for (size_t i = 0; i < FD * FH * FW * CI * CO; ++i)
    {
        k.b[i] = (party == SERVER ? GroupElement(prngShared.get<uint64_t>()) : recv_ge(Bin));
        mod(k.b[i], Bin);
    }

    for (size_t i = 0; i < d0 * d1 * d2 * d3 * d4; ++i)
    {
        k.c[i] = (party == SERVER ? GroupElement(prngShared.get<uint64_t>()) : recv_ge(Bout));
        mod(k.c[i], Bout);
    }
    return k;
}

ReluKeyPack Dealer::recv_relu_key(int Bin, int Bout)
{
    int groupSize = 2;
    ReluKeyPack kp;
    kp.Bin = Bin;
    kp.Bout = Bout;
    kp.k = recv_block_array(Bin + 1);
    kp.g = new GroupElement[groupSize];
    for (int i = 0; i < groupSize; ++i)
    {
        kp.g[i] = recv_ge(Bout);
    }
    if (Bout > 32)
    {
        kp.v = recv_ge_array(Bout, Bin * groupSize);
    }
    else
    {
        kp.v = new GroupElement[Bin * groupSize];
        for (int i = 0; i < Bin * groupSize; ++i)
        {
            kp.v[i] = recv_ge(Bout);
        }
    }
    kp.e_b0 = recv_ge(Bout);
    kp.e_b1 = recv_ge(Bout);
    kp.beta_b0 = recv_ge(Bout);
    kp.beta_b1 = recv_ge(Bout);
    kp.r_b = recv_ge(Bout);
    kp.drelu = recv_ge(1);
    return kp;
}

MaxpoolKeyPack Dealer::recv_maxpool_key(int Bin, int Bout)
{
    MaxpoolKeyPack kp;
    kp.Bin = Bin;
    kp.Bout = Bout;
    kp.reluKey = recv_relu_key(Bin, Bout);
    kp.rb = recv_ge(Bout);
    return kp;
}

MaxpoolDoubleKeyPack Dealer::recv_maxpool_double_key(int Bin, int Bout)
{
    MaxpoolDoubleKeyPack kp;
    kp.Bin = Bin;
    kp.Bout = Bout;
    kp.reluKey = recv_relu_2round_key(Bin, Bout);
    kp.rb = recv_ge(Bout);
    return kp;
}

ScmpKeyPack Dealer::recv_scmp_keypack(int Bin, int Bout)
{
    int groupSize = 1;
    ScmpKeyPack kp;
    kp.Bin = Bin;
    kp.Bout = Bout;
    kp.dualDcfKey = recv_ddcf_keypack(Bin - 1, Bout, groupSize);
    kp.rb = recv_ge(Bout);
    return kp;
}

PublicDivKeyPack Dealer::recv_pubdiv_key(int Bin, int Bout)
{
    int groupSize = 1;
    PublicDivKeyPack kp;
    kp.Bin = Bin;
    kp.Bout = Bout;
    kp.dualDcfKey = recv_ddcf_keypack(Bin, Bout, groupSize);
    kp.scmpKey = recv_scmp_keypack(Bin, Bout);
    kp.zb = recv_ge(Bout);
    return kp;
}

ARSKeyPack Dealer::recv_ars_key(int Bin, int Bout, int shift)
{
    ARSKeyPack kp;
    kp.Bin = Bin;
    kp.Bout = Bout;
    kp.shift = shift;

    int dcfGroupSize = 1, ddcfGroupSize = 2;
    if (!FSSConfig::stochasticT)
        kp.dcfKey = recv_dcf_keypack(shift, Bout, dcfGroupSize);
    if (Bout > Bin - shift)
    {
        kp.dualDcfKey = recv_ddcf_keypack(Bin - 1, Bout, ddcfGroupSize);
    }
    kp.rb = recv_ge(Bout);
    return kp;
}

void Peer::sync()
{
    this->keyBuf->sync();
}

SignedPublicDivKeyPack Dealer::recv_signedpubdiv_key(int Bin, int Bout)
{
    SignedPublicDivKeyPack kp;
    kp.d = recv_ge(Bin);
    kp.Bin = Bin;
    kp.Bout = Bout;
    int groupSize = 1;
    kp.dcfKey = recv_dcf_keypack(Bin, Bout, groupSize);
    kp.publicICkey = recv_publicIC_key(Bin, Bout);
    kp.scmpKey = recv_scmp_keypack(Bin, Bout);
    kp.A_share = recv_ge(Bin);
    kp.corr_share = recv_ge(Bout);
    kp.B_share = recv_ge(Bout);
    kp.rdiv_share = recv_ge(Bout);
    kp.rout_temp_share = recv_ge(Bout);
    kp.rout_share = recv_ge(Bout);
    return kp;
}

PublicICKeyPack Dealer::recv_publicIC_key(int Bin, int Bout)
{
    PublicICKeyPack kp;
    kp.Bin = Bin;
    kp.Bout = Bout;
    int groupSize = 1;
    kp.dcfKey = recv_dcf_keypack(Bin, Bout, groupSize);
    kp.zb = recv_ge(Bout);
    return kp;
}

SplineKeyPack Dealer::recv_spline_key(int Bin, int Bout, int numPoly, int degree)
{
    SplineKeyPack kp;
    kp.Bin = Bin;
    kp.Bout = Bout;
    kp.numPoly = numPoly;
    kp.degree = degree;
    kp.dcfKey = recv_dcf_keypack(16, Bout, numPoly * (degree + 1));

    kp.p.resize(numPoly + 1);
    for (int i = 0; i < numPoly + 1; ++i)
    {
        kp.p[i] = recv_ge(Bin);
    }

    kp.e_b.resize(numPoly);
    for (int i = 0; i < numPoly; ++i)
    {
        kp.e_b[i].resize(degree + 1);
        for (int j = 0; j < degree + 1; ++j)
        {
            kp.e_b[i][j] = recv_ge(Bout);
        }
    }

    kp.beta_b.resize(numPoly * (degree + 1));
    for (int i = 0; i < numPoly * (degree + 1); ++i)
    {
        kp.beta_b[i] = recv_ge(Bout);
    }

    kp.r_b = recv_ge(Bout);
    return kp;
}

ReluTruncateKeyPack Dealer::recv_relu_truncate_key(int Bin, int Bout, int s)
{
    ReluTruncateKeyPack kp;
    kp.Bin = Bin;
    kp.Bout = Bout;
    kp.shift = s;
    kp.dcfKeyN = recv_dcf_keypack(Bin, s, 1);
    if (!FSSConfig::stochasticRT)
        kp.dcfKeyS = recv_dcf_keypack(s, Bin, 1);
    kp.zTruncate = recv_ge(Bin);
    kp.a = recv_ge(Bin);
    kp.b = recv_ge(Bin);
    kp.c = recv_ge(Bin);
    kp.d1 = recv_ge(Bin);
    kp.d2 = recv_ge(Bin);
    return kp;
}

Relu2RoundKeyPack Dealer::recv_relu_2round_key(int effectiveBin, int Bin)
{
    Relu2RoundKeyPack kp;
    kp.effectiveBin = effectiveBin;
    kp.Bin = Bin;
    kp.dcfKey = recv_dcf_keypack(effectiveBin, 1, 1);
    kp.a = recv_ge(Bin);
    kp.b = recv_ge(Bin);
    kp.c = recv_ge(Bin);
    kp.d1 = recv_ge(Bin);
    kp.d2 = recv_ge(Bin);
    return kp;
}

SelectKeyPack Dealer::recv_select_key(int Bin)
{
    SelectKeyPack kp;
    kp.Bin = Bin;
    kp.a = recv_ge(Bin);
    kp.b = recv_ge(Bin);
    kp.c = recv_ge(Bin);
    kp.d1 = recv_ge(Bin);
    kp.d2 = recv_ge(Bin);
    return kp;
}

void Peer::send_bulkylrs_key(const BulkyLRSKeyPack &kp, int bl, int m)
{
    send_dcf_keypack(kp.dcfKeyN);
    for (int i = 0; i < m; ++i)
    {
        send_dcf_keypack(kp.dcfKeyS[i]);
        send_ge(kp.z[i], bl);
    }
    send_ge(kp.out, bl);
}

void Peer::send_taylor_key(const TaylorKeyPack &kp, int bl, int m)
{
    send_dcf_keypack(kp.msnzbKey.micKey.dcfKey);
    for (int i = 0; i < m; ++i)
    {
        send_ge(kp.msnzbKey.micKey.z[i], bl);
    }
    send_ge(kp.msnzbKey.r, bl);
    send_ge(kp.squareKey.a, bl);
    send_ge(kp.squareKey.b, bl);
    send_ge(kp.privateScaleKey.rin, bl);
    send_ge(kp.privateScaleKey.rout, bl);
    send_bulkylrs_key(kp.lrsKeys[0], bl, m);
    send_bulkylrs_key(kp.lrsKeys[1], bl, m);
    send_ge(69, bl);
}
BulkyLRSKeyPack Dealer::recv_bulkylrs_key(int bl, int m, uint64_t *scales)
{
    BulkyLRSKeyPack kp;
    kp.dcfKeyN = recv_dcf_keypack(bl, bl, 1);
    kp.z = new GroupElement[m];
    kp.dcfKeyS = new DCFKeyPack[m];
    for (int i = 0; i < m; ++i)
    {
        kp.dcfKeyS[i] = recv_dcf_keypack(scales[i], bl, 1);
        kp.z[i] = recv_ge(bl);
    }
    kp.out = recv_ge(bl);
    return kp;
}

TaylorKeyPack Dealer::recv_taylor_key(int bl, int m, int sf)
{
    TaylorKeyPack kp;
    kp.msnzbKey.micKey.dcfKey = recv_dcf_keypack(bl, bl, 1);
    kp.msnzbKey.micKey.z = new GroupElement[m];
    for (int i = 0; i < m; ++i)
    {
        kp.msnzbKey.micKey.z[i] = recv_ge(bl);
    }
    kp.msnzbKey.r = recv_ge(bl);
    kp.squareKey.a = recv_ge(bl);
    kp.squareKey.b = recv_ge(bl);
    kp.privateScaleKey.rin = recv_ge(bl);
    kp.privateScaleKey.rout = recv_ge(bl);
    uint64_t scales[m];
    for (int i = 0; i < m; ++i)
    {
        scales[i] = sf + i;
    }
    kp.lrsKeys[0] = recv_bulkylrs_key(bl, m, scales);
    for (int i = 0; i < m; ++i)
    {
        scales[i] = sf + 3 * i;
    }
    kp.lrsKeys[1] = recv_bulkylrs_key(bl, m, scales);
    GroupElement ping = recv_ge(bl);
    always_assert(ping == 69);
    return kp;
}

BitwiseAndKeyPack Dealer::recv_bitwise_and_key()
{
    BitwiseAndKeyPack kp;
    kp.t[0] = recv_ge(64);
    kp.t[1] = recv_ge(64);
    kp.t[2] = recv_ge(64);
    kp.t[3] = recv_ge(64);
    return kp;
}

// void Peer::send_mic_key(const MICKeyPack &kp, int bl, int m)
// {
//     send_dcf_keypack(kp.dcfKey);
//     for(int i = 0; i < m; ++i)
//     {
//         send_ge(kp.z[i], bl);
//     }
// }

// void Peer::send_fix_to_float_key(const FixToFloatKeyPack &kp, int bl)
// {
//     send_mic_key(kp.micKey, bl, 2*bl);
// }

MICKeyPack Dealer::recv_mic_key(int bin, int bout, int m)
{
    MICKeyPack kp;
    kp.dcfKey = recv_dcf_keypack(bin, bout, 1);
    kp.z = new GroupElement[m];
    for (int i = 0; i < m; ++i)
    {
        kp.z[i] = recv_ge(bout);
    }
    return kp;
}

FixToFloatKeyPack Dealer::recv_fix_to_float_key(int bl)
{
    FixToFloatKeyPack kp;
    kp.micKey = recv_mic_key(bl, bl, 2 * bl);
    kp.rs = recv_ge(1);
    kp.rpow = recv_ge(bl);
    kp.ry = recv_ge(bl);
    kp.selectKey = recv_select_key(bl);
    kp.rm = recv_ge(bl);
    return kp;
}

FloatToFixKeyPack Dealer::recv_float_to_fix_key(int bl)
{
    FloatToFixKeyPack kp;
    kp.dcfKey = recv_dcf_keypack(24, 1, 1);
    kp.selectKey = recv_select_key(bl);
    kp.rm = recv_ge(24);
    kp.re = recv_ge(10);
    kp.rw = recv_ge(1);
    // kp.rt = recv_ge(bl);
    kp.rh = recv_ge(bl);
    for (int i = 0; i < 1024; ++i)
    {
        kp.p[i] = recv_ge(bl);
    }
    for (int i = 0; i < 1024; ++i)
    {
        kp.q[i] = recv_ge(bl);
    }
    kp.arsKey = recv_ars_key(bl, bl, 23);
    return kp;
}

ReluExtendKeyPack Dealer::recv_relu_extend_key(int bin, int bout)
{
    ReluExtendKeyPack kp;
    kp.dcfKey = recv_dcf_keypack(bin, 2, 1);
    kp.rd = recv_ge(2);
    kp.rw = recv_ge(2);
    for (int i = 0; i < 4; ++i)
    {
        kp.p[i] = recv_ge(bout);
    }
    for (int i = 0; i < 2; ++i)
    {
        kp.q[i] = recv_ge(bout);
    }
    return kp;
}

SignExtend2KeyPack Dealer::recv_sign_extend2_key(int Bin, int Bout)
{
    SignExtend2KeyPack kp;
    kp.dcfKey = recv_dcf_keypack(Bin, 1, 1);
    kp.rw = recv_ge(1);
    kp.p[0] = recv_ge(Bout);
    kp.p[1] = recv_ge(Bout);
    return kp;
}

TripleKeyPack Dealer::recv_triple_key(int bw, int64_t na, int64_t nb, int64_t nc)
{
    TripleKeyPack k;
    k.bw = bw;
    k.na = na;
    k.nb = nb;
    k.nc = nc;
    k.a = make_array<GroupElement>(na);
    k.b = make_array<GroupElement>(nb);
    k.c = make_array<GroupElement>(nc);

    for (size_t i = 0; i < na; ++i)
    {
        k.a[i] = (party == SERVER ? GroupElement(prngShared.get<uint64_t>()) : recv_ge(bw));
        mod(k.a[i], bw);
    }

    for (size_t i = 0; i < nb; ++i)
    {
        k.b[i] = (party == SERVER ? GroupElement(prngShared.get<uint64_t>()) : recv_ge(bw));
        mod(k.b[i], bw);
    }

    for (size_t i = 0; i < nc; ++i)
    {
        k.c[i] = (party == SERVER ? GroupElement(prngShared.get<uint64_t>()) : recv_ge(bw));
        mod(k.c[i], bw);
    }
    return k;
}

EdabitsPrTruncKeyPack Dealer::recv_edabits_prtrunc_key(int bw)
{
    EdabitsPrTruncKeyPack kp;
    kp.a = recv_ge(bw);
    kp.b = recv_ge(bw);
    return kp;
}

PubCmpKeyPack Dealer::recv_pubcmp_key(int bin)
{
    PubCmpKeyPack kp;
    kp.bin = bin;
    kp.dcfKey = recv_dcf_keypack(bin, 1, 1);
    kp.rout = recv_ge(1);
    return kp;
}

ClipKeyPack Dealer::recv_clip_key(int bin)
{
    ClipKeyPack kp;
    kp.bin = bin;
    kp.cmpKey = recv_pubcmp_key(bin);
    kp.a = recv_ge(bin);
    kp.b = recv_ge(bin);
    kp.c = recv_ge(bin);
    kp.d1 = recv_ge(bin);
    kp.d2 = recv_ge(bin);
    return kp;
}

LUTKeyPack Dealer::recv_lut_key(int bin, int bout)
{
    LUTKeyPack kp;
    kp.bin = bin;
    kp.bout = bout;

    kp.dpfKey = recv_dpf_keypack(bin, bout);
    kp.rout = recv_ge(bout);
    return kp;
}

F2BF16KeyPack Dealer::recv_f2bf16_key(int bin)
{
    F2BF16KeyPack kp;
    kp.bin = bin;
    kp.dcfKey = recv_dcf_keypack(bin, bin, 1);
    kp.dcfTruncate = recv_dcf_keypack(bin - 8, 8, 1);
    kp.rout_k = recv_ge(bin);
    kp.rout_m = recv_ge(bin);
    kp.rin = recv_ge(bin);
    kp.prod = recv_ge(bin);
    kp.rout = recv_ge(13);
    kp.rProd = recv_ge(bin);
    return kp;
}

TruncateReduceKeyPack Dealer::recv_truncate_reduce_key(int bin, int shift)
{
    TruncateReduceKeyPack kp;
    kp.bin = bin;
    kp.shift = shift;

    kp.dcfKey = recv_dcf_keypack(shift, bin - shift, 1);
    kp.rout = recv_ge(bin - shift);
    return kp;
}

LUTSSKeyPack Dealer::recv_lutss_key(int bin, int bout)
{
    LUTSSKeyPack kp;
    kp.bin = bin;
    kp.bout = bout;

    kp.b0 = recv_ge(64);
    kp.b1 = recv_ge(64);
    kp.b2 = recv_ge(64);
    kp.b3 = recv_ge(64);
    kp.routRes = recv_ge(bout);
    kp.routCorr = recv_ge(bout);
    kp.rout = recv_ge(bout);
    return kp;
}

LUTDPFETKeyPack Dealer::recv_lutdpfet_key(int bin, int bout)
{
    LUTDPFETKeyPack kp;
    kp.bin = bin;
    kp.bout = bout;

    kp.dpfKey = recv_dpfet_keypack(bin);
    kp.routRes = recv_ge(bout);
    kp.routCorr = recv_ge(1);
    return kp;
}

SlothDreluKeyPack Dealer::recv_slothdrelu_key(int bin)
{
    SlothDreluKeyPack kp;
    kp.bin = bin;
    kp.dpfKey = recv_dpfet_keypack(bin - 1);
    kp.r = recv_ge(1);
    return kp;
}

WrapDPFKeyPack Dealer::recv_wrap_dpf_key(int bin)
{
    WrapDPFKeyPack kp;
    kp.bin = bin;
    kp.dpfKey = recv_dpfet_keypack(bin);
    kp.r = recv_ge(1);
    return kp;
}

WrapSSKeyPack Dealer::recv_wrap_ss_key(int bin)
{
    WrapSSKeyPack kp;
    kp.bin = bin;
    kp.b0 = recv_ge(64);
    if (bin == 7)
        kp.b1 = recv_ge(64);
    return kp;
}

SlothLRSKeyPack Dealer::recv_sloth_lrs_key(int bin, int shift)
{
    SlothLRSKeyPack kp;
    kp.bin = bin;
    kp.shift = shift;
    kp.rout = recv_ge(bin);
    kp.msb = recv_ge(bin);
    kp.select = recv_ge(bin);

    return kp;
}

SlothSignExtendKeyPack Dealer::recv_sloth_sign_extend_key(int bin, int bout)
{
    SlothSignExtendKeyPack kp;
    kp.bin = bin;
    kp.bout = bout;
    kp.rout = recv_ge(bout);
    kp.select = recv_ge(bout);

    return kp;
}

void Peer::send_sloth_sign_extend_key(const SlothSignExtendKeyPack &kp)
{
    send_ge(kp.rout, kp.bout);
    send_ge(kp.select, kp.bout);
}

size_t bw_to_bytes(int bw) {
    if (bw > 32) return 8;
    if (bw > 16) return 4;
    if (bw > 8) return 2;
    return 1;
}

size_t get_dpf_key_pack_size_in_bytes(const DPFKeyPack& kp) {
    return (kp.bin + 1) * sizeof(osuCrypto::block) + 
           sizeof(kp.tLcw) +  // 始终传输 8 字节
           sizeof(kp.tRcw) +  // 始终传输 8 字节
           sizeof(kp.payload); // 始终传输 8 字节
}
// In aux_parameter/comms.cpp (Peer/Dealer class implementation)
void Peer::send_elemwisemul_key(const ElemWiseMulKeyPack &k) {
    send_ge_array(k.a, k.size);
    send_ge_array(k.b, k.size);
    send_ge_array(k.c, k.size);
}

ElemWiseMulKeyPack Dealer::recv_elemwisemul_key(int32_t size) {
    ElemWiseMulKeyPack k;
    k.size = size;
    k.a = new GroupElement[size];
    k.b = new GroupElement[size];
    k.c = new GroupElement[size];
    recv_ge_array(k.a, size);
    recv_ge_array(k.b, size);
    recv_ge_array(k.c, size);
    return k;
}
void Peer::send_dpf_route_key(const DpfRouteKeyPack &k) {
    
    // 1. 计算总大小
    const GroupElement SENTINEL = 42;
    const size_t SENTINEL_SIZE = sizeof(GroupElement); // 8 字节
    size_t total_dpf_keys_size = 0;
    for (int i = 0; i < k.size; ++i) {
        total_dpf_keys_size += get_dpf_key_pack_size_in_bytes(k.routing_keys[i]);
    }
    size_t r_shares_size = k.size * sizeof(GroupElement);
    size_t s_shares_size = k.size * sizeof(GroupElement);
    size_t total_size = SENTINEL_SIZE + total_dpf_keys_size + r_shares_size + s_shares_size;

    // 2. 创建缓冲区并拷贝数据
    char* buffer = new char[total_size];
    char* current_ptr = buffer;
    memcpy(current_ptr, &SENTINEL, SENTINEL_SIZE);
    current_ptr += SENTINEL_SIZE;
    // 2.1 拷贝 DPF 密钥
    for (int i = 0; i < k.size; ++i) {
        const DPFKeyPack& dpf_key = k.routing_keys[i];
        size_t s_bytes = (dpf_key.bin + 1) * sizeof(osuCrypto::block);
        memcpy(current_ptr, dpf_key.s, s_bytes);
        current_ptr += s_bytes;
        memcpy(current_ptr, &dpf_key.tLcw, sizeof(dpf_key.tLcw));
        current_ptr += sizeof(dpf_key.tLcw);
        memcpy(current_ptr, &dpf_key.tRcw, sizeof(dpf_key.tRcw));
        current_ptr += sizeof(dpf_key.tRcw);
        memcpy(current_ptr, &dpf_key.payload, sizeof(dpf_key.payload));
        current_ptr += sizeof(dpf_key.payload);
    }
    
    memcpy(current_ptr, k.r_shares, r_shares_size);
    current_ptr += r_shares_size;

    memcpy(current_ptr, k.s_shares, s_shares_size);
    current_ptr += s_shares_size;

    this->keyBuf->write(buffer, total_size);


    delete[] buffer;
}

DpfRouteKeyPack Dealer::recv_dpf_route_key(int size, int data_bin, int rank_bin) {
    const GroupElement SENTINEL = 42;
    const size_t SENTINEL_SIZE = sizeof(GroupElement);
    // 1. 创建密钥包对象 (构造函数会分配内部数组)
    DpfRouteKeyPack k(size, data_bin, rank_bin);
    for (int i = 0; i < size; ++i) {
        k.routing_keys[i].bin = rank_bin;
        k.routing_keys[i].bout = data_bin;
    }
    
    // 2. 计算总大小并一次性接收
    size_t total_dpf_keys_size = 0;
    for (int i = 0; i < size; ++i) {
        total_dpf_keys_size += get_dpf_key_pack_size_in_bytes(k.routing_keys[i]);
    }
    size_t r_shares_size = size * sizeof(GroupElement);
    size_t s_shares_size = size * sizeof(GroupElement);
    size_t total_size = SENTINEL_SIZE + total_dpf_keys_size + r_shares_size + s_shares_size;
    


    char* buffer = new char[total_size];
    this->keyBuf->read(buffer, total_size);
    char* current_ptr = buffer;
    GroupElement t;
    memcpy(&t, current_ptr, SENTINEL_SIZE);
    current_ptr += SENTINEL_SIZE;
    always_assert(t == SENTINEL);
    for (int i = 0; i < size; ++i) {
        DPFKeyPack& dpf_key = k.routing_keys[i];
        dpf_key.s = new osuCrypto::block[dpf_key.bin + 1];
        size_t s_bytes = (dpf_key.bin + 1) * sizeof(osuCrypto::block);
        memcpy(dpf_key.s, current_ptr, s_bytes);
        current_ptr += s_bytes;
        memcpy(&dpf_key.tLcw, current_ptr, sizeof(dpf_key.tLcw));
        current_ptr += sizeof(dpf_key.tLcw);
        memcpy(&dpf_key.tRcw, current_ptr, sizeof(dpf_key.tRcw));
        current_ptr += sizeof(dpf_key.tRcw);
        memcpy(&dpf_key.payload, current_ptr, sizeof(dpf_key.payload));
        current_ptr += sizeof(dpf_key.payload);
    }
      std::cerr << "\nend for \n" << std::endl;
    // 3.2 解析 r_shares
    memcpy(k.r_shares, current_ptr, r_shares_size);
    current_ptr += r_shares_size;

    // 3.3 解析 s_shares
    memcpy(k.s_shares, current_ptr, s_shares_size);
    // 4. 清理并返回
    delete[] buffer;
    return k;
}

void Peer::recv_ge_array(GroupElement *arr, int size)
{
    int bitlength = sizeof(GroupElement) * 8; 
    recv_batched_input(arr, size, bitlength);
}

// void Peer::send_graph_update_key(const GraphUpdateKeyPack &k) {
//     size_t total_size = 0;
//     const GroupElement SENTINEL = 42;
//     const size_t SENTINEL_SIZE = sizeof(GroupElement);
//     total_size +=  SENTINEL_SIZE + sizeof(k.n) + sizeof(k.c) + sizeof(k.A_bw) + sizeof(k.A_data_bw) + sizeof(k.F_bw) + sizeof(k.F_data_bw);
    
//     size_t keys_A_size = 0;
//     for (int i = 0; i < k.n; ++i) {
//         keys_A_size += get_dpf_key_pack_size_in_bytes(k.keys_A[i]);
//     }
//     total_size += keys_A_size;

//     size_t keys_F_size = 0;
//     for (int i = 0; i < k.c; ++i) {
//         keys_F_size += get_dpf_key_pack_size_in_bytes(k.keys_F[i]);
//     }
//     total_size += keys_F_size;

//     char* buffer = new char[total_size];
//     char* current_ptr = buffer;
//     memcpy(current_ptr, &SENTINEL, SENTINEL_SIZE);current_ptr += SENTINEL_SIZE;
//     memcpy(current_ptr, &k.n, sizeof(k.n)); current_ptr += sizeof(k.n);
//     memcpy(current_ptr, &k.c, sizeof(k.c)); current_ptr += sizeof(k.c);
//     memcpy(current_ptr, &k.A_bw, sizeof(k.A_bw)); current_ptr += sizeof(k.A_bw);
//     memcpy(current_ptr, &k.A_data_bw, sizeof(k.A_data_bw)); current_ptr += sizeof(k.A_data_bw);
//     memcpy(current_ptr, &k.F_bw, sizeof(k.F_bw)); current_ptr += sizeof(k.F_bw);
//     memcpy(current_ptr, &k.F_data_bw, sizeof(k.F_data_bw)); current_ptr += sizeof(k.F_data_bw);

//     for (int i = 0; i < k.n; ++i) {
//         const DPFKeyPack& dpf_key = k.keys_A[i];
//         size_t s_bytes = (dpf_key.bin + 1) * sizeof(osuCrypto::block);
        
//         memcpy(current_ptr, dpf_key.s, s_bytes); current_ptr += s_bytes;
//         memcpy(current_ptr, &dpf_key.tLcw, sizeof(dpf_key.tLcw)); current_ptr += sizeof(dpf_key.tLcw);
//         memcpy(current_ptr, &dpf_key.tRcw, sizeof(dpf_key.tRcw)); current_ptr += sizeof(dpf_key.tRcw);
//         memcpy(current_ptr, &dpf_key.payload, sizeof(dpf_key.payload)); current_ptr += sizeof(dpf_key.payload);
//     }

//     for (int i = 0; i < k.c; ++i) {
//         const DPFKeyPack& dpf_key = k.keys_F[i];
//         size_t s_bytes = (dpf_key.bin + 1) * sizeof(osuCrypto::block);

//         memcpy(current_ptr, dpf_key.s, s_bytes); current_ptr += s_bytes;
//         memcpy(current_ptr, &dpf_key.tLcw, sizeof(dpf_key.tLcw)); current_ptr += sizeof(dpf_key.tLcw);
//         memcpy(current_ptr, &dpf_key.tRcw, sizeof(dpf_key.tRcw)); current_ptr += sizeof(dpf_key.tRcw);
//         memcpy(current_ptr, &dpf_key.payload, sizeof(dpf_key.payload)); current_ptr += sizeof(dpf_key.payload);
//     }
    
//     this->keyBuf->write(buffer, total_size);

//     delete[] buffer;
// }

// GraphUpdateKeyPack Dealer::recv_graph_update_key() {
//     const GroupElement EXPECTED_SENTINEL = 42;
//     GroupElement received_sentinel;
    
//     this->keyBuf->read((char*)&received_sentinel, sizeof(GroupElement));
    
//     always_assert(received_sentinel == EXPECTED_SENTINEL);

//     int n, c, A_bw, A_data_bw, F_bw, F_data_bw;
//     this->keyBuf->read((char*)&n, sizeof(int));
//     this->keyBuf->read((char*)&c, sizeof(int));
//     this->keyBuf->read((char*)&A_bw, sizeof(int));
//     this->keyBuf->read((char*)&A_data_bw, sizeof(int));
//     this->keyBuf->read((char*)&F_bw, sizeof(int));
//     this->keyBuf->read((char*)&F_data_bw, sizeof(int));

//     GraphUpdateKeyPack k(n, c, A_bw, F_bw, A_data_bw, F_data_bw);

//     size_t keys_A_size = 0;
//     for (int i = 0; i < n; ++i) {
//         k.keys_A[i].bin = A_bw;
//         k.keys_A[i].bout = A_data_bw;
//         keys_A_size += get_dpf_key_pack_size_in_bytes(k.keys_A[i]);
//     }

//     size_t keys_F_size = 0;
//     for (int i = 0; i < c; ++i) {
//         k.keys_F[i].bin = F_bw;
//         k.keys_F[i].bout = F_data_bw;
//         keys_F_size += get_dpf_key_pack_size_in_bytes(k.keys_F[i]);
//     }
    
//     size_t remaining_size = keys_A_size + keys_F_size;

//     char* buffer = new char[remaining_size];
//     this->keyBuf->read(buffer, remaining_size);
    
//     char* current_ptr = buffer;

//     for (int i = 0; i < n; ++i) {
//         DPFKeyPack& dpf_key = k.keys_A[i];
        
//         dpf_key.s = new osuCrypto::block[dpf_key.bin + 1];
//         size_t s_bytes = (dpf_key.bin + 1) * sizeof(osuCrypto::block);
        
//         memcpy(dpf_key.s, current_ptr, s_bytes); 
//         current_ptr += s_bytes;
        
//         memcpy(&dpf_key.tLcw, current_ptr, sizeof(dpf_key.tLcw)); 
//         current_ptr += sizeof(dpf_key.tLcw);
        
//         memcpy(&dpf_key.tRcw, current_ptr, sizeof(dpf_key.tRcw)); 
//         current_ptr += sizeof(dpf_key.tRcw);
        
//         memcpy(&dpf_key.payload, current_ptr, sizeof(dpf_key.payload)); 
//         current_ptr += sizeof(dpf_key.payload);
//     }

//     for (int i = 0; i < c; ++i) {
//         DPFKeyPack& dpf_key = k.keys_F[i];
        
//         dpf_key.s = new osuCrypto::block[dpf_key.bin + 1];
//         size_t s_bytes = (dpf_key.bin + 1) * sizeof(osuCrypto::block);

//         memcpy(dpf_key.s, current_ptr, s_bytes); 
//         current_ptr += s_bytes;

//         memcpy(&dpf_key.tLcw, current_ptr, sizeof(dpf_key.tLcw)); 
//         current_ptr += sizeof(dpf_key.tLcw);

//         memcpy(&dpf_key.tRcw, current_ptr, sizeof(dpf_key.tRcw)); 
//         current_ptr += sizeof(dpf_key.tRcw);

//         memcpy(&dpf_key.payload, current_ptr, sizeof(dpf_key.payload)); 
//         current_ptr += sizeof(dpf_key.payload);
//     }
    
//     delete[] buffer;
//     return k;
// }

void Dealer::recv_dpf_keypack_t(DPFKeyPack& kp, int bin, int bout)
{
    // 不再创建临时的 DPFKeyPack kp;
    // 直接使用传入的引用 kp
    kp.bin = bin;
    kp.bout = bout;

    // recv_block_array 分配的内存直接赋给 kp.s
    kp.s = recv_block_array(bin + 1);

    kp.tLcw = recv_ge(bin);
    kp.tRcw = recv_ge(bin);
    kp.payload = recv_ge(bout);

    GroupElement t = recv_ge(64);
    always_assert(t == 42);

    // 无需返回，因为我们已经修改了传入的对象
}

void Peer::send_graph_update_key(const GraphUpdateKeyPack &k) {
    // 1. 发送元数据
    // 使用 send_ge 而不是直接写入，以保持代码风格统一性
    // 这里我们假设密钥的元数据（如维度、位宽）可以安全地用64位GroupElement传输
    send_ge((GroupElement)k.n, 64);
    send_ge((GroupElement)k.c, 64);
    send_ge((GroupElement)k.A_bw, 64);
    send_ge((GroupElement)k.A_data_bw, 64);
    send_ge((GroupElement)k.F_bw, 64);
    send_ge((GroupElement)k.F_data_bw, 64);

    // 2. 逐个发送 keys_A 中的 DPF 密钥
    for (int i = 0; i < k.n; ++i) {
        send_dpf_keypack(k.keys_A[i]);
    }

    // 3. 逐个发送 keys_F 中的 DPF 密钥
    for (int i = 0; i < k.c; ++i) {
        send_dpf_keypack(k.keys_F[i]);
    }
}

GraphUpdateKeyPack Dealer::recv_graph_update_key() {
    // 1. 接收元数据 (这部分保持不变)
    int n = (int)recv_ge(64);
    int c = (int)recv_ge(64);
    int A_bw = (int)recv_ge(64);
    int A_data_bw = (int)recv_ge(64);
    int F_bw = (int)recv_ge(64);
    int F_data_bw = (int)recv_ge(64);

    // 2. 构造密钥包对象
    GraphUpdateKeyPack k(n, c, A_bw, F_bw, A_data_bw, F_data_bw);

    // 3. 逐个接收 keys_A 的 DPF 密钥 (修改调用方式)
    for (int i = 0; i < n; ++i) {
        // 直接在 k.keys_A[i] 上填充数据，没有临时对象，没有赋值操作
        recv_dpf_keypack_t(k.keys_A[i], A_bw, A_data_bw);
    }

    // 4. 逐个接收 keys_F 的 DPF 密钥 (修改调用方式)
    for (int i = 0; i < c; ++i) {
        recv_dpf_keypack_t(k.keys_F[i], F_bw, F_data_bw);
    }

    // 5. 返回构造好的密钥包
    return k;
}

// 在 comms.cpp
void Peer::send_fast_relu_key(const FastReluKeyPack &kp)
{
    // 复用已有的 dcf_keypack 发送逻辑
    send_dcf_keypack(kp.dcfKey);
    // 发送额外的 r_sh
    send_ge(kp.r_sh, kp.Bin);

    send_ge_array(kp.b_sh, 2); 
}

FastReluKeyPack Dealer::recv_fast_relu_key(int Bin, int Bout)
{
    FastReluKeyPack kp;
    kp.Bin = Bin;
    kp.Bout = Bout;
    // groupSize硬编码为2，因为这是FastSecNet ReLU协议的要求
    kp.dcfKey = recv_dcf_keypack(Bin, Bout, 2); 
    kp.r_sh = recv_ge(Bin);
    recv_ge_array(kp.b_sh, 2);
    return kp;
}

void Peer::send_ars_crypten_key(const ARS_CrypTen_Style_KeyPack &kp) {
    keyBuf->write((char *)&kp.r_share, sizeof(GroupElement));
    keyBuf->write((char *)&kp.theta_r_share, sizeof(GroupElement));
}

// --- Dealer (接收方) 的实现 ---
ARS_CrypTen_Style_KeyPack Dealer::recv_ars_crypten_key() {
    ARS_CrypTen_Style_KeyPack kp;
    keyBuf->read((char *)&kp.r_share, sizeof(GroupElement));
    keyBuf->read((char *)&kp.theta_r_share, sizeof(GroupElement));
    return kp;
}