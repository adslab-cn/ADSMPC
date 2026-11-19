# 三方 RSS TCP 通信

`crypto/commom/comms-3PC.h` 声明接口，`comms-3PC.cpp` 实现连接、序列化与共享随机流。
两者编入现有 `FSS` 静态库；头文件不包含旧 `comms.h` 或协议密钥头。
本模块支持 Linux/WSL IPv4 TCP，不依赖旧 `Peer`、`Dealer` 或 `Config::party` 的运行状态。

## 连接与生命周期

- `Endpoint3PC` 包含数值 IPv4 地址 `ip` 和非零 `uint16_t port`；默认 IP 为 `127.0.0.2`，
  供同一 Linux/WSL 中的三个进程使用。
- `Comm3PC(self, endpoints, options)` 中，地址数组始终按 P0/P1/P2 排列，所有进程使用相同配置。
- `Comm3PCOptions::connectTimeout`、`ioTimeout` 是 `std::chrono::milliseconds`，分别默认 30 秒，必须为正值。
- 每方先监听本地端口，随后向编号更小的参与方建连；每对参与方使用一条全双工连接。
  本地监听绑定全部 IPv4 接口，配置中的 IP 用于其他参与方主动连接；同一机器的三方使用不同端口。
  两端地址均为回环地址时，主动连接的源 IP 绑定到本方配置的 IP，源端口由系统分配；
  非回环连接保持系统选源行为。
- 构造函数完成参与方握手、三方就绪同步、成对种子初始化和最终同步后返回。
  任意启动顺序均可；所有参与方必须在建连总超时内启动。
- 单次 TCP 建连尝试最多等待 1 秒；单次握手最多等待 `min(ioTimeout, 1 秒)`，
  均受同一个建连总截止时间约束。临时拒绝、重置或等待超时会重试。
  错误的协议标识、版本、参与方编号或重复连接立即报错。
- 两个类均禁止拷贝和移动。`close()` 可重复调用，析构自动关闭。
  `comm.peer(party)`、`prev()`、`next()` 返回上下文拥有的连接引用，调用方不负责删除。
- `Peer3PC(fd, ioTimeout)` 可接管已连接的 socket，包括初始化失败时的所有权；
  通常由 `Comm3PC` 创建，单独构造不会执行身份握手或初始化 PRNG。

参数错误抛出 `std::invalid_argument`；超时、系统调用错误抛出 `std::system_error`；
协议不匹配抛出 `std::runtime_error`；使用已关闭连接抛出 `std::logic_error`。
I/O 失败关闭受影响连接，`Comm3PC` 的交换或同步失败关闭整个上下文。
发生通信失败后，应销毁上下文并由三方重新初始化，不能继续使用失去同步的随机流。
参数校验失败不会发送或接收数据。

## 收发接口

`Peer3PC` 提供以下接口。所有数组由调用方分配，数量为 `std::size_t`；
接收接口不返回需手动释放的新数组。

| 数据 | 发送 | 接收 |
|---|---|---|
| 原始字节 | `send_bytes(data, bytes)` | `recv_bytes(data, bytes)` |
| 环元素 | `send_ge(value, bw)` | `recv_ge(bw)` |
| 环元素数组 | `send_ge_array(data, count, bw)` | `recv_ge_array(data, count, bw)` |
| 128 位 block | `send_block(value)` | `recv_block()` |
| block 数组 | `send_block_array(data, count)` | `recv_block_array(data, count)` |
| 算术 RSS | `send_rss_share(value, bw)` | `recv_rss_share(bw)` |
| 算术 RSS 数组 | `send_rss_share_array(data, count, bw)` | `recv_rss_share_array(data, count, bw)` |
| 布尔 RSS | `send_rss_xor_share(value, bw)` | `recv_rss_xor_share(bw)` |
| 布尔 RSS 数组 | `send_rss_xor_share_array(data, count, bw)` | `recv_rss_xor_share_array(data, count, bw)` |

`bw` 取 1～64。整数采用小端编码，按 1/2/4/8 字节分档，而不是紧密按位打包：
1～8 位为 1 字节，9～16 位为 2 字节，17～32 位为 4 字节，33～64 位为 8 字节。
发送和接收均截取有效低位。RSS 每项按 `first`、`second` 排列；block 保持 16 字节内容。

零长度数组允许空指针，且不传输数据；非法位宽仍会报错。
非空空指针和长度乘法溢出在传输前拒绝。类型化收发使用临时编码缓冲区，
因此内存需求与数组大小成正比；分配失败发生在网络传输之前。

本模块不添加逐消息头；双方必须约定类型、数量、位宽与调用顺序。
`send_*` 和 `recv_*` 在内部循环处理短写、短读及 `EINTR`，
一次完整传输使用同一个 I/O 截止时间，收到部分数据不会重置超时。

### 同时交换、同步与统计

大数组双向或环形交换使用：

~~~cpp
comm.exchange_bytes(sendTo, sendData, sendBytes, recvFrom, recvData, recvBytes);
comm.exchange_ge_array(sendTo, sendData, sendCount,
                       recvFrom, recvData, recvCount, bw);
~~~

两个方向通过非阻塞 socket 与 `poll` 同时推进，避免三方先发送后接收时因缓冲区满而互相等待。
`exchange_bytes` 要求收发缓冲区不重叠；`exchange_ge_array` 先编码发送数据，支持原地交换。
目标和来源均须为另外两方之一；同一方同时充当发送目标与接收来源是合法的。
同一上下文不支持调用方多线程并发使用。

- `peer.sync()` 要求同一条边的双方都调用。
- `comm.sync()` 是三方屏障：P0 收齐到达信号后发送释放信号。
- 两个类均提供 `bytesSent()`、`bytesReceived()`、
  `zeroBytesSent()`、`zeroBytesReceived()`。
- 每个 Peer 的统计对应它持有的连接；Comm 的统计是两个 Peer 的合计。
  包含该连接上的握手、种子、同步和业务数据，不包含 TCP/IP 报文头。
  失败且未加入上下文的临时连接不在最终 Peer 的累计中。
  可在构造及一次三方同步之后清零，单独测量后续业务。
- 失败的部分收发也按已经成功读写的字节累计；不改动旧 FSS 统计变量。

## 成对共享随机流

每条边由编号较小的一方通过 Linux `getrandom` 获取两个独立 128 位种子，
分别供算术、布尔用途。随机源不足或失败会报错，不回退到固定值或时间种子。

每方持有四个独立 PRNG：

| 访问器 | 对应关系 |
|---|---|
| `prevArithmeticPrng()` | 与上一方共享，生成算术分量 `x_i` |
| `nextArithmeticPrng()` | 与下一方共享，生成算术分量 `x_(i+1)` |
| `prevBooleanPrng()` | 与上一方共享，生成布尔分量 `x_i` |
| `nextBooleanPrng()` | 与下一方共享，生成布尔分量 `x_(i+1)` |

初始化只设置种子，不消费业务随机数。共享边两端需以相同用途、顺序和位宽调用随机份额函数。
输入持有方的随机数另行管理，不复用这些共享流。种子和份额不输出到日志。
连接上的种子分发采用明文 TCP，适用于受控实验环境和半诚实参与方假设；
角色握手不提供密码学认证或恶意安全保证。本轮没有实现 TLS、私有输入分发、
公开重构、秘密乘法、秘密 AND 或文件／内存通信后端。

## 三方初始化示例

下面代码保存为 `build/rss_comms_example.cpp`。三个进程分别持有本方的随机份额；
示例只打印连接统计，不打印份额或种子。

~~~cpp
#include <crypto/commom/comms-3PC.h>

#include <array>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv)
{
    try {
        if (argc < 2 || argc > 3) {
            std::cerr << "Usage: rss_comms_example <0|1|2> [base-port]\n";
            return 1;
        }
        const int id = std::stoi(argv[1]);
        const int base = argc == 3 ? std::stoi(argv[2]) : 43000;
        if (id < 0 || id > 2 || base < 1 || base > 65533) {
            throw std::invalid_argument("invalid party or base port");
        }
        const std::array<Endpoint3PC, 3> nodes{{
            {"127.0.0.2", static_cast<std::uint16_t>(base)},
            {"127.0.0.2", static_cast<std::uint16_t>(base + 1)},
            {"127.0.0.2", static_cast<std::uint16_t>(base + 2)}
        }};
        Comm3PC comm(static_cast<RssParty>(id), nodes);
        const RssShare arithmetic = randomShareRss(
            64, comm.prevArithmeticPrng(), comm.nextArithmeticPrng());
        const RssXorShare boolean = randomShareRssXor(
            64, comm.prevBooleanPrng(), comm.nextBooleanPrng());
        (void)arithmetic;
        (void)boolean;
        comm.sync();
        std::cout << "P" << id << " ready; sent=" << comm.bytesSent()
                  << " received=" << comm.bytesReceived() << '\n';
        // Do not print the seeds or share components.
        comm.sync();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
~~~

在项目根目录构建并编译示例：

~~~bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build -j4
c++ -std=gnu++17 -maes -march=native -fopenmp -pthread \
    -I. -Icrypto/cryptoTools build/rss_comms_example.cpp \
    build/crypto/libFSS.a build/crypto/cryptoTools/libcryptoTools.a \
    -o build/rss_comms_example
~~~

在同一 WSL 的三个终端分别执行：

~~~bash
/home/anfy/projects/ADSMPC-C/build/rss_comms_example 0
/home/anfy/projects/ADSMPC-C/build/rss_comms_example 1
/home/anfy/projects/ADSMPC-C/build/rss_comms_example 2
~~~

默认端口为 43000～43002，第二个参数可指定基准端口。
同一 WSL 的三方使用 `127.0.0.2`，不需要新增网卡地址或修改系统路由。
本次环境中，`ip route get 127.0.0.1 ipproto tcp` 指向 `loopback0` 转发接口，
而 `127.0.0.2` 指向 `lo`。默认源地址仍可能为 `127.0.0.1`，
因此通信实现也显式绑定本方回环源地址，保证请求和回包都走本地路径。
显式配置的 `127.0.0.1` 不会被自动改写；已有本机调用代码应将三项 IP 一并改为 `127.0.0.2`。
跨机器部署时，将示例地址数组中的三项 IP 改为对应机器地址；
三方配置必须相同，且端口需可互相访问。自动验收使用本机回环连接，
没有实际部署到三台独立机器。

## 验证与备份

本次回环路径修复验收：默认 Release 全项目构建通过，包含已有基准、CNN/BPGNN 和测试；
CTest 三项全部通过，通信测试不启用跟踪连续十轮通过，共覆盖六十次启动顺序。
新回归用例在旧实现上失败、在修复后通过；完整三进程计算每方核对 16,896 个结果。
文档示例已重新编译，并以 P2、P0、P1 顺序启动三个进程，全部正常退出。
头文件独立、重复及与旧通信头混合包含通过；构建目标与 BUILD_TESTING 控制方式保持不变。
此前已验证 BUILD_TESTING=OFF 时不生成两个 RSS 测试目标。

成功建连场景使用公开接口默认的 30 秒建连超时；故障测试另外设置 100/150 毫秒等短超时，
验证截止时间和清理逻辑。之前使用 `127.0.0.1` 的复测曾出现握手与建连偶发失败；
低干扰日志和独立 TCP 对照测试定位到本机转发路径，不能仅靠增加超时解决。
当前默认使用直接回环地址并绑定源地址；仍保留对单次建连及握手的有界重试，
错误信息标注参与方及初始化阶段。

`rss_comms_test` 自动启动、监控和回收子进程，动态分配测试端口，注册为 CTest `rss_comms`。
它覆盖六种启动顺序、全部位宽和数据类型、字节编码、空数组及非法参数、
每方每种启动顺序 768 组算术/布尔随机份额、8 MiB 双向环形交换、4 MiB 同一对端双向交换、
短读短写、EINTR、提前断开、SIGPIPE、总超时、错误/重复握手、启动期无应答重试及资源清理。
另有回环源地址回归：P2 配置 `127.0.0.3`、P0 配置 `127.0.0.2`，
接收端验证实际来源确为 P2 的地址，防止误绑定目标地址或继续使用系统默认源地址。
测试使用显式失败检查，在 Release / NDEBUG 下有效。
`BUILD_TESTING=OFF` 时不创建测试目标。

~~~bash
ctest --test-dir /home/anfy/projects/ADSMPC-C/build --output-on-failure
ctest --test-dir /home/anfy/projects/ADSMPC-C/build -R '^rss_comms$' --output-on-failure
~~~

本次修复前的完整源码备份（不含构建产物）：
`/home/anfy/projects/.codex-backups/ADSMPC-C-comms3pc-fix-uif8lrbc/source`

修复原因、修改清单、前后回归结果及十轮复测日志见
`build/comms3pc-fix-20261115-182411/report.json` 和同目录 `REPORT.txt`。
其中 `tcp-routes.txt`、`plain-tcp-ab.json` 记录当前环境的路由与普通 TCP 对照结果。

首次实现时的历史清单仍保留为 `build/comms3pc-manifest.json`，
当时备份为 `/home/anfy/projects/.codex-backups/ADSMPC-C-comms3pc-andry7hf/source`。
