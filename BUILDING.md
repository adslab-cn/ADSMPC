# ADSMPC-C 构建与头文件更新说明

本次更新针对重组后的目录结构，保留已有 Config 命名空间和 commom 目录名。
构建维护涵盖 include 路径、头文件依赖、分层 CMake、bitpack 链接修复、密钥/释放函数分层，以及 RSS 基础份额操作和三方 TCP 通信；未改动既有 FSS 协议算法。
最新密钥布局、接口迁移和释放问题记录见 [KEY_LAYOUT.md](KEY_LAYOUT.md)。
三方 RSS TCP 接口、共享 PRNG、初始化示例与异常处理见 [RSS_COMMS.md](RSS_COMMS.md)。

## WSL 构建

在 Ubuntu / WSL 终端执行：

~~~bash
cd /home/anfy/projects/ADSMPC-C
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
ctest --test-dir build --output-on-failure
~~~

依赖为 C++17 编译器、CMake >= 3.17、Eigen3、OpenMP 和 Threads。当前验证环境为 Ubuntu 22.04 / WSL2、g++ 9.5.0。
沿用 AES-NI 和 `-march=native`，生成的二进制面向当前 x86 主机。
保持 GNU C++17 扩展，因为现有代码包含变长数组。

`build/compile_commands.json` 由 CMake 自动生成，编辑器可直接使用该文件获得实际 include 路径及编译参数。

只构建库：

~~~bash
cmake -S . -B build-libs -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
cmake --build build-libs -j4
~~~

`BUILD_TESTING=OFF` 同时关闭 bitpack 测试、基准程序和多方示例。默认开启时，编译全部现有程序，CTest 运行 bitpack、rss_group_element 和 rss_comms 三项测试，不自动启动多方训练或长时间基准。

## CMake 分层

当前维护 16 份 CMakeLists.txt；密钥拆分新增了 protocol/util 构建入口：

| 层级 | 职责 |
|---|---|
| 根目录 | C++17、Eigen3/OpenMP/Threads、功能开关、公共头文件入口 |
| crypto | 组织密码学模块，创建 FSS 静态库 |
| crypto/commom | 将配置、原有通信、三方 RSS TCP 通信、PRNG、统计和工具实现加入 FSS |
| crypto/primitives | 管理 ASS/RSS 头文件和 FSS/OT 子目录 |
| crypto/primitives/FSS | 将 dpf.cpp、dcf.cpp、原语 freekey.cpp 加入 FSS |
| crypto/primitives/OT | 仅在显式开启且存在外部 SCI 依赖时创建 SCI-OTPrimitive |
| crypto/protocol | 显式列出 18 个协议实现，并管理 util 子目录 |
| crypto/protocol/util | 管理上层协议密钥声明，将协议 freekey.cpp 加入 FSS |
| crypto/cryptoTools | 创建 cryptoTools 并管理 bitpack 子目录 |
| crypto/cryptoTools/bitpack | 创建 bitpack 和受 BUILD_TESTING 控制的测试 |
| network | 创建 sytorch，传播 FSS 与 Eigen 等依赖 |
| network/src | 管理 random.cpp、后端及可选 SecFloat 实现 |
| network/src/backend | 管理三个现有后端实现 |
| tests | 组织基准与多方示例 |
| tests/bench | 保留全部 9 个基准目标名称，增加 RSS 正确性测试 |
| tests/e2e | 保留 CNN、BPGNN 目标名称，增加自动三进程 RSS 通信测试 |

FSS 仍是一个静态库，各目录只维护其 source 列表，避免重复编译和循环静态库链接。
纯头文件目录由所属模块管理。comms-3PC.cpp/.h 现已实现 TCP 三方通信并纳入 FSS；protocol/util 包含协议密钥和释放实现。

产物位于：

- 密码学库：`build/crypto/libFSS.a`
- 网络库：`build/network/libsytorch.a`
- 基准程序：`build/tests/bench/`
- 多方示例：`build/tests/e2e/CNN`、`build/tests/e2e/BPGNN`
- bitpack 测试：`build/crypto/cryptoTools/bitpack/bitpack_test`

## include 规则与源码调整

项目自有跨目录 include 使用项目根目录相对路径，例如：

~~~cpp
#include <crypto/commom/config.h>
#include <crypto/primitives/FSS/dpf.h>
#include <crypto/primitives/ASS/group_element.h>
#include <crypto/protocol/api.h>
#include <network/include/module.h>
~~~

- 清除了源码及遗留注释中的旧 `<FSS/...>` 引用；Config 命名空间保持一致。
- 网络层与密码学层的 utils.h，以及 ASS/RSS 两个 group_element.h，使用各自明确路径。
- 正确的同目录双引号引用保留。系统 `<assert.h>` 与 `<crypto/commom/assert.h>` 分开，不把所有叶子目录加入搜索路径。
- cryptoTools、bitpack 和 GSL 保持第三方公开 include 前缀，外部 Eigen 统一为 `<Eigen/Dense>`。
- RSS 头加入包含保护和 ASS 类型依赖；ASS 显式包含 `<utility>`；graph.h 显式包含 `<algorithm>`；prng.cpp 包含自己的声明头。
- 修复 bitpack::mod 的链接问题：移除仅放在 .cpp 定义上的 inline，使其与公开头声明一致并导出符号，函数体保持不变。

## RSS 基础份额接口

`crypto/primitives/RSS/group_element.h` 保持仅头文件实现，复用 ASS 的 `GroupElement = uint64_t` 和 `mod`。
保留已有六个函数的签名，新增独立的 `RssXorShare` 与十二个辅助接口。
`RssShare` 表示模 `2^bw` 的算术分享；`RssXorShare` 表示低 `bw` 位的按位 XOR 分享，两者不隐式转换。

两种类型均约定：`P_i` 持有 `(x_i, x_(i+1))`，下标对 3 取模，字段为 `first`、`second`。
`RssParty::P0/P1/P2` 是独立的三方编号，不自动对应 FSS 的 DEALER/SERVER/CLIENT 角色。

| 接口 | 参数 | 返回及含义 |
|---|---|---|
| `splitShareRss` / `splitShareRssXor` | `(value, bw, ownerPrng)` | 三方份额数组，分别对应 P0/P1/P2 |
| `rssFromPublic` / `rssFromPublicXor` | `(value, party, bw)` | 将公开值按 `(value, 0, 0)` 编码，返回指定方的份额 |
| `rssAdd` / `rssSub` | `(x, y, bw)` | 算术份额的加法、减法 |
| `rssMulPublic` | `(x, scalar, bw)` | 算术份额乘公开环元素 |
| `rssNeg` | `(x, bw)` | 算术份额取负 |
| `rssAddPublic` / `rssSubPublic` | `(x, value, party, bw)` | 算术秘密加、减公开值，公开值只注入 x0 的两个副本 |
| `rssXor` | `(x, y, bw)` | 布尔份额按位 XOR |
| `rssXorPublic` | `(x, value, party, bw)` | 布尔份额与公开值 XOR |
| `rssAndPublic` | `(x, mask, bw)` | 布尔份额与公开掩码按位 AND |
| `rssNot` | `(x, party, bw)` | 与低 bw 位全一的公开掩码 XOR |
| `rssCombine` / `rssCombineXor` | `(local, missing, bw)` | 本地重构为一个环元素或位向量；missing 必须为 `x_(i+2)` |
| `randomShareRss` / `randomShareRssXor` | `(bw, prevPrng, nextPrng)` | 未知随机秘密的本地方份额 |

所有接口要求 `1 <= bw <= 64`，参与方参数只能取 `P0/P1/P2`。
非法参数抛出 `std::invalid_argument`；随机接口在读取 PRNG 之前检查位宽。
输出分量全部归一化到指定宽度，允许输入包含超宽位；64 位算术依靠无符号回绕。
这些检查在 Release / `NDEBUG` 下同样有效。

输入拆分只供知道明文的输入持有方调用，并要求调用方安全初始化 `ownerPrng`。
返回的三项必须分别分发给相应参与方；本层不负责网络发送。
两个 `rssCombine` 接口只进行本地计算，不负责获取缺失分量或验证其他参与方提供的数据。

### 成对共享的 PRNG

对 `P_i`，`prevPrng` 与上一方共享，生成 `first = x_i`；
`nextPrng` 与下一方共享，生成 `second = x_(i+1)`：

| 参与方 | prevPrng 的共享边 | nextPrng 的共享边 |
|---|---|---|
| P0 | P2–P0（x0） | P0–P1（x1） |
| P1 | P0–P1（x1） | P1–P2（x2） |
| P2 | P1–P2（x2） | P2–P0（x0） |

三条边使用独立随机流。每条边两端持有独立的、同种子且同步消费的 PRNG 实例；
相应调用必须对齐位宽、次序和用途。算术、布尔随机份额以及输入拆分使用独立流。
本层不交换种子、不自动同步，也不复用 FSS 的全局 PRNG。
该接口生成未知随机值的分享，与 `splitShareCommonPRNG(value, bw)` 的指定值拆分语义不同。

RSS 头文件提供本地份额操作，三方 TCP 通信和共享 PRNG 初始化见 [RSS_COMMS.md](RSS_COMMS.md)。
私有输入分发、公开重构、秘密乘法、秘密 AND、算术 MSB 提取、秘密幂运算、截断和算术/布尔转换仍需协议层实现。
ASS 的 `pow`、`msb` 仍用于标量，
不能通过对算术份额的两个分量分别调用来得到秘密值的对应结果。

### RSS 基础份额验收记录（通信模块加入前）

- 默认全项目 Release 构建通过，包括已有库、九个基准程序、CNN/BPGNN 和新增测试。
- CTest 的 `bitpack`、`rss_group_element` 共 2 项通过。
- RSS 测试覆盖全部 1～64 位宽、5,184 组边界/随机输入、2,560 组随机份额样本和 1,414,868 项显式检查。
- 使用六个成对同种子的端点 PRNG，并以独立参考流验证分量来源；连续调用跨越 PRNG 缓冲区边界。
- 非法位宽、非法参与方及异常调用不消耗随机数的检查通过；测试不依赖 `assert`。
- 单独包含、重复包含、与 ASS/FSS 头混合包含、六个原有函数签名及算术份额布局检查通过。
- `BUILD_TESTING=OFF` 时不生成 RSS 测试目标，CTest 列表为空。

单独运行 RSS 测试：

~~~bash
ctest --test-dir /home/anfy/projects/ADSMPC-C/build -R '^rss_group_element$' --output-on-failure
~~~

验收记录位于 `build/rss-header-checks.json`、`build/rss-config-checks.json`、
`build/rss-configure.log`、`build/rss-build.log`、`build/rss-ctest.log` 和 `build/rss-manifest.json`。
修改清单：RSS 头文件、`tests/bench/CMakeLists.txt`、新增 `tests/bench/rss_group_element_test.cpp`，
以及本构建说明。ASS/FSS 源码、全局 PRNG 与现有协议实现保持原样。

修改前的完整源码备份（不含构建产物）：
`/home/anfy/projects/.codex-backups/ADSMPC-C-rss-i6zhf4_x/source`

## 三方 RSS 通信构建

三方 RSS 同一 WSL 部署请使用默认的 `127.0.0.2`，并为三方配置不同端口。
通信实现会绑定本方回环源地址，避免当前环境中 `127.0.0.1` 的转发路径影响连接。
跨机器部署继续显式配置各机器的 IPv4 地址；详见 `RSS_COMMS.md`。

公共模块将 comms-3PC.cpp 编入 FSS。rss_comms_test 链接 FSS，由 CTest 自动启动并回收子进程；
它不启动 CNN/BPGNN 训练。使用说明与三方初始化示例见 [RSS_COMMS.md](RSS_COMMS.md)。

~~~bash
cmake --build /home/anfy/projects/ADSMPC-C/build --target rss_comms_test -j4
ctest --test-dir /home/anfy/projects/ADSMPC-C/build -R '^rss_comms$' --output-on-failure
~~~

默认建连与 I/O 超时均为 30 秒，可分别配置。通信测试覆盖全部六种启动顺序、
1～64 位宽、三方共享随机流、大数组交换和异常路径，使用短超时验证故障行为。
当前 TCP 实现适用于受控实验环境和半诚实参与方，不提供 TLS 或恶意安全保证。

本次本机通信修复通过全部三项 CTest，并在不启用跟踪的情况下连续通过十轮通信测试。
最新报告保存在 `build/comms3pc-fix-20261115-182411/report.json`；
首次实现的历史清单仍保留为 `build/comms3pc-manifest.json`。

## 外部依赖未包含的模块

| 可选模块 | 默认值 | 依赖 |
|---|---|---|
| ADSMPC_ENABLE_OT | OFF | 外部 SCI-utils CMake 目标，以及 SCI 的 utils/emp-tool.h |
| ADSMPC_ENABLE_SECFLOAT | OFF | 外部 SCI-FloatML CMake 目标，以及 library_float.h |

这些缺失引用已检查，但没有伪造头文件、删除相关实现或下载依赖。
OT 的两处旧相对路径改为 `<utils/emp-tool.h>`，由 SCI 的 include 根目录提供。

显式启用时，CMake 在配置阶段检查依赖并给出错误，不会把缺失目标默认为一个裸链接库名称。
外部依赖必须由上层工程先提供对应的 CMake 目标，并分别设置：

- `ADSMPC_SCI_INCLUDE_DIR`：其下包含 `utils/emp-tool.h` 的兼容 SCI 目录。
- `ADSMPC_SECFLOAT_INCLUDE_DIR`：包含 `library_float.h` 的兼容 SecFloat 目录。

仅把功能开关设为 ON 并不能自动安装依赖；上述两个可选模块尚未在本仓库验证编译。

## 此前 include/CMake 与密钥拆分记录

以下为 RSS 补齐之前的验收记录，数量和报告均对应当时版本：

- 扫描 130 个源码/头文件（含无扩展名的 GSL 文件）中的 511 条有效 include 指令：必需头文件路径全部可解析。
- 旧 FSS 头路径和旧 FSSConfig 标识符无残留。
- 52 个项目头文件分别单独包含、重复包含的语法检查通过，包括 Config、ASS、RSS。
- 默认 Release 构建通过：49 个翻译单元，包含全部库、9 个基准程序、CNN、BPGNN 和 bitpack_test。
- 现有 bitpack CTest 通过。
- BUILD_TESTING=OFF 的配置与目标清单检查通过；开启缺依赖 OT/SecFloat 时按预期配置失败。
- 对照备份确认密钥拆分保留了 59 个类型定义和 65 个释放函数体；先前构建维护的修改为 include、包含保护、bitpack 链接定义及失效注释清理。

现有 api.cpp 的计时统计存在一条 narrowing-conversion 编译警告，未在本次构建路径更新中修改其逻辑。
这次没有执行多方 CNN/BPGNN 训练，也未重新评估密码学协议性能。

构建目录中的验收记录：

- `build/key-split-include-audit.json`：密钥拆分时的 include 审计及三处可选依赖缺口。
- `build/key-split-checks.json`：密钥拆分时的 52 个头文件检查、类型/函数一致性、隔离和构建结果。
- `build/config-checks.json`：构建开关检查。
- `build/key-split-manifest.json`：本次密钥拆分的新增/修改/删除文件清单与备份路径。
- `build/key-split-build.log`：本次编译日志。
- `build/update-manifest.json`：先前 include/CMake 更新的历史清单。

密钥拆分前的完整源码备份：
`/home/anfy/projects/.codex-backups/ADSMPC-C-key-split-yvmos_2o/source`

`build/` 中的文件是本机生成的产物；移除构建目录会同时移除上述验收记录，但不影响源码或备份。
