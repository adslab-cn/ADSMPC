# 密钥与释放函数分层

将原先 commom 下混合的 keypack / freekey 拆为两层。原有两个文件已删除，项目内所有引用迁移至新路径，不提供旧路径转发头。

## 目录与使用方式

| 层级 | 目录 | 文件 | 类型数 | 释放函数／重载数 |
|---|---|---|---:|---:|
| FSS 原语 | crypto/primitives/FSS | keypack.h、freekey.h、freekey.cpp | 14 | 20 |
| 上层协议 | crypto/protocol/util | keypack.h、freekey.h、freekey.cpp | 45 | 45 |

原语调用方包含：

~~~cpp
#include <crypto/primitives/FSS/keypack.h>
#include <crypto/primitives/FSS/freekey.h>
~~~

协议调用方包含：

~~~cpp
#include <crypto/protocol/util/keypack.h>
#include <crypto/protocol/util/freekey.h>
~~~

若同一文件直接使用两层释放接口，显式包含两层 freekey.h。两层头文件不再因包含 freekey 而自动引入通信类型；需要 Peer/Dealer 的调用方应显式包含 `crypto/commom/comms.h`。

类型仍位于原来的全局命名空间；Config 命名空间保持不变。
类型名、字段、字段顺序、构造函数、释放函数名称/参数及实现体均保持一致。
原来注释掉的 SplineOneKeyPack 随协议层迁移，仍然处于注释中。

## 依赖方向与链接

- 原语 keypack.h 依赖基础环元素和 cryptoTools，不包含上层协议密钥。
- 协议 keypack.h 包含原语 keypack.h，支持协议密钥嵌入原语密钥。
- 两层 freekey.h 仅包含自身 keypack.h、标准库头和函数声明。
- 释放函数实现放入对应的 freekey.cpp，移除定义上的 inline，编译进现有 FSS 静态库。调用方须链接 FSS；原有 sytorch 和测试目标已通过依赖传递链接它。
- freekey.cpp 在需要完整通信上下文时包含 comms.h；本次隔离公开头文件依赖，不重构通信模块的运行时职责。
- comms.h 显式包含两层 keypack.h，保持已有密钥序列化接口和数据格式。
- protocol/util 新增 CMakeLists.txt，两个 freekey.cpp 都只编译一次。现有库和可执行目标名称不变。

## 验证结果

- 逐项比对备份：59 个类型定义完全一致；65 个释放函数定义仅移除 inline，函数体与参数完全一致。
- 65 个函数声明与定义对应，两个新实现对象分别导出 20 和 45 个释放函数／重载。
- 52 个项目头文件独立包含、重复包含的语法检查通过，其中包含全部四个新头文件。
- 编译器依赖清单确认：原语 keypack.h 和 freekey.h 不传递包含 crypto/protocol/ 或 comms.h。
- 默认 Release 构建通过，包含全部库、9 个基准程序、CNN、BPGNN 和 bitpack_test，共 49 个翻译单元。
- 现有 CTest 的 1 项 bitpack 测试通过。本次未执行多方训练或重新进行密码学性能实验。
- 项目源码不存在旧两个头文件的 include 引用。

构建命令见 [BUILDING.md](BUILDING.md)。本次详细记录位于：

- `build/key-split-manifest.json`：完整类型与函数归属、include 迁移及修改清单。
- `build/key-split-checks.json`：定义一致性、头文件隔离、导出符号和构建/CTest 结果。
- `build/key-split-include-audit.json`：全项目 include 审计，仍仅有默认关闭的 SCI/SecFloat 外部依赖缺口。
- `build/key-split-build.log`：编译日志。

## 保留的既有释放问题

`freeDPFKeyPack(DPFKeyPack&)` 和 `freeDPFKeyPack(DPFETKeyPack&)` 在非内存缓冲区路径仍使用 `delete key.s`，而相关数组由 `new[]` 分配，存在原有的分配/释放不匹配问题。
此问题已记录，按本次约定不在结构拆分中修复；上述构建和 CTest 通过不表示这些释放路径已通过内存安全检查。
原有共享数组只释放一次、内存缓冲区中不归本方所有的区域不释放等规则均保持原状。

## 完整类型归属

### FSS 原语层（14 个）

| 类型 | 所属头文件 |
|---|---|
| `DCFKeyPack` | `crypto/primitives/FSS/keypack.h` |
| `DualDCFKeyPack` | `crypto/primitives/FSS/keypack.h` |
| `DPFKeyPack` | `crypto/primitives/FSS/keypack.h` |
| `DPFETKeyPack` | `crypto/primitives/FSS/keypack.h` |
| `GTDCFKeyPack` | `crypto/primitives/FSS/keypack.h` |
| `QuadDPFKeyPack` | `crypto/primitives/FSS/keypack.h` |
| `OctDPFKeyPack` | `crypto/primitives/FSS/keypack.h` |
| `VerDPFKeyPack` | `crypto/primitives/FSS/keypack.h` |
| `IFSSAuthShare` | `crypto/primitives/FSS/keypack.h` |
| `IFSSGlobalMACKey` | `crypto/primitives/FSS/keypack.h` |
| `IFSS_DPFKeyPack` | `crypto/primitives/FSS/keypack.h` |
| `IFSS_DCF_TwoDCFKeyPack` | `crypto/primitives/FSS/keypack.h` |
| `IFSS_DCFKeyPack` | `crypto/primitives/FSS/keypack.h` |
| `DIFKeyPack` | `crypto/primitives/FSS/keypack.h` |

### 上层协议（45 个）

| 类型 | 所属头文件 |
|---|---|
| `AddKey` | `crypto/protocol/util/keypack.h` |
| `MultKey` | `crypto/protocol/util/keypack.h` |
| `MatMulKey` | `crypto/protocol/util/keypack.h` |
| `MultKeyNew` | `crypto/protocol/util/keypack.h` |
| `Conv2DKey` | `crypto/protocol/util/keypack.h` |
| `Conv3DKey` | `crypto/protocol/util/keypack.h` |
| `TripleKeyPack` | `crypto/protocol/util/keypack.h` |
| `ScmpKeyPack` | `crypto/protocol/util/keypack.h` |
| `PublicICKeyPack` | `crypto/protocol/util/keypack.h` |
| `PublicDivKeyPack` | `crypto/protocol/util/keypack.h` |
| `SignedPublicDivKeyPack` | `crypto/protocol/util/keypack.h` |
| `ReluKeyPack` | `crypto/protocol/util/keypack.h` |
| `MaxpoolKeyPack` | `crypto/protocol/util/keypack.h` |
| `ARSKeyPack` | `crypto/protocol/util/keypack.h` |
| `ReluTruncateKeyPack` | `crypto/protocol/util/keypack.h` |
| `Relu2RoundKeyPack` | `crypto/protocol/util/keypack.h` |
| `SplineKeyPack` | `crypto/protocol/util/keypack.h` |
| `PrivateScaleKeyPack` | `crypto/protocol/util/keypack.h` |
| `SquareKey` | `crypto/protocol/util/keypack.h` |
| `TaylorSqKey` | `crypto/protocol/util/keypack.h` |
| `MICKeyPack` | `crypto/protocol/util/keypack.h` |
| `MSNZBKeyPack` | `crypto/protocol/util/keypack.h` |
| `BulkyLRSKeyPack` | `crypto/protocol/util/keypack.h` |
| `TaylorKeyPack` | `crypto/protocol/util/keypack.h` |
| `SelectKeyPack` | `crypto/protocol/util/keypack.h` |
| `MaxpoolDoubleKeyPack` | `crypto/protocol/util/keypack.h` |
| `BitwiseAndKeyPack` | `crypto/protocol/util/keypack.h` |
| `FixToFloatKeyPack` | `crypto/protocol/util/keypack.h` |
| `FloatToFixKeyPack` | `crypto/protocol/util/keypack.h` |
| `ReluExtendKeyPack` | `crypto/protocol/util/keypack.h` |
| `SignExtend2KeyPack` | `crypto/protocol/util/keypack.h` |
| `EdabitsPrTruncKeyPack` | `crypto/protocol/util/keypack.h` |
| `PubCmpKeyPack` | `crypto/protocol/util/keypack.h` |
| `ClipKeyPack` | `crypto/protocol/util/keypack.h` |
| `LUTKeyPack` | `crypto/protocol/util/keypack.h` |
| `F2BF16KeyPack` | `crypto/protocol/util/keypack.h` |
| `TruncateReduceKeyPack` | `crypto/protocol/util/keypack.h` |
| `LUTSSKeyPack` | `crypto/protocol/util/keypack.h` |
| `LUTDPFETKeyPack` | `crypto/protocol/util/keypack.h` |
| `SlothDreluKeyPack` | `crypto/protocol/util/keypack.h` |
| `WrapSSKeyPack` | `crypto/protocol/util/keypack.h` |
| `WrapDPFKeyPack` | `crypto/protocol/util/keypack.h` |
| `SlothLRSKeyPack` | `crypto/protocol/util/keypack.h` |
| `SlothTRKeyPack` | `crypto/protocol/util/keypack.h` |
| `SlothSignExtendKeyPack` | `crypto/protocol/util/keypack.h` |

## 修改前备份

`/home/anfy/projects/.codex-backups/ADSMPC-C-key-split-yvmos_2o/source`

备份不包含可重新生成的 build 目录。
