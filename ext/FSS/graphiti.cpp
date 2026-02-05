#include <FSS/api.h>
#include <FSS/utils.h>
#include <FSS/comms.h>

// Shuffle 协议实现：按要求交换序列首尾
void Shuffle(int size, GroupElement* arr) {
    if (size < 2) return;
    GroupElement tmp = arr[0];
    arr[0] = arr[size - 1];
    arr[size - 1] = tmp;
    
    // 模拟在线同步开销
    if (FSSConfig::party != DEALER) {
        FSSConfig::peer->sync();
    }
}

// Graphiti 线性前缀和扫描：实现 Propagate 和 Gather 的在线逻辑
void PrefixSumScan(int N, GroupElement* arr) {
    for (int i = 1; i < N; i++) {
        arr[i] = arr[i] + arr[i - 1];
        mod(arr[i], FSSConfig::bitlength);
    }
}