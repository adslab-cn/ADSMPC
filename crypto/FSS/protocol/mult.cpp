// Authors: Kanav Gupta, Neha Jawalkar
// Copyright:
// 
// Copyright (c) 2024 Microsoft Research
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

/*
Authors: Deepak Kumaraswamy, Kanav Gupta
Copyright:
Copyright (c) 2022 Microsoft Research
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "../aux_parameter/comms.h"
#include "../primitives/dcf.h"
#include "mult.h"
#include <assert.h>
#include <utility>

std::pair<MultKey, MultKey> MultGen(GroupElement rin1, GroupElement rin2, GroupElement rout)
{
    
    MultKey k1, k2;
    // k1.Bin = Bin; k2.Bin = Bin;
    // k1.Bout = Bout; k2.Bout = Bout;

    GroupElement c  = rin1 * rin2 + rout;
    auto a_split = splitShare(rin1, 64);
    auto b_split = splitShare(rin2, 64);
    auto c_split = splitShare(c, 64);
    
    k1.a = (a_split.first);
    k1.b = (b_split.first);
    k1.c = (c_split.first);
    
    k2.a = (a_split.second);
    k2.b = (b_split.second);
    k2.c = (c_split.second);
    
    return std::make_pair(k1, k2);
}

GroupElement MultEval(int party, const MultKey &k, const GroupElement &l, const GroupElement &r)
{
    return party * (l * r) - l * k.b - r * k.a + k.c;
}



std::pair<SquareKey, SquareKey> keyGenSquare(GroupElement rin, GroupElement rout)
{
    SquareKey k1, k2;

    // rin 来自 Dealer 的 dummy 变量，如果是随机垃圾值，正好作为随机数 a
    // 如果是 0，虽然不安全，但逻辑也是对的。
    GroupElement a = rin; 
    GroupElement c = a * a; // c = a^2

    auto a_split = splitShare(a, 64);
    auto c_split = splitShare(c, 64);
    
    // 注意：我们将 [a] 存在 k.b 中，将 [c] 存在 k.c 中
    k1.b = (a_split.first);
    k1.c = (c_split.first);
    
    k2.b = (a_split.second);
    k2.c = (c_split.second);
    
    return std::make_pair(k1, k2);
}

// Beaver 协议的最后一步本地计算
// [x^2] = e^2 + 2*e*[a] + [c], 其中 e = x - a 是公开值
GroupElement evalSquare(int party, GroupElement e_public, const SquareKey &k)
{
    // k.b 是 [a], k.c 是 [a^2]
    // 结果 = party * e^2 + 2 * e * [a] + [c]
    return (party * e_public * e_public) + (2 * e_public * k.b) + k.c;
}