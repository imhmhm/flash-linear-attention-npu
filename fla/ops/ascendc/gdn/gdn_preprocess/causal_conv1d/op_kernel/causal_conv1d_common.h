/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file causal_conv1d_common.h
 * \brief CausalConv1d shared constants, ring-buffer helpers and buffer layouts.
 */

#ifndef CAUSAL_CONV1D_COMMON_H
#define CAUSAL_CONV1D_COMMON_H

#include "kernel_operator.h"

namespace NsCausalConv1dCommon {

constexpr int32_t MAX_WIDTH = 4;
// Upper bound of the runtime-width FN layout (widths 5..8 have no compile-time
// specialization and run on the generic path with a width-sized ring buffer).
constexpr int32_t MAX_WIDTH_RT = 8;
constexpr int32_t MAX_BLOCK_DIM = 4096;

constexpr int32_t RING_SLOTS = 5;

__aicore__ inline int32_t SlotCurr(int32_t t)
{
    return (t + 3) % RING_SLOTS;
}

__aicore__ inline int32_t SlotHist(int32_t t, int32_t i)
{
    return (t + 3 - i) % RING_SLOTS;
}

__aicore__ inline int32_t SlotPrefetch(int32_t t)
{
    return (t + 4) % RING_SLOTS;
}

struct CalcBufLayout {
    AscendC::LocalTensor<float> weightF;
    AscendC::LocalTensor<float> biasF;
    AscendC::LocalTensor<float> accF;
    AscendC::LocalTensor<float> tmpF;
    AscendC::LocalTensor<float> currF;

    __aicore__ inline CalcBufLayout() = default;

    __aicore__ static inline CalcBufLayout FromCalcBuf(AscendC::TBuf<AscendC::QuePosition::VECCALC> &calcBuf)
    {
        return FromCalcBuf(calcBuf, MAX_WIDTH, MAX_BLOCK_DIM);
    }

    // Runtime-width layout: the weight region is [0, weightSlots) at the given
    // slot pitch; bias/acc/tmp/curr follow at one slot each.
    __aicore__ static inline CalcBufLayout FromCalcBuf(AscendC::TBuf<AscendC::QuePosition::VECCALC> &calcBuf,
                                                       int32_t weightSlots, int32_t pitch)
    {
        CalcBufLayout layout;
        AscendC::LocalTensor<float> calc = calcBuf.template Get<float>();
        layout.weightF = calc;
        layout.biasF = calc[weightSlots * pitch];
        layout.accF = layout.biasF[pitch];
        layout.tmpF = layout.accF[pitch];
        layout.currF = layout.tmpF[pitch];
        return layout;
    }
};

}

#endif
