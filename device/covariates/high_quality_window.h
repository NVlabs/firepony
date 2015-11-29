/*
 * Firepony
 *
 * Copyright (c) 2014-2015, NVIDIA CORPORATION
 * Copyright (c) 2015, Nuno Subtil <subtil@gmail.com>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *    * Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *    * Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *    * Neither the name of the copyright holders nor the names of its
 *      contributors may be used to endorse or promote products derived from
 *      this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include "../../types.h"
#include "../alignment_data_device.h"
#include "../firepony_context.h"
#include "covariates.h"
#include "../expected_error.h"
#include "../empirical_quality.h"

#include "packer_context.h"
#include "packer_cycle_illumina.h"
#include "packer_quality_score.h"
#include "generate_event_key.h"

#include "../primitives/util.h"

#include "../../table_formatter.h"

#include <thrust/functional.h>

namespace firepony {

template <target_system system>
struct compute_high_quality_windows : public lambda<system>
{
    LAMBDA_INHERIT;

    // any bases with q <= LOW_QUAL_TAIL are considered low quality
    static constexpr uint32 LOW_QUAL_TAIL = 2;

    LIFT_HOST_DEVICE ushort2 compute(const uint32 read_index)
    {
        const CRQ_index idx = batch.crq_index(read_index);

        const auto& window = ctx.cigar.read_window_clipped[read_index];

        ushort2 high_qual_window;
        high_qual_window = window;

        while(batch.qualities[idx.qual_start + high_qual_window.x] <= LOW_QUAL_TAIL &&
                high_qual_window.x < high_qual_window.y)
        {
            high_qual_window.x++;
        }

        while(batch.qualities[idx.qual_start + high_qual_window.y] <= LOW_QUAL_TAIL &&
                high_qual_window.y > high_qual_window.x)
        {
            high_qual_window.y--;
        }

        return high_qual_window;
    }

    LIFT_HOST_DEVICE void operator() (const uint32 read_index)
    {
        auto& high_qual_window = ctx.covariates.high_quality_window[read_index];
        high_qual_window = compute(read_index);
    }
};

} // namespace firepony
