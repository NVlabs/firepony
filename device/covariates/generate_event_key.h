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

namespace firepony {

// generate a single event key using the given covariate packer
template <target_system system, typename covariate_packer>
LIFT_HOST_DEVICE static bool generate_covariate_event_key(covariate_key_set& keys,
                                                          firepony_context<system>& ctx, const alignment_batch_device<system>& batch,
                                                          const uint32 cigar_event_index)
{
    const uint32 read_index = ctx.cigar.cigar_event_read_index[cigar_event_index];

    if (read_index == uint32(-1))
    {
        return false;
    }

    const auto idx = batch.crq_index(read_index);
    const auto read_bp_offset = ctx.cigar.cigar_event_read_coordinates[cigar_event_index];

    if (read_bp_offset == uint16(-1))
    {
        return false;
    }

    if (read_bp_offset < ctx.cigar.read_window_clipped[read_index].x ||
        read_bp_offset > ctx.cigar.read_window_clipped[read_index].y)
    {
        return false;
    }

    if (ctx.active_location_list[idx.read_start + read_bp_offset] == 0)
    {
        return false;
    }

    if (ctx.cigar.cigar_events[cigar_event_index] == cigar_event::S)
    {
        return false;
    }

    keys = covariate_packer::chain::encode(ctx, batch, read_index, read_bp_offset, cigar_event_index, covariate_key_set{0, 0, 0});
    return true;
}

} // namespace firepony
