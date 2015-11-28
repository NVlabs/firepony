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
#include "high_quality_window.h"
#include "gather.h"

#include "../primitives/util.h"

#include "../../table_formatter.h"

#include <thrust/functional.h>

namespace firepony {

// updates a set of covariate tables for a given event
template <typename packer_table, typename... packer_chain>
static void covariate_gatherer(const uint32 tid,
                               firepony_context<host>& ctx, const alignment_batch_device<host>& batch, const uint32 cigar_event_index, packer_table& first, packer_chain&... next)
{
    auto& table = first.table;

    covariate_key_set keys;
    bool key_valid;

    key_valid = generate_covariate_event_key<host, typename packer_table::packer>(keys, ctx, batch, cigar_event_index);

    if (key_valid)
    {
        const uint32 read_index = ctx.cigar.cigar_event_read_index[cigar_event_index];
        const auto idx = batch.crq_index(read_index);
        const auto read_bp_offset = ctx.cigar.cigar_event_read_coordinates[cigar_event_index];

        auto& val_M = table.value(tid, keys.M);
        val_M.observations++;
        val_M.mismatches += ctx.fractional_error.snp_errors[idx.qual_start + read_bp_offset];

        auto& val_I = table.value(tid, keys.I);
        val_I.observations++;
        val_I.mismatches += ctx.fractional_error.insertion_errors[idx.qual_start + read_bp_offset];

        auto& val_D = table.value(tid, keys.D);
        val_D.observations++;
        val_D.mismatches += ctx.fractional_error.deletion_errors[idx.qual_start + read_bp_offset];
    }

    // recurse for next argument
    covariate_gatherer<packer_chain...>(tid, ctx, batch, cigar_event_index, next...);
}

// terminator for covariate_gatherer
template <typename... packer_chain>
static void covariate_gatherer(const uint32, firepony_context<host>&, const alignment_batch_device<host>&, const uint32)
{ }

struct covariate_gather_worker : public lambda<host>
{
    LAMBDA_INHERIT_SYS(host);

    void operator() (const uint32 tid)
    {
        auto& cv = ctx.covariates;

        auto t_qual = make_packer_table<covariate_packer_quality_score<host>>(cv.quality);
        auto t_cycle = make_packer_table<covariate_packer_cycle_illumina<host>>(cv.cycle);
        auto t_context = make_packer_table<covariate_packer_context<host>>(cv.context);

        const uint32 num_threads = command_line_options.cpu_threads;
        constexpr uint32 grain_size = 1000;

        for(uint32 start = grain_size * tid;
            start < ctx.cigar.cigar_event_read_coordinates.size();
            start += grain_size * num_threads)
        {
            const uint32 stop = std::min(start + grain_size, ctx.cigar.cigar_event_read_coordinates.size());

            for(uint32 cigar_event_index = start; cigar_event_index < stop; cigar_event_index++)
            {
                covariate_gatherer(tid, ctx, batch, cigar_event_index,
                                   t_qual, t_cycle, t_context);
            }
        }
    }
};

template <>
void gather_covariates<host>(firepony_context<host>& context, const alignment_batch<host>& batch)
{
    auto& cv = context.covariates;

    // compute the "high quality" windows (i.e., clip off low quality ends from each read)
    cv.high_quality_window.resize(batch.device.num_reads);
    parallel<host>::for_each(context.active_read_list.begin(),
                             context.active_read_list.end(),
                             compute_high_quality_windows<host>(context, batch.device));

    cv.quality.init();
    cv.cycle.init();
    cv.context.init();

    parallel<host>::for_each(command_line_options.cpu_threads,
                             covariate_gather_worker(context, batch.device));
}

} // namespace firepony
