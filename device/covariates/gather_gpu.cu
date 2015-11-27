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

#include "../primitives/util.h"

#include "../../table_formatter.h"

#include <thrust/functional.h>

namespace firepony {

// helper struct for keeping track of covariate packers together with output tables
template <typename _covariate_packer>
struct covariate_packer_table
{
    typedef _covariate_packer packer;
    covariate_observation_table<cuda> table;

    covariate_packer_table(covariate_observation_table<cuda> table)
        : table(table)
    { }
};

// updates a set of covariate tables for a given event
template <typename packer_table, typename... packer_chain>
static __device__ void covariate_gatherer(firepony_context<cuda>& ctx, const alignment_batch_device<cuda>& batch, const uint32 cigar_event_index, packer_table& first, packer_chain&... next)
{
    auto& table = first.table;

    covariate_key_set keys;
    bool key_valid;

    key_valid = generate_covariate_event_key<cuda, typename packer_table::packer>(keys, ctx, batch, cigar_event_index);

    if (key_valid)
    {
        const uint32 read_index = ctx.cigar.cigar_event_read_index[cigar_event_index];
        const auto idx = batch.crq_index(read_index);
        const auto read_bp_offset = ctx.cigar.cigar_event_read_coordinates[cigar_event_index];

        table.keys  [cigar_event_index * 3 + 0] = keys.M;
        table.values[cigar_event_index * 3 + 0].observations = 1;
        table.values[cigar_event_index * 3 + 0].mismatches = ctx.fractional_error.snp_errors[idx.qual_start + read_bp_offset];

        table.keys  [cigar_event_index * 3 + 1] = keys.I;
        table.values[cigar_event_index * 3 + 1].observations = 1;
        table.values[cigar_event_index * 3 + 1].mismatches = ctx.fractional_error.insertion_errors[idx.qual_start + read_bp_offset];

        table.keys  [cigar_event_index * 3 + 2] = keys.D;
        table.values[cigar_event_index * 3 + 2].observations = 1;
        table.values[cigar_event_index * 3 + 2].mismatches = ctx.fractional_error.deletion_errors[idx.qual_start + read_bp_offset];
    }

    // recurse for next argument
    covariate_gatherer<packer_chain...>(ctx, batch, cigar_event_index, next...);
}

// terminator for covariate_gatherer
template <typename... packer_chain>
static __device__ void covariate_gatherer(firepony_context<cuda>& ctx, const alignment_batch_device<cuda>& batch, const uint32 cigar_event_index)
{ }

template <typename packer_table>
struct covariate_gatherer_single
{
    firepony_context<cuda> ctx;
    const alignment_batch_device<cuda> batch;
    packer_table packer;

    covariate_gatherer_single(firepony_context<cuda> ctx,
                              const alignment_batch_device<cuda> batch,
                              packer_table& packer)
        : ctx(ctx),
          batch(batch),
          packer(packer)
    { }

    __device__ void operator() (const uint32 cigar_event_index)
    {
        covariate_gatherer<packer_table>(ctx, batch, cigar_event_index, packer);
    }
};

// functor that determines if a key is valid
// note: operator() returns a uint32 to allow for composition of this functor with reduction operators
template <typename covariate_packer>
struct is_key_valid : public thrust::unary_function<covariate_key, uint32>
{
    __device__ uint32 operator() (const covariate_key key)
    {
        constexpr bool sparse = covariate_packer::chain::is_sparse(covariate_packer::TargetCovariate);

        if (key == covariate_key(-1) ||
            (sparse && covariate_packer::decode(key, covariate_packer::TargetCovariate) == covariate_packer::chain::invalid_key(covariate_packer::TargetCovariate)))
        {
            return 0;
        } else {
            return 1;
        }
    }
};

template <typename covariate_packer, typename Tuple>
struct is_key_value_pair_valid : public thrust::unary_function<Tuple, bool>
{
    __device__ uint32 operator() (const Tuple& T)
    {
        covariate_key key = thrust::get<0>(T);
        return is_key_valid<covariate_packer>()(key) ? true : false;
    }
};

// processes a batch of reads and updates covariate table data for a given table
// uses the filer/sort/pack algorithm
template <typename covariate_packer>
static void build_covariates_table(covariate_observation_table<cuda>& table, firepony_context<cuda>& context, const alignment_batch<cuda>& batch)
{
    auto& cv = context.covariates;
    auto& scratch_table = cv.scratch_table_space;

    scoped_allocation<cuda, covariate_observation_value> temp_values;
    scoped_allocation<cuda, covariate_key> temp_keys;

    timer<cuda> covariates_gather, covariates_filter, covariates_sort, covariates_pack;

    covariates_gather.start();

    // set up a scratch table space with enough room for 3 keys per cigar event
    scratch_table.resize(context.cigar.cigar_events.size() * 3);

    // mark all keys as invalid
    thrust::fill(lift::backend_policy<cuda>::execution_policy(),
                 scratch_table.keys.begin(),
                 scratch_table.keys.end(),
                 covariate_key(-1));

    // generate keys into the scratch table
    auto packer = covariate_packer_table<covariate_packer>(scratch_table);
    parallel<cuda>::for_each(thrust::make_counting_iterator(0u),
                               thrust::make_counting_iterator(0u) + context.cigar.cigar_event_read_coordinates.size(),
                               covariate_gatherer_single<decltype(packer)>(context, batch.device, packer));

    covariates_gather.stop();

    covariates_filter.start();

    // count valid keys
    uint32 valid_keys = parallel<cuda>::sum(thrust::make_transform_iterator(scratch_table.keys.begin(),
                                                                              is_key_valid<covariate_packer>()),
                                              scratch_table.keys.size(),
                                              context.temp_storage);

    if (valid_keys)
    {
        // concatenate valid keys to the end of the output table
        size_t off = table.size();
        table.resize(table.size() + valid_keys);

        parallel<cuda>::copy_if(thrust::make_zip_iterator(thrust::make_tuple(scratch_table.keys.begin(),
                                                                               scratch_table.values.begin())),
                                  scratch_table.keys.size(),
                                  thrust::make_zip_iterator(thrust::make_tuple(table.keys.begin() + off,
                                                                               table.values.begin() + off)),
                                  is_key_value_pair_valid<covariate_packer,
                                                          thrust::tuple<const covariate_key&, const typename covariate_observation_table<cuda>::value_type&> >(),
                                  context.temp_storage);
    }

    covariates_filter.stop();

    if (valid_keys)
    {
        // sort and reduce the table by key
        covariates_sort.start();
        table.sort(temp_keys, temp_values, context.temp_storage, covariate_packer::chain::bits_used);
        covariates_sort.stop();

        covariates_pack.start();
        table.pack(temp_keys, temp_values, context.temp_storage);
        covariates_pack.stop();
    }

    parallel<cuda>::synchronize();

    context.stats.covariates_gather.add(covariates_gather);
    context.stats.covariates_filter.add(covariates_filter);

    if (valid_keys)
    {
        context.stats.covariates_sort.add(covariates_sort);
        context.stats.covariates_pack.add(covariates_pack);
    }
}

struct compute_high_quality_windows : public lambda<cuda>
{
    LAMBDA_INHERIT_SYS(cuda);

    enum {
        // any bases with q <= LOW_QUAL_TAIL are considered low quality
        LOW_QUAL_TAIL = 2
    };

    LIFT_DEVICE void operator() (const uint32 read_index)
    {
        const CRQ_index idx = batch.crq_index(read_index);

        const auto& window = ctx.cigar.read_window_clipped[read_index];
        auto& low_qual_window = ctx.covariates.high_quality_window[read_index];

        low_qual_window = window;

        while(batch.qualities[idx.qual_start + low_qual_window.x] <= LOW_QUAL_TAIL &&
                low_qual_window.x < low_qual_window.y)
        {
            low_qual_window.x++;
        }

        while(batch.qualities[idx.qual_start + low_qual_window.y] <= LOW_QUAL_TAIL &&
                low_qual_window.y > low_qual_window.x)
        {
            low_qual_window.y--;
        }
    }
};

template <>
void gather_covariates<cuda>(firepony_context<cuda>& context, const alignment_batch<cuda>& batch)
{
    auto& cv = context.covariates;

    // compute the "high quality" windows (i.e., clip off low quality ends from each read)
    cv.high_quality_window.resize(batch.device.num_reads);
    parallel<cuda>::for_each(context.active_read_list.begin(),
                               context.active_read_list.end(),
                               compute_high_quality_windows(context, batch.device));

    build_covariates_table<covariate_packer_quality_score<cuda> >(cv.quality, context, batch);
    build_covariates_table<covariate_packer_cycle_illumina<cuda> >(cv.cycle, context, batch);
    build_covariates_table<covariate_packer_context<cuda> >(cv.context, context, batch);
}

} // namespace firepony
