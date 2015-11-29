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

template <target_system system> void flush_covariates(firepony_context<system>& context)
{
    auto& cv = context.covariates;
    cv.quality.flush();
    cv.cycle.flush();
    cv.context.flush();
}
INSTANTIATE(flush_covariates);

template <target_system system> void postprocess_covariates(firepony_context<system>& context)
{
    auto& cv = context.covariates;

    // sort and pack all tables
    // this is required because we may have collected results from different devices
    scoped_allocation<system, covariate_observation_value> temp_values;
    scoped_allocation<system, covariate_key> temp_keys;

    cv.quality.sort(temp_keys, temp_values, context.temp_storage, covariate_packer_quality_score<system>::chain::bits_used);
    cv.quality.pack(temp_keys, temp_values, context.temp_storage);

    cv.cycle.sort(temp_keys, temp_values, context.temp_storage, covariate_packer_cycle_illumina<system>::chain::bits_used);
    cv.cycle.pack(temp_keys, temp_values, context.temp_storage);

    cv.context.sort(temp_keys, temp_values, context.temp_storage, covariate_packer_context<system>::chain::bits_used);
    cv.context.pack(temp_keys, temp_values, context.temp_storage);
}
INSTANTIATE(postprocess_covariates);

template <target_system system>
void output_covariates(firepony_context<system>& context)
{
    covariate_packer_quality_score<system>::dump_table(context, context.covariates.empirical_quality);

    table_formatter fmt("RecalTable2");
    fmt.add_column("ReadGroup", table_formatter::FMT_STRING);
    // for some odd reason, GATK thinks the quality score is a string...
    fmt.add_column("QualityScore", table_formatter::FMT_STRING);
    // CovariateValue has to be sent as string, since the actual data type will vary
    fmt.add_column("CovariateValue", table_formatter::FMT_STRING);
    fmt.add_column("CovariateName", table_formatter::FMT_STRING);
    fmt.add_column("EventType", table_formatter::FMT_CHAR);
    fmt.add_column("EmpiricalQuality", table_formatter::FMT_FLOAT_4);
    fmt.add_column("Observations", table_formatter::FMT_UINT64);
    fmt.add_column("Errors", table_formatter::FMT_FLOAT_2, table_formatter::ALIGNMENT_RIGHT, table_formatter::ALIGNMENT_LEFT);

    // preprocess table data to compute column widths
    covariate_packer_context<system>::dump_table(context, context.covariates.empirical_context, fmt);
    covariate_packer_cycle_illumina<system>::dump_table(context, context.covariates.empirical_cycle, fmt);
    fmt.end_table();

    // output table
    covariate_packer_context<system>::dump_table(context, context.covariates.empirical_context, fmt);
    covariate_packer_cycle_illumina<system>::dump_table(context, context.covariates.empirical_cycle, fmt);
    fmt.end_table();
}
INSTANTIATE(output_covariates);

template <target_system system, typename covariate_packer>
static void build_empirical_table(firepony_context<system>& context, covariate_empirical_table<system>& out, covariate_observation_table<system>& in)
{
    if (in.size() == 0)
    {
        // if we didn't gather any entries in the table, there's nothing to do
        return;
    }

    // convert the observation keys to empirical value keys
    covariate_observation_to_empirical_table(context, in, out);
    // compute the expected error for each entry
    compute_expected_error<system, covariate_packer>(context, out);
    // finally compute the empirical quality for this table
    compute_empirical_quality(context, out, true);
}

template <target_system system>
void compute_empirical_quality_scores(firepony_context<system>& context)
{
    auto& cv = context.covariates;

    build_empirical_table<system, covariate_packer_quality_score<system> >(context, cv.empirical_quality, cv.quality);
    build_empirical_table<system, covariate_packer_cycle_illumina<system> >(context, cv.empirical_cycle, cv.cycle);
    build_empirical_table<system, covariate_packer_context<system> >(context, cv.empirical_context, cv.context);
}
INSTANTIATE(compute_empirical_quality_scores);

} // namespace firepony
