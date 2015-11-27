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

namespace firepony {

typedef uint32 covariate_key;

// the value for each row of a covariate observation table
struct covariate_observation_value
{
    uint64 observations;
    float mismatches;
};

// the value for each row of a covariate empirical table
// this includes all the info in the covariate observation table plus computed values
struct covariate_empirical_value
{
    uint64 observations;
    double mismatches;
    double expected_errors;
    double estimated_quality;
    double empirical_quality;
};

// covariate table
// stores a list of key-value pairs, where the key is a covariate_key and the value is either covariate_observation_value or covariate_empirical_value
template <target_system system, typename covariate_value>
struct covariate_table
{
    typedef covariate_value value_type;

    persistent_allocation<system, covariate_key> keys;
    persistent_allocation<system, covariate_value> values;

    void resize(size_t size)
    {
        keys.resize(size);
        values.resize(size);
    }

    size_t size(void) const
    {
        assert(keys.size() == values.size());
        return keys.size();
    }

    template <target_system other_system>
    void copyfrom(covariate_table<other_system, covariate_value>& other)
    {
        keys.resize(other.keys.size());
        values.resize(other.values.size());

        thrust::copy(other.keys.t_begin(), other.keys.t_end(), keys.t_begin());
        thrust::copy(other.values.t_begin(), other.values.t_end(), values.t_begin());
    }

    // cross-device table concatenation
    template <target_system other_system>
    void concat(const lift::compute_device& my_device, const lift::compute_device& other_device, covariate_table<other_system, covariate_value>& other)
    {
        size_t off = size();

        keys.resize(keys.size() + other.keys.size());
        values.resize(values.size() + other.values.size());

        cross_device_copy(my_device, keys, off,
                          other_device, other.keys, 0,
                          other.keys.size());
        cross_device_copy(my_device, values, off,
                          other_device, other.values, 0,
                          other.values.size());
    }

    void sort(allocation<system, covariate_key>& temp_keys,
              allocation<system, covariate_value>& temp_values,
              allocation<system, uint8>& temp_storage,
              uint32 num_key_bits);

    void pack(allocation<system, covariate_key>& temp_keys,
              allocation<system, covariate_value>& temp_values,
              allocation<system, uint8>& temp_storage);

    struct view
    {
        pointer<system, uint32> keys;
        pointer<system, covariate_value> values;
    };

    operator view()
    {
        struct view v = {
            keys,
            values,
        };

        return v;
    }
};

template <target_system system> using covariate_observation_table = covariate_table<system, covariate_observation_value>;
template <target_system system> using covariate_empirical_table = covariate_table<system, covariate_empirical_value>;

template <target_system system> void covariate_observation_to_empirical_table(firepony_context<system>& context,
                                                                              const covariate_observation_table<system>& observation_table,
                                                                              covariate_empirical_table<system>& empirical_table);

} // namespace firepony
