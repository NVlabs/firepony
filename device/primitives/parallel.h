/*
 * Copyright (c) 2012-14, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 *
 *
 *
 *
 *
 *
 *
 */

#pragma once

#define WAR_CUB_COPY_FLAGGED 1

#include "../../types.h"
#include "backends.h"

#include <thrust/scan.h>
#include <thrust/copy.h>
#include <thrust/sort.h>

#include <cub/device/device_reduce.cuh>
#include <cub/device/device_select.cuh>
#include <cub/device/device_radix_sort.cuh>

namespace firepony {

struct copy_if_flagged
{
    CUDA_HOST_DEVICE bool operator() (const uint8 val)
    {
        return bool(val);
    }
};

// thrust-based implementation of parallel primitives
template <target_system system>
struct parallel_thrust
{
    template <typename InputIterator, typename UnaryFunction>
    static inline InputIterator for_each(InputIterator first, InputIterator last, UnaryFunction f)
    {
        return thrust::for_each(firepony::backend_policy<system>::execution_policy(), first, last, f);
    }

    // shortcut to run for_each on a whole vector
    template <typename T, typename UnaryFunction>
    static inline typename vector<system, T>::iterator for_each(vector<system, T>& vector, UnaryFunction f)
    {
        return thrust::for_each(firepony::backend_policy<system>::par, vector.begin(), vector.end(), f);
    }

    template <typename InputIterator, typename OutputIterator, typename Predicate>
    static inline void inclusive_scan(InputIterator first,
                                      size_t len,
                                      OutputIterator result,
                                      Predicate op)
    {
        thrust::inclusive_scan(first, first + len, result, op);
    }

    template <typename InputIterator, typename OutputIterator, typename Predicate>
    static inline size_t copy_if(InputIterator first,
                                 size_t len,
                                 OutputIterator result,
                                 Predicate op,
                                 d_vector_u8<system>& temp_storage)
    {
        // use the fallback thrust version
        OutputIterator out_last;
        out_last = thrust::copy_if(first, first + len, result, op);
        return out_last - result;
    }

    template <typename InputIterator, typename FlagIterator, typename OutputIterator>
    static inline size_t copy_flagged(InputIterator first,
                                      size_t len,
                                      OutputIterator result,
                                      FlagIterator flags,
                                      d_vector_u8<system>& temp_storage)
    {
        OutputIterator out_last;
        out_last = thrust::copy_if(first, first + len, flags, result, copy_if_flagged());
        return out_last - result;
    }

    template <typename InputIterator>
    static inline int64 sum(InputIterator first,
                            size_t len,
                            d_vector_u8<system>& temp_storage)
    {
        return thrust::reduce(first, first + len, int64(0));
    }

    template <typename Key, typename Value>
    static inline void sort_by_key(d_vector<system, Key>& keys,
                                   d_vector<system, Value>& values,
                                   d_vector<system, Key>& temp_keys,
                                   d_vector<system, Value>& temp_values,
                                   d_vector_u8<system>& temp_storage,
                                   int num_key_bits = sizeof(Key) * 8)
    {
        thrust::sort_by_key(keys.begin(), keys.end(), values.begin());
    }
};

// default to thrust
template <target_system system>
struct parallel : public parallel_thrust<system>
{
    using parallel_thrust<system>::for_each;
    using parallel_thrust<system>::inclusive_scan;
    using parallel_thrust<system>::copy_if;
    using parallel_thrust<system>::copy_flagged;
    using parallel_thrust<system>::sum;
    using parallel_thrust<system>::sort_by_key;
};

// specialization for the cuda backend based on CUB primitives
template <>
struct parallel<cuda> : public parallel_thrust<cuda>
{
    using parallel_thrust<cuda>::for_each;

    template <typename InputIterator, typename OutputIterator, typename Predicate>
    static inline void inclusive_scan(InputIterator first,
                                      size_t len,
                                      OutputIterator result,
                                      Predicate op)
    {
        thrust::inclusive_scan(first, first + len, result, op);
    }

    template <typename InputIterator, typename OutputIterator, typename Predicate>
    static inline size_t copy_if(InputIterator first,
                                 size_t len,
                                 OutputIterator result,
                                 Predicate op,
                                 d_vector_u8<cuda>& temp_storage)
    {
        d_vector_i32<cuda> num_selected(1);

        // determine amount of temp storage required
        size_t temp_bytes = 0;
        cub::DeviceSelect::If(nullptr,
                temp_bytes,
                first,
                result,
                num_selected.begin(),
                len,
                op);

        // make sure we have enough temp storage
        temp_storage.resize(temp_bytes);

        cub::DeviceSelect::If(thrust::raw_pointer_cast(temp_storage.data()),
                temp_bytes,
                first,
                result,
                num_selected.begin(),
                len,
                op);

        return size_t(num_selected[0]);
    }

    // xxxnsubtil: cub::DeviceSelect::Flagged seems problematic
#if !WAR_CUB_COPY_FLAGGED
    template <typename InputIterator, typename FlagIterator, typename OutputIterator>
    static inline size_t copy_flagged_cub(InputIterator first,
                                          size_t len,
                                          OutputIterator result,
                                          FlagIterator flags,
                                          d_vector_u8<cuda>& temp_storage)
    {
        d_vector<cuda, size_t> num_selected(1);

        // determine amount of temp storage required
        size_t temp_bytes = 0;
        cub::DeviceSelect::Flagged(nullptr,
                temp_bytes,
                first,
                flags,
                result,
                num_selected.begin(),
                len);

        // make sure we have enough temp storage
        temp_storage.resize(temp_bytes);

        cub::DeviceSelect::Flagged(thrust::raw_pointer_cast(temp_storage.data()),
                temp_bytes,
                first,
                flags,
                result,
                num_selected.begin(),
                len);

        return size_t(num_selected[0]);
    }
#else
    using parallel_thrust<cuda>::copy_flagged;
#endif

    template <typename InputIterator>
    static inline int64 sum(InputIterator first,
                            size_t len,
                            d_vector_u8<cuda>& temp_storage)
    {
        d_vector_i64<cuda> result(1);

        size_t temp_bytes = 0;
        cub::DeviceReduce::Sum(nullptr,
                temp_bytes,
                first,
                result.begin(),
                len);

        temp_storage.resize(temp_bytes);

        cub::DeviceReduce::Sum(thrust::raw_pointer_cast(temp_storage.data()),
                temp_bytes,
                first,
                result.begin(),
                len);

        return int64(result[0]);
    }

    template <typename Key, typename Value>
    static inline void sort_by_key(d_vector<cuda, Key>& keys,
                                   d_vector<cuda, Value>& values,
                                   d_vector<cuda, Key>& temp_keys,
                                   d_vector<cuda, Value>& temp_values,
                                   d_vector_u8<cuda>& temp_storage,
                                   int num_key_bits = sizeof(Key) * 8)
    {
        const size_t len = keys.size();
        assert(keys.size() == values.size());

        temp_keys.resize(len);
        temp_values.resize(len);

        cub::DoubleBuffer<Key> d_keys(thrust::raw_pointer_cast(keys.data()),
                thrust::raw_pointer_cast(temp_keys.data()));
        cub::DoubleBuffer<Value> d_values(thrust::raw_pointer_cast(values.data()),
                thrust::raw_pointer_cast(temp_values.data()));

        size_t temp_storage_bytes = 0;
        cub::DeviceRadixSort::SortPairs(nullptr,
                temp_storage_bytes,
                d_keys,
                d_values,
                len,
                0,
                num_key_bits);

        temp_storage.resize(temp_storage_bytes);

        cub::DeviceRadixSort::SortPairs(thrust::raw_pointer_cast(temp_storage.data()),
                temp_storage_bytes,
                d_keys,
                d_values,
                len,
                0,
                num_key_bits);

        if (thrust::raw_pointer_cast(keys.data()) != d_keys.Current())
        {
            cudaMemcpy(thrust::raw_pointer_cast(keys.data()), d_keys.Current(), sizeof(Key) * len, cudaMemcpyDeviceToDevice);
        }

        if (thrust::raw_pointer_cast(values.data()) != d_values.Current())
        {
            cudaMemcpy(thrust::raw_pointer_cast(values.data()), d_values.Current(), sizeof(Value) * len, cudaMemcpyDeviceToDevice);
        }
    }
};

} // namespace firepony

