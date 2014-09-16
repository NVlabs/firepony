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

#include "bqsr_types.h"

struct baq_context
{
    // read and reference windows for HMM
    D_VectorU16_2 read_windows;
    D_VectorU32_2 reference_windows;

    // BAQ'ed qualities for each read, same size as each read
    D_VectorU8 qualities;
    // BAQ state vector, same size as each read
    D_VectorU32 state;

    // forward and backward HMM matrices
    // each read requires read_len * 6 * (bandWidth + 1)
    D_VectorF64 forward;
    D_VectorF64 backward;
    // index vector for forward/backward matrices
    D_VectorU32 matrix_index;

    // scaling factors
    D_VectorF64 scaling;
    // index vector for scaling factors
    D_VectorU32 scaling_index;

    struct view
    {
        D_VectorU16_2::plain_view_type read_windows;
        D_VectorU32_2::plain_view_type reference_windows;
        D_VectorU8::plain_view_type qualities;
        D_VectorU32::plain_view_type state;
        D_VectorF64::plain_view_type forward;
        D_VectorF64::plain_view_type backward;
        D_VectorU32::plain_view_type matrix_index;
        D_VectorF64::plain_view_type scaling;
        D_VectorU32::plain_view_type scaling_index;
    };

    operator view()
    {
        view v = {
                plain_view(read_windows),
                plain_view(reference_windows),
                plain_view(qualities),
                plain_view(state),
                plain_view(forward),
                plain_view(backward),
                plain_view(matrix_index),
                plain_view(scaling),
                plain_view(scaling_index),
        };

        return v;
    }
};

void baq_reads(bqsr_context *context, const reference_genome& reference, const BAM_alignment_batch_device& batch);
void debug_baq(bqsr_context *context, const reference_genome& genome, const BAM_alignment_batch_host& batch, int read_index);
