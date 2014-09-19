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

#include <stdio.h>
#include <zlib/zlib.h>

#include <nvbio/basic/types.h>
#include <nvbio/basic/exceptions.h>

#include "util.h"
#include "bam_loader.h"

using namespace nvbio;

BAMfile::BAMfile(const char *fname)
    : eof(false)
{
    fp = gzopen(fname, "rb");
    if (fp == NULL)
        throw nvbio::runtime_error("could not open %s", fname);

    if (!init())
        throw nvbio::runtime_error("error parsing BAM file header");
}

BAMfile::~BAMfile()
{
    gzclose(fp);
}

bool BAMfile::readData(void *output, unsigned int len, int line)
{
    unsigned int ret;

    if (eof)
    {
        return false;
    }

    ret = gzread(fp, output, len);
    if (ret > 0)
    {
        return true;
    } else {
        // check for EOF separately; zlib will not always return Z_STREAM_END at EOF below
        if (gzeof(fp))
        {
            eof = true;
        } else {
            // ask zlib what happened and inform the user
            int err;
            const char *msg;

            msg = gzerror(fp, &err);
            // we're making the assumption that we never see Z_STREAM_END here
            assert(err != Z_STREAM_END);

            if (err == 0)
            {
                ret = gzread(fp, output, len);
                if (ret > 0)
                {
                    return true;
                } else {
                    throw nvbio::runtime_error("error processing BAM file (line %d): zlib error %d (%s) ret = %d", line, err, msg, ret);
                }
            }
        }

        return false;
    }
}

// note: tag must include the ':'
// start_ptr points to the start of the string to parse
// end_ptr points at the end of the string
// len contains the length of the string
char *BAMfile::parse_header_tag(uint32 *len, const char *tag, char *start_ptr, char *end_ptr)
{
    char *tag_id = strstr(start_ptr, tag);
    if (tag_id == NULL)
    {
        // tag not found
        return NULL;
    }

    tag_id += 3;

    // ID:string is followed either by
    // - other tags (separated by spaces or tabs)
    // - a line break
    // - the end of the header string

    // search for the tab first
    char *id_end = strchr(tag_id, '\t');

    if (id_end == NULL)
        // not found? try a space
        id_end = strchr(tag_id, ' ');

    if (id_end == NULL)
        // not found? try the line break
        id_end = end_ptr;

    *len = id_end - tag_id;
    return tag_id;
}

bool BAMfile::init(void)
{
    int c;

// read in a structure field from fp
// returns error (local variable of the right type) if read fails
#define GZREAD(field)                                           \
    if (readData(&(field), sizeof(field), __LINE__) == false) {           \
        eof = true;                                             \
        return false;                                           \
    }

    // parse the BAM header
    GZREAD(header.magic);

    if (header.magic[0] != 'B' ||
        header.magic[1] != 'A' ||
        header.magic[2] != 'M' ||
        header.magic[3] != '\1')
    {
        throw nvbio::runtime_error("error parsing BAM file (invalid magic)");
    }

    // read in header
    GZREAD(header.l_text);
    header.text.resize(header.l_text);
    readData(&header.text[0], header.l_text, __LINE__);

    // parse header text looking for @RG identifiers and put them in our hash map
    header.n_read_groups = 0;

    char *start_ptr = strchr(&header.text[0], '@');
    while(start_ptr != NULL && *start_ptr != '\0')
    {
        assert(*start_ptr == '@');
        start_ptr++;

        char *end_ptr = strchr(start_ptr, '\n');

        if (start_ptr[0] == 'R' && start_ptr[1] == 'G')
        {
            char *id;
            uint32 len;

            // search for the platform unit first; if found, use that as the read group
            id = parse_header_tag(&len, "PU:", start_ptr, end_ptr);
            if (!id)
            {
                // no PU tag, search for ID
                id = parse_header_tag(&len, "ID:", start_ptr, end_ptr);
            }

            if (id)
            {
                char *id_end = id + len;

                // null out the end of the string
                char tmp = *id_end;
                *id_end = '\0';

                header.read_group_names.push_back(std::string(id));
                // store the hash and a sequential ID in the map
                header.rg_name_to_id[bqsr_string_hash(id)] = header.n_read_groups;
                header.n_read_groups++;

                // restore the nulled-out character
                *id_end = tmp;
            }
        }

        start_ptr = end_ptr + 1;
    }

    // read reference sequence data
    GZREAD(header.n_ref);
    for(c = 0; c < header.n_ref; c++)
    {
        std::string name;
        int32 l_name, l_ref;

        GZREAD(l_name);
        name.resize(l_name);
        readData(&name[0], l_name, __LINE__);
        GZREAD(l_ref);

        header.sq_names.push_back(name);
        header.sq_lengths.push_back(l_ref);
    }

    data_start = gztell(fp);
    return true;

#undef GZREAD
}

bool BAMfile::next_batch(BAM_alignment_batch *batch, bool skip_headers, const uint32 batch_size)
{
    BAM_alignment_batch_host *h_batch = &batch->host;
    h_batch->reset(batch_size, skip_headers);

    // temp vector for translating CIGARs
    nvbio::vector<host_tag, uint32> cigar_temp(64);
    BAM_alignment_header discard_alignment;

    uint32 read_id;
    for(read_id = 0; read_id < batch_size; read_id++)
    {
        z_off_t read_block_start;

        if (eof)
            break;

        if (!skip_headers)
        {
            // allocate space for another read
            h_batch->align_headers.resize(read_id + 1);
        }

        // figure out storage for the alignment header: either a temp stack object or the output object
        BAM_alignment_header& align = (skip_headers ? discard_alignment : h_batch->align_headers[read_id]);

// utility macro to read in a value from disk
#define GZREAD(field)                                           \
    if (readData(&(field), sizeof(field), __LINE__) == false) {           \
        eof = true;                                             \
        break;                                                  \
    }

        // read in the block_size
        GZREAD(align.block_size);

        // record the starting file position for this read block
        read_block_start = gztell(fp);

        GZREAD(align.refID);
        GZREAD(align.pos);
        GZREAD(align.bin_mq_nl);
        GZREAD(align.flag_nc);
        GZREAD(align.l_seq);
        GZREAD(align.next_refID);
        GZREAD(align.next_pos);
        GZREAD(align.tlen);

        const uint32 read_name_off = h_batch->names.size();

        if (!skip_headers)
        {
            const uint32 read_name_len = align.l_read_name();

            h_batch->names.resize(read_name_off + read_name_len + 1);
            readData(&h_batch->names[read_name_off], read_name_len, __LINE__);
            h_batch->names[read_name_off + read_name_len] = '\0';
        } else {
            const uint32 read_name_len = align.bin_mq_nl & 0xff;
            gzseek(fp, read_name_len, SEEK_CUR);
        }

        h_batch->num_reads++;

        // push the CRQ index
        BAM_CRQ_index crq_index(h_batch->cigars.size(), align.num_cigar_ops(), h_batch->reads.size(), align.l_seq, h_batch->qualities.size(), align.l_seq);
        h_batch->crq_index.push_back(crq_index);

        // push the alignment position
        h_batch->alignment_positions.push_back(align.pos); // BAM pos is 0-based

        // make sure the refID is valid and push it
        if (align.refID < 0 || align.refID >= header.n_ref)
        {
            // push an invalid refID
            h_batch->alignment_sequence_IDs.push_back(uint32(-1));
        } else {
            h_batch->alignment_sequence_IDs.push_back(align.refID);
        }

        h_batch->mapq.push_back(align.mapq());

        // figure out the CIGAR length and make sure we can store it
        const uint32 cigar_len = (align.flag_nc & 0xffff);
        //batch->cigars.resize(crq_index.cigar + cigar_len);

        // read in the CIGAR and translate it into BAM_cigar_ops
        cigar_temp.resize(cigar_len);
        if (cigar_len)
        {
            readData(&cigar_temp[0], sizeof(uint32) * cigar_len, __LINE__);
            for(uint32 c = 0; c < cigar_len; c++)
            {
                cigar_op op;
                op.op = cigar_temp[c] & 0xf;
                op.len = cigar_temp[c] >> 4;

                h_batch->cigars.push_back(op);
            }
        }

        // figure out the length of the sequence data,
        // rounded up to reach a dword boundary
        const uint32 padded_read_len_bp = ((align.l_seq + 7) / 8) * 8;

        // make sure we have enough memory, then read in the sequence
        h_batch->reads.resize(crq_index.read_start + padded_read_len_bp);
        uint32 *storage = (uint32 *)h_batch->reads.addrof(crq_index.read_start);
        readData(storage, align.l_seq / 2, __LINE__);

        // swap nibbles since nvbio expects them swapped
        // xxxnsubtil: this should really be fixed in PackedStream instead
        for(uint32 c = 0; c < padded_read_len_bp / 8; c++)
        {
            uint32 a = storage[c] & 0xf0f0f0f0;
            uint32 b = storage[c] & 0x0f0f0f0f;
            storage[c] = (a >> 4) | (b << 4);
        }

        // read in quality data
        h_batch->qualities.resize(crq_index.read_start + align.l_seq);
        readData(&h_batch->qualities[crq_index.read_start], align.l_seq, __LINE__);

        // store the flags for each
        h_batch->flags.push_back(align.flags());

        // compute auxiliary data size
        const uint32 aux_len = align.block_size - (gztell(fp) - read_block_start);

        // read in aux data
        const uint32 aux_start = h_batch->aux_data.size();
        h_batch->aux_data.resize(aux_start + aux_len);
        readData(&h_batch->aux_data[aux_start], aux_len, __LINE__);

        // push the header index and read in aux data
        BAM_alignment_index idx(h_batch->aux_data.size(), aux_len, read_name_off);
        h_batch->index.push_back(idx);

        // walk the aux data looking for the read group
        char *aux_ptr = &h_batch->aux_data[aux_start];
        while(aux_ptr < &h_batch->aux_data[aux_start + aux_len])
        {
            BAM_alignment_tag *tag = (BAM_alignment_tag *) aux_ptr;
            aux_ptr += 3;

            // compute tag length
            uint32 tag_len;
            switch(tag->val_type)
            {
            case 'A': // printable char
            case 'c': // i8
            case 'C': // u8
                tag_len = 1;
                break;

            case 's': // i16
            case 'S': // u16
                tag_len = 2;
                break;

            case 'i': // i32
            case 'I': // u32
            case 'f': // f32
                tag_len = 4;
                break;

            case 'Z':
                tag_len = strlen(aux_ptr) + 1;
                break;

            case 'H':
                // no clue what to do here
                NVBIO_CUDA_ASSERT(!"someone please help me!!");
                exit(1);
                break;

            case 'B':
                // implement later...
                NVBIO_CUDA_ASSERT(!"byte array not implemented");
                exit(1);
                break;

            default:
                NVBIO_CUDA_ASSERT(!"invalid tag type");
                exit(1);
                break;
            }

            if (tag->tag[0] == 'R' && tag->tag[1] == 'G')
            {
                // found a read group tag
                assert(tag->val_type == 'Z');
                uint32 h = bqsr_string_hash(aux_ptr);

                if (header.rg_name_to_id.find(h) == header.rg_name_to_id.end())
                {
                    // invalid read group
                    h_batch->read_groups[read_id] = uint32(-1);
                } else {
                    h_batch->read_groups[read_id] = header.rg_name_to_id[h];
                }
            }

            aux_ptr += tag_len;
        }
#undef GZREAD
    }

    if (read_id == 0)
        return false;

    return true;
}