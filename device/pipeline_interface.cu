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

#include "pipeline.h"

#include "alignment_data_device.h"
#include "../sequence_database.h"
#include "../variant_database.h"
#include "../command_line.h"

#include <lift/backends.h>
#include <lift/sys/compute_device.h>

#include <thread>

#include <lift/sys/cuda/compute_device_cuda.h>
#include <lift/sys/host/compute_device_host.h>
#include <tbb/task_scheduler_init.h>

namespace firepony {

template <target_system system> void firepony_process_batch(firepony_context<system>& context, const alignment_batch<system>& batch);
template <target_system system> void firepony_pipeline_end(firepony_context<system>& context);
template <target_system system> void firepony_postprocess(firepony_context<system>& context);

template <target_system system_dst, target_system system_src>
void firepony_gather_intermediates(firepony_context<system_dst>& context, firepony_context<system_src>& other)
{
    context.covariates.quality.concat(context.compute_device, other.compute_device, other.covariates.quality);
    context.covariates.cycle.concat(context.compute_device, other.compute_device, other.covariates.cycle);
    context.covariates.context.concat(context.compute_device, other.compute_device, other.covariates.context);
}

template <target_system system>
struct firepony_device_pipeline : public firepony_pipeline
{
    lift::compute_device *device;

    uint32 consumer_id;

    sequence_database_host *host_reference;
    variant_database_host *host_dbsnp;

    alignment_header<system> *header;
    sequence_database_storage<system> *reference;
    variant_database_storage<system> *dbsnp;

    firepony_context<system> *context;
    alignment_batch<system> *batch;

    io_thread *reader;

    std::thread thread;

    firepony_device_pipeline(uint32 consumer_id, lift::compute_device *device)
        : consumer_id(consumer_id), device(device)
    { }

    virtual std::string get_name(void) override
    {
        return std::string(device->get_name());
    }

    virtual target_system get_system(void) override
    {
        return device->get_system();
    }

    virtual size_t get_total_memory(void) override
    {
        return size_t(-1);
    }

    virtual pipeline_statistics& statistics(void) override
    {
        return context->stats;
    }

    virtual void setup(io_thread *reader,
                       const runtime_options *options,
                       alignment_header_host *h_header,
                       sequence_database_host *host_reference,
                       variant_database_host *host_dbsnp) override
    {
        device->enable();

        this->reader = reader;
        this->host_reference = host_reference;
        this->host_dbsnp = host_dbsnp;

        header = new alignment_header<system>(*h_header);
        reference = new sequence_database_storage<system>();
        dbsnp = new variant_database_storage<system>();

        header->download();

        context = new firepony_context<system>(*device, *options, *header);
        batch = new alignment_batch<system>();
    }

    virtual void start(void) override
    {
        thread = std::thread(&firepony_device_pipeline<system>::run, this);
    }

    virtual void join(void) override
    {
        thread.join();
    }

    virtual void gather_intermediates(firepony_pipeline *other) override
    {
        switch(other->get_system())
        {
        case lift::cuda:
        {
            device->enable();

            firepony_device_pipeline<lift::cuda> *other_cuda = (firepony_device_pipeline<lift::cuda> *) other;
            firepony_gather_intermediates(*context, *other_cuda->context);
            break;
        }

        case lift::host:
        {
            firepony_device_pipeline<lift::host> *other_tbb = (firepony_device_pipeline<lift::host> *) other;
            firepony_gather_intermediates(*context, *other_tbb->context);
            break;
        }

        default:
            assert(!"can't happen");
        }
    }

    virtual void postprocess(void) override
    {
        device->enable();

        // this object must stay alive on the stack for the scheduler to work
        // this means we have to declare it even for the GPU path
        tbb::task_scheduler_init init(tbb::task_scheduler_init::deferred);
        if (system == host)
        {
            init.initialize(command_line_options.cpu_threads);
        }

        firepony_postprocess(*context);
    }

private:
    void run(void)
    {
        device->enable();

        // this object must stay alive on the stack for the scheduler to work
        // this means we have to declare it even for the GPU path
        tbb::task_scheduler_init init(tbb::task_scheduler_init::deferred);
        if (system == host)
        {
            init.initialize(command_line_options.cpu_threads);
        }

        timer<host> io_timer;
        alignment_batch_host *h_batch;

        for(;;)
        {
            // try to get a batch to work on
            io_timer.start();
            h_batch = reader->get_batch();
            io_timer.stop();

            if (h_batch == nullptr)
            {
                // no more data, we're done
                break;
            }

            if (!command_line_options.null_pipeline)
            {
                // download/evict reference and dbsnp segments
                reference->update_resident_set(*host_reference, h_batch->chromosome_map);
                dbsnp->update_resident_set(*host_dbsnp, h_batch->chromosome_map);

                // download alignment data to the device
                batch->download(h_batch);

                // update context database pointers
                context->update_databases(*reference, *dbsnp);

                // process the batch
                firepony_process_batch(*context, *batch);
            }

            // return it to the reader for reuse
            reader->retire_batch(h_batch);
        }

        if (!command_line_options.null_pipeline)
            firepony_pipeline_end(*context);

        statistics().io.add(io_timer);
    }
};

template <>
std::string firepony_device_pipeline<lift::host>::get_name(void)
{
    lift::compute_device_host& d = (lift::compute_device_host&)(*device);

    char buf[1024];
    snprintf(buf, sizeof(buf), " (%d threads)", d.num_threads);

    return device->get_name() + std::string(buf);
}

template <>
size_t firepony_device_pipeline<lift::cuda>::get_total_memory(void)
{
    lift::compute_device_cuda& d = (lift::compute_device_cuda&)(*device);
    return size_t(d.config.total_memory);
}

firepony_pipeline *firepony_pipeline::create(lift::compute_device *device)
{
    static uint32 current_consumer_id = 0;
    uint32 consumer_id = current_consumer_id;
    current_consumer_id++;
    const auto system = device->get_system();

    switch(system)
    {
    case lift::cuda:
        return new firepony_device_pipeline<lift::cuda>(consumer_id, device);

    case lift::host:
        return new firepony_device_pipeline<lift::host>(consumer_id, device);

    default:
        current_consumer_id--;  // we didn't actually create anything
        return nullptr;
    }
}

} // namespace firepony
