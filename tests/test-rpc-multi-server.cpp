#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cpp.h"
#include "ggml-impl.h"
#include "ggml-rpc.h"
#include "ggml.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <vector>

static ggml_backend_meta_split_state test_split(const ggml_tensor *, void * data) {
    return *(ggml_backend_meta_split_state *) data;
}

static void test_meta_uploads(ggml_backend_t a, ggml_backend_t b) {
    ggml_backend_dev_t devices[] = {ggml_backend_get_device(a), ggml_backend_get_device(b)};
    for (auto type : {GGML_TYPE_F32, GGML_TYPE_Q4_K}) {
        for (int axis : {0, 1, 2, int(GGML_BACKEND_SPLIT_AXIS_MIRRORED)}) {
            for (int layout = 0; layout < 3; layout++) {
                if (layout == 2 && axis > 1) {
                    continue;
                }
                ggml_context_ptr ctx(ggml_init({ggml_tensor_overhead(), nullptr, true}));
                auto * tensor = ggml_new_tensor_3d(ctx.get(), type, 2048, 16, layout == 2 && axis == 0 ? 1 : 8);
                ggml_backend_meta_split_state split = {};
                split.axis = (ggml_backend_meta_split_axis) axis;
                split.n_segments = 1;
                split.nr[0] = 1;
                if (axis < GGML_MAX_DIMS) {
                    const int64_t n = tensor->ne[axis];
                    split.ne[0] = layout == 1 ? 0 : n/4;
                    split.ne[1] = n - split.ne[0];
                    if (layout == 2) {
                        split.n_segments = 2;
                        split.nr[0] = 2;
                        split.nr[1] = 1;
                        split.ne[0] = split.ne[1] = split.ne[2] = n/8;
                        split.ne[3] = 3*n/8;
                    }
                }
                auto * dev = ggml_backend_meta_device(devices, 2, test_split, &split);
                ggml_backend_ptr backend(ggml_backend_dev_init(dev, nullptr));
                ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
                GGML_ASSERT(buffer);
                ggml_backend_buffer_set_usage(buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
                std::vector<uint8_t> expected(ggml_nbytes(tensor)), output(expected.size());
                for (size_t i = 0; i < expected.size(); i++) {
                    expected[i] = (17*i + 3) % 251;
                }
                ggml_backend_tensor_set(tensor, expected.data(), 0, expected.size());
                ggml_backend_tensor_get(tensor, output.data(), 0, output.size());
                GGML_ASSERT(output == expected);
                for (auto & value : expected) {
                    value ^= 0x5a;
                }
                std::array<ggml_backend_event_ptr, 2> events;
                std::array<std::vector<uint8_t>, 2> staging;
                for (auto & event : events) {
                    event.reset(ggml_backend_event_new(dev));
                    GGML_ASSERT(event);
                    ggml_backend_event_record(event.get(), backend.get());
                }
                const size_t chunks[] = {13, 257, 65537, 65541};
                size_t index = 0;
                for (size_t offset = 0; offset < expected.size(); index++) {
                    const size_t slot = index % events.size();
                    ggml_backend_event_synchronize(events[slot].get());
                    const size_t size = std::min(chunks[index % 4], expected.size() - offset);
                    staging[slot].assign(expected.begin() + offset, expected.begin() + offset + size);
                    ggml_backend_tensor_set_async(backend.get(), tensor, staging[slot].data(), offset, size);
                    ggml_backend_event_record(events[slot].get(), backend.get());
                    offset += size;
                }
                for (size_t i = 0; i < events.size(); i++) {
                    ggml_backend_event_synchronize(events[i].get());
                    std::fill(staging[i].begin(), staging[i].end(), 0);
                }
                ggml_backend_tensor_get(tensor, output.data(), 0, output.size());
                GGML_ASSERT(output == expected);
            }
        }
    }
}

static void test_uploads(ggml_backend_t backend) {
    ggml_init_params params = {ggml_tensor_overhead(), nullptr, true};
    ggml_context * ctx = ggml_init(params);
    GGML_ASSERT(ctx != nullptr);
    const size_t count = 4 * 1024 * 1024 + 3;
    ggml_tensor * tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, count);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buffer != nullptr);
    ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::vector<float> input(count), output(count);
    for (size_t i = 0; i < count; i++) {
        input[i] = float(i % 113);
    }
    const auto expected = input;
    ggml_backend_tensor_set(tensor, input.data(), 0, input.size() * sizeof(float));
    std::fill(input.begin(), input.end(), -1.0f);
    ggml_backend_tensor_get(tensor, output.data(), 0, output.size() * sizeof(float));
    GGML_ASSERT(output == expected);

    ggml_backend_event_t event = ggml_backend_event_new(ggml_backend_get_device(backend));
    GGML_ASSERT(event != nullptr);
    for (const size_t length : {count, size_t(17), count - 3, count - 3}) {
        const size_t offset = length == count ? 0 : 1;
        input = expected;
        ggml_backend_tensor_set_async(backend, tensor, input.data(), offset * sizeof(float), length * sizeof(float));
        ggml_backend_event_record(event, backend);
        ggml_backend_event_synchronize(event);
        std::fill(input.begin(), input.end(), -2.0f);
        ggml_backend_tensor_get(tensor, output.data(), offset * sizeof(float), length * sizeof(float));
        GGML_ASSERT(std::equal(output.begin(), output.begin() + length, expected.begin()));
    }
    ggml_backend_event_free(event);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

static void test_strided_upload(ggml_backend_t backend) {
    ggml_context_ptr ctx(ggml_init({2*ggml_tensor_overhead(), nullptr, true}));
    auto * tensor = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 256);
    auto * view = ggml_view_2d(ctx.get(), tensor, 8, 8, 16*sizeof(float), 5*sizeof(float));
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer);
    std::vector<float> input(88), output(256, 0.0f), expected(256, 0.0f);
    ggml_backend_tensor_set(tensor, output.data(), 0, output.size()*sizeof(float));
    for (size_t row = 0; row < 8; row++) {
        for (size_t col = 0; col < 8; col++) {
            input[row*11 + col] = expected[5 + row*16 + col] = float(1 + row*8 + col);
        }
    }
    ggml_backend_event_ptr event(ggml_backend_event_new(ggml_backend_get_device(backend)));
    GGML_ASSERT(event);
    ggml_backend_tensor_set_2d_async(backend, view, input.data(), 0, 8*sizeof(float), 8, 16*sizeof(float), 11*sizeof(float));
    ggml_backend_event_record(event.get(), backend);
    ggml_backend_event_synchronize(event.get());
    std::fill(input.begin(), input.end(), -1.0f);
    ggml_backend_tensor_get(tensor, output.data(), 0, output.size()*sizeof(float));
    GGML_ASSERT(output == expected);
}

static void test_allreduce(ggml_backend_t a, ggml_backend_t b) {
    auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(a));
    auto comm_init = (ggml_backend_comm_init_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_init");
    auto comm_free = (ggml_backend_comm_free_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_free");
    auto allreduce = (ggml_backend_comm_allreduce_tensor_t) ggml_backend_reg_get_proc_address(
            reg, "ggml_backend_comm_allreduce_tensor");
    GGML_ASSERT(comm_init && comm_free && allreduce);
    ggml_backend_t backends[] = {a, b};
    void * comm = comm_init(backends, 2);
    GGML_ASSERT(comm);

    // Exercise both wire formats, including the BF16 threshold and scratch reuse.
    for (int64_t count : {4096, 32767, 32768, 32769, 65536, 32768}) {
        std::array<ggml_context_ptr, 2> contexts;
        std::array<ggml_backend_buffer_ptr, 2> buffers;
        ggml_tensor * tensors[2];
        std::array<std::vector<float>, 2> values, output;
        for (size_t rank = 0; rank < 2; rank++) {
            contexts[rank].reset(ggml_init({ggml_tensor_overhead(), nullptr, true}));
            tensors[rank] = ggml_new_tensor_1d(contexts[rank].get(), GGML_TYPE_F32, count);
            tensors[rank]->flags |= GGML_TENSOR_FLAG_COMPUTE;
            buffers[rank].reset(ggml_backend_alloc_ctx_tensors(contexts[rank].get(), backends[rank]));
            GGML_ASSERT(buffers[rank]);
            values[rank].resize(count);
            output[rank].resize(count);
        }
        for (int iteration = 0; iteration < 2; iteration++) {
            for (size_t rank = 0; rank < 2; rank++) {
                for (int64_t i = 0; i < count; i++) {
                    // These contributions are deliberately not exactly representable in BF16.
                    values[rank][i] = rank == 0 ? 1.001f + float((i + iteration) % 97)*0.113f
                                               : -0.9984f - float((3*i + iteration) % 89)*0.027f;
                }
                ggml_backend_tensor_set(tensors[rank], values[rank].data(), 0, count*sizeof(float));
            }
            GGML_ASSERT(allreduce(comm, tensors));
            for (size_t rank = 0; rank < 2; rank++) {
                ggml_backend_tensor_get(tensors[rank], output[rank].data(), 0, count*sizeof(float));
            }
            for (int64_t i = 0; i < count; i++) {
                float lhs = values[0][i], rhs = values[1][i];
                if (count >= 32768) {
                    lhs = ggml_bf16_to_fp32(ggml_fp32_to_bf16(lhs));
                    rhs = ggml_bf16_to_fp32(ggml_fp32_to_bf16(rhs));
                }
                const float expected = lhs + rhs;
                if (output[0][i] != expected || output[1][i] != expected) {
                    fprintf(stderr, "allreduce count=%lld index=%lld: ranks %g, %g; expected %g\n",
                            (long long) count, (long long) i, output[0][i], output[1][i], expected);
                    GGML_ABORT("RPC allreduce must produce the same sum on both ranks");
                }
            }
        }
        fprintf(stderr, "allreduce count=%lld: PASS\n", (long long) count);
    }
    comm_free(comm);
}

int main(int argc, char ** argv) {
    GGML_ASSERT(argc == 3);
    ggml_backend_load_all();

    const char * endpoint_a = argv[1];
    const char * endpoint_b = argv[2];

    ggml_backend_t backend_a = ggml_backend_rpc_init(endpoint_a, 0);
    ggml_backend_t backend_b = ggml_backend_rpc_init(endpoint_b, 0);
    GGML_ASSERT(backend_a != nullptr);
    GGML_ASSERT(backend_b != nullptr);

    ggml_init_params params = {
        /* .mem_size   = */ 3*ggml_tensor_overhead() + ggml_graph_overhead_custom(1, false),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    GGML_ASSERT(ctx != nullptr);

    ggml_tensor * tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend_a);
    GGML_ASSERT(buffer != nullptr);

    // A remote pointer allocated by server A is not meaningful to server B.
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 1, false);
    graph->nodes[0] = tensor;
    graph->n_nodes = 1;

    GGML_ASSERT(ggml_backend_graph_compute(backend_b, graph) == GGML_STATUS_SUCCESS);
    // Wait for server B to finish the graph before the script checks its log.
    size_t free_mem;
    size_t total_mem;
    ggml_backend_rpc_get_device_memory(endpoint_b, 0, &free_mem, &total_mem);
    GGML_ASSERT(total_mem > 0);
    ggml_backend_buffer_free(buffer);

    // Two tensors with the same ne[] but different nb[] must not share a cached alloc size.
    // ref: https://github.com/ggml-org/llama.cpp/issues/28360
    ggml_backend_buffer_type_t buft = ggml_backend_rpc_buffer_type(endpoint_a, 0);
    GGML_ASSERT(buft != nullptr);

    // MUL_MAT may need extra memory, so the size is read from the server [TAG_ALLOC_SIZE_EXPAND]
    ggml_tensor * packed = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
    packed->op = GGML_OP_MUL_MAT;

    // same ne[], twice the row stride
    ggml_tensor * strided = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
    strided->op = GGML_OP_MUL_MAT;
    strided->nb[1] = 2*strided->nb[1];
    strided->nb[2] = strided->ne[1]*strided->nb[1];
    strided->nb[3] = strided->nb[2];
    GGML_ASSERT(ggml_nbytes(strided) > ggml_nbytes(packed));

    // ask for the packed tensor first, so a cache keyed without nb[] holds the smaller size
    GGML_ASSERT(ggml_backend_buft_get_alloc_size(buft, packed)  >= ggml_nbytes(packed));
    GGML_ASSERT(ggml_backend_buft_get_alloc_size(buft, strided) >= ggml_nbytes(strided));

    ggml_free(ctx);
    test_allreduce(backend_a, backend_b);
    test_uploads(backend_a);
    test_uploads(backend_b);
    test_strided_upload(backend_a);
    test_strided_upload(backend_b);
    test_meta_uploads(backend_a, backend_b);
    ggml_backend_free(backend_b);
    ggml_backend_free(backend_a);
    return 0;
}
