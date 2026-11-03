#pragma once

#include "common.cuh"
#include <memory>
#include <vector>

struct ggml_cuda_q8_reuse_scope;
inline thread_local ggml_cuda_q8_reuse_scope * ggml_cuda_q8_reuse_current = nullptr;

struct ggml_cuda_q8_reuse_scope {
    struct entry {
        const ggml_tensor * input;
        int first;
        int last;
        int count = 1;
        bool ready = false;
        ggml_cuda_pool_alloc<char> quantized;

        entry(const ggml_tensor * input, int i) : input(input), first(i), last(i) {}
    };

    ggml_backend_cuda_context & ctx;
    ggml_cuda_q8_reuse_scope * previous;
    std::vector<std::unique_ptr<entry>> entries;
    uint64_t uid;
    int hits = 0;
    int misses = 0;

    static const ggml_tensor * origin(const ggml_tensor * tensor) {
        while (tensor->src[0] && (tensor->op == GGML_OP_VIEW || tensor->op == GGML_OP_RESHAPE ||
               tensor->op == GGML_OP_PERMUTE || tensor->op == GGML_OP_TRANSPOSE)) {
            tensor = tensor->src[0];
        }
        return tensor;
    }

    static bool same_input(const ggml_tensor * a, const ggml_tensor * b) {
        return origin(a) == origin(b) && a->data == b->data && a->type == b->type &&
               ggml_are_same_shape(a, b) && ggml_are_same_stride(a, b);
    }

    static bool overlaps(const ggml_tensor * a, const ggml_tensor * b) {
        const uintptr_t pa = (uintptr_t) a->data;
        const uintptr_t pb = (uintptr_t) b->data;
        return pa <= pb ? pb - pa < ggml_nbytes(a) : pa - pb < ggml_nbytes(b);
    }

    ggml_cuda_q8_reuse_scope(ggml_backend_cuda_context & ctx, const ggml_cgraph * graph, bool single_stream)
        : ctx(ctx), previous(ggml_cuda_q8_reuse_current), uid(graph->uid) {
        ggml_cuda_q8_reuse_current = nullptr;
        static const bool enabled = [] {
            const char * value = getenv("GGML_CUDA_Q8_REUSE");
            return value && atoi(value) != 0;
        }();
        if (!enabled || !single_stream || ctx.curr_stream_no != 0) {
            return;
        }

        for (int i = 0; i < graph->n_nodes; ++i) {
            const ggml_tensor * node = graph->nodes[i];
            if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 ||
                (node->op != GGML_OP_MUL_MAT && node->op != GGML_OP_MUL_MAT_ID) ||
                !ggml_is_quantized(node->src[0]->type)) {
                continue;
            }
            const ggml_tensor * input = node->src[1];
            if (input->type != GGML_TYPE_F32 || input->ne[0] <= 0 || input->ne[0] > 65536 ||
                input->ne[1] != 1 || input->ne[2] != 1 || input->ne[3] != 1 ||
                !ggml_is_contiguous(input)) {
                continue;
            }
            entry * found = nullptr;
            for (auto & candidate : entries) {
                if (same_input(candidate->input, input)) {
                    found = candidate.get();
                    break;
                }
            }
            if (found) {
                found->last = i;
                ++found->count;
            } else if (entries.size() < 32) {
                entries.emplace_back(std::make_unique<entry>(input, i));
            }
        }

        for (auto it = entries.begin(); it != entries.end();) {
            entry & candidate = **it;
            bool valid = candidate.count > 1;
            // Include skipped fusion outputs in the write check.
            for (int i = candidate.first; valid && i < candidate.last; ++i) {
                const ggml_tensor * node = graph->nodes[i];
                if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 ||
                    node->op == GGML_OP_NONE || node->op == GGML_OP_VIEW || node->op == GGML_OP_RESHAPE ||
                    node->op == GGML_OP_PERMUTE || node->op == GGML_OP_TRANSPOSE) {
                    continue;
                }
                valid = !overlaps(candidate.input, node);
            }
            if (!valid) {
                it = entries.erase(it);
            } else {
                ++it;
            }
        }
        for (auto & candidate : entries) {
            const size_t size = GGML_PAD(candidate->input->ne[0], MATRIX_ROW_PADDING) * sizeof(block_q8_1) / QK8_1;
            candidate->quantized.alloc(ctx.pool(), size);
        }
        ggml_cuda_q8_reuse_current = this;
    }

    ~ggml_cuda_q8_reuse_scope() {
        if (getenv("GLM_Q8_REUSE_STATS") && !entries.empty()) {
            fprintf(stderr, "Q8R\t%llu\t%zu\t%d\t%d\n", (unsigned long long) uid, entries.size(), hits, misses);
        }
        // The VMM scratch pool frees allocations in reverse order.
        while (!entries.empty()) {
            entries.pop_back();
        }
        ggml_cuda_q8_reuse_current = previous;
    }

    char * get(ggml_backend_cuda_context & caller, const ggml_tensor * input, bool & ready) {
        if (&caller != &ctx || caller.curr_stream_no != 0) {
            return nullptr;
        }
        for (auto & candidate : entries) {
            if (same_input(candidate->input, input)) {
                ready = candidate->ready;
                candidate->ready = true;
                ready ? ++hits : ++misses;
                return candidate->quantized.get();
            }
        }
        return nullptr;
    }
};
