#include "allreduce.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

#include "convert.cuh"
#include "ggml-impl.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>

// ---------------------------------------------------------------------------
// CUDA AllReduce for tensor-parallel inference across two GPUs.
//
// Provides an in-place sum reduction over matching tensors on two CUDA
// devices in the same process.  Used by the tensor-split path alongside
// NCCL; targets setups without NVLink, where data is exchanged between the
// GPUs by staging it through pinned host memory over PCIe.
//
// Two reduction strategies are selected per call by tensor size:
//
//   * Chunked kernel path (small reductions): a single CUDA kernel both
//     stages data through pinned host memory and performs the local sum.
//     Cross-GPU synchronization happens *inside the kernel* (busy-wait on
//     a host-memory flag), which keeps launch overhead low for the
//     latency-sensitive token-generation case.
//
//   * Copyengine path (large reductions): the transfer is split into
//     D2H + H2D cudaMemcpyAsync chunks driven by the GPU's copy engine,
//     followed by a small device-side add kernel.  Cross-GPU
//     synchronization happens *outside the kernel*, via CUDA events between
//     streams.  This keeps the compute engine free while large transfers
//     are in flight, which matters for prefill-sized tensors.  host_large
//     and dev_tmp work as ring buffers indexed by chunk slot, so reductions
//     larger than copy_bytes recycle slots within the same call -- no outer
//     chunker needed.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Cross-GPU signal mechanism
//
// One int per (slot, rank) pair in pinned host memory.  Each AR call writes a
// strictly increasing token (= the AR call number) into its own arrival int.
// The peer spins until its read of the other's arrival int equals the token
// it expects for this call -- a mismatch means the peer hasn't arrived yet.
// Tokens never repeat over realistic call rates (32-bit int wraps in tens of
// days at thousands of ARs/sec), so arrival ints don't need to be reset
// between calls; we initialize once at pipeline init and let the values
// accumulate.
//
// There is exactly one writer (the owning GPU) and one reader (the peer), so
// we don't need atomics.  A volatile store paired with __threadfence_system()
// provides the release ordering that makes the D2H writes visible system-wide
// before the arrival token is observed.
//
// atomicAdd_system() requires hostNativeAtomicSupported, which is unavailable
// on PCIe-attached consumer GPUs without NVLink, so the volatile path is the
// portable choice.
// ---------------------------------------------------------------------------

static __device__ __forceinline__ void ggml_cuda_ar_signal_set(int * p, int token) {
    *(volatile int *)p = token;
}
static __device__ __forceinline__ int ggml_cuda_ar_signal_get(const int * p) {
    return *(const volatile int *)p;
}

// Byte spacing between adjacent arrival ints.  64 bytes (one cache line)
// ensures each GPU/block's arrival slot lives on its own line, preventing
// false-sharing stalls on the polling GPU.
static constexpr size_t GGML_CUDA_AR_ARRIVAL_STRIDE = 64;

// Number of blocks the chunked kernel launches with.  Each block stripes a
// disjoint slice of the data and synchronizes through its own arrival-token
// slot so multiple SMs can pump PCIe stores in parallel.
static constexpr int GGML_CUDA_AR_KERNEL_BLOCKS = 8;

// ---------------------------------------------------------------------------
// Chunked kernel AllReduce -- 2 GPUs, supports float, half, and bfloat16.
//
// Both GPUs run this kernel simultaneously on independent streams.  sendbuf
// and recvbuf live in T_dst (the caller's tensor type); host_mine / host_other
// carry data in T_wire (the on-wire type, possibly narrower than T_dst -- e.g.
// T_dst=F32 with T_wire=BF16 halves the bytes pushed across PCIe).  When
// T_dst == T_wire the casts below are no-ops.
//
// The AR is split into n_chunks pieces of up to chunk_max elements each.
// Each block (8 of them) stripes vectors across (gridDim.x * blockDim.x)
// threads to keep multiple SMs issuing PCIe stores in parallel.  For each
// chunk c, both GPUs run:
//
//   Phase 1:  D2H[c]  threadfence_system  signal(token+c)
//   Phase 2:  wait_peer(token+c)  threadfence_system  H2D[c]+sum
//
// Phase 1 issues all D2Hs and signals each per-chunk token as soon as that
// chunk's writes are visible; phase 2 then waits on each token before its
// matching H2D.  Each H2D waits only for its own chunk to be ready on the
// peer rather than for peer's entire D2H phase, so when the two GPUs drift
// out of phase one GPU's H2D can overlap with the other's still-in-progress
// D2H (different PCIe directions on different links).  Buffer is not reused
// across chunks -- host_mine / host_other are sized to the full AR -- so
// peer can read chunk c at any time after we've signalled.
//
// Each block has its own arrival-token slot (offset by bid * ARRIVAL_STRIDE);
// thread 0 of each block signals/spins on that slot independently of other
// blocks.  Tail elements (the leftover < ELEMS_PER_VEC at the end) are
// handled only by block 0 of the last chunk to avoid cross-block writes to
// the same slots.
//
// Spin uses signed-difference compare so a peer that has already advanced
// past the target token does not deadlock us; this also handles 32-bit wrap.
// Outer call_count advances by GGML_CUDA_AR_TOKEN_STRIDE so the inner
// per-chunk tokens never overlap with adjacent ARs.
// ---------------------------------------------------------------------------
template <typename T_dst, typename T_wire>
static __global__ void ggml_cuda_ar_kernel(
        const T_dst  *              sendbuf,
        T_dst        *              recvbuf,
        T_wire       * __restrict__ host_mine,
        const T_wire * __restrict__ host_other,
        int                         count,
        int                         chunk_max,
        int                         n_chunks,
        int *                       arrival_mine,
        int *                       arrival_other,
        int                         token) {

    constexpr int ELEMS_PER_VEC = ggml_cuda_get_max_cpy_bytes() / sizeof(T_wire);
    constexpr int ARRIVAL_INTS  = (int)(GGML_CUDA_AR_ARRIVAL_STRIDE / sizeof(int));

    const int tid  = threadIdx.x;
    const int nt   = blockDim.x;
    const int bid  = blockIdx.x;
    const int gtid = bid * nt + tid;
    const int gnt  = gridDim.x * nt;

    int       * my_slot    = arrival_mine  + bid * ARRIVAL_INTS;
    const int * other_slot = arrival_other + bid * ARRIVAL_INTS;

    // Phase 1: per-chunk D2H.  After each chunk's writes, fence + signal the
    // matching token so peer can begin reading that chunk.
    for (int c = 0; c < n_chunks; ++c) {
        const int chunk_offset = c * chunk_max;
        const int chunk_count  = min(chunk_max, count - chunk_offset);
        const int chunk_vec    = chunk_count / ELEMS_PER_VEC;
        const bool is_last     = (c == n_chunks - 1);

        for (int i = gtid; i < chunk_vec; i += gnt) {
            const int off = chunk_offset + i * ELEMS_PER_VEC;
            T_wire wire[ELEMS_PER_VEC];
            #pragma unroll
            for (int k = 0; k < ELEMS_PER_VEC; ++k) {
                wire[k] = ggml_cuda_cast<T_wire>(sendbuf[off + k]);
            }
            ggml_cuda_memcpy_1<sizeof(wire)>(&host_mine[off], wire);
        }
        // Tail elements live in the last chunk only (count contiguous).
        if (is_last && bid == 0) {
            const int tail_start = chunk_offset + chunk_vec * ELEMS_PER_VEC;
            const int leftover   = count - tail_start;
            if (tid < leftover) {
                host_mine[tail_start + tid] = ggml_cuda_cast<T_wire>(sendbuf[tail_start + tid]);
            }
        }

        __threadfence_system();
        __syncthreads();

        if (tid == 0) {
            ggml_cuda_ar_signal_set(my_slot, token + c);
            __threadfence_system();
        }
        __syncthreads();
    }

    // Phase 2: per-chunk pre-spin then H2D + reduce.
    for (int c = 0; c < n_chunks; ++c) {
        if (tid == 0) {
            const int target = token + c;
            while ((int) (ggml_cuda_ar_signal_get(other_slot) - target) < 0) {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
                __nanosleep(100);
#else
                NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
            }
        }
        __syncthreads();
        __threadfence_system();   // acquire peer's writes for chunk c

        const int chunk_offset = c * chunk_max;
        const int chunk_count  = min(chunk_max, count - chunk_offset);
        const int chunk_vec    = chunk_count / ELEMS_PER_VEC;
        const bool is_last     = (c == n_chunks - 1);

        for (int i = gtid; i < chunk_vec; i += gnt) {
            const int off = chunk_offset + i * ELEMS_PER_VEC;
            T_wire wire[ELEMS_PER_VEC];
            ggml_cuda_memcpy_1<sizeof(wire)>(wire, &host_other[off]);
            #pragma unroll
            for (int k = 0; k < ELEMS_PER_VEC; ++k) {
                const T_wire d_low = ggml_cuda_cast<T_wire>(sendbuf[off + k]);
                recvbuf[off + k] = ggml_cuda_cast<T_dst>(
                    ggml_cuda_cast<float>(d_low) + ggml_cuda_cast<float>(wire[k]));
            }
        }
        if (is_last && bid == 0) {
            const int tail_start = chunk_offset + chunk_vec * ELEMS_PER_VEC;
            const int leftover   = count - tail_start;
            if (tid < leftover) {
                const T_wire d_low = ggml_cuda_cast<T_wire>(sendbuf[tail_start + tid]);
                recvbuf[tail_start + tid] = ggml_cuda_cast<T_dst>(
                    ggml_cuda_cast<float>(d_low) +
                    ggml_cuda_cast<float>(host_other[tail_start + tid]));
            }
        }
    }
}

// Combined load-convert-add kernel.  The peer's contribution arrives as T_src
// (which may be a lower-precision type than T_dst when the BF16 round-trip is
// active).  For bit-equivalence between the two GPUs, dst is first rounded
// through T_src's precision via ggml_cuda_cast -- peer already truncated its own
// value the same way before sending -- so both sides perform identical
// arithmetic.  When T_dst == T_src the round-trip cast is a no-op.
template <typename T_dst, typename T_src>
static __global__ void ggml_cuda_ar_add_kernel(
        T_dst       * __restrict__ dst,
        const T_src * __restrict__ src,
        int count) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int nt  = gridDim.x * blockDim.x;
    for (int i = tid; i < count; i += nt) {
        const T_src d_low = ggml_cuda_cast<T_src>(dst[i]);
        dst[i] = ggml_cuda_cast<T_dst>(
            ggml_cuda_cast<float>(d_low) + ggml_cuda_cast<float>(src[i]));
    }
}

// ---------------------------------------------------------------------------
// Pipeline structure
// ---------------------------------------------------------------------------

// Number of slots in the event / arrival / host_buf ring used by the kernel
// path.  Each AR call grabs slot = call_count % POOL_SIZE; reusing a slot is
// guaranteed safe by a cross-device cudaStreamWaitEvent on the peer's prior
// ev.ker for that slot, issued at AR launch time (see copy_impl).  Four
// slots is the sweet spot in practice: enough to keep small ARs pipelined
// without holding too much pinned memory.
static constexpr int GGML_CUDA_AR_POOL_SIZE = 4;

// Maximum AR wire size (bytes per GPU) handled by the chunked kernel path.
// host_buf is sized to this per pool slot, so it must accommodate the full
// AR -- the kernel does not reuse buffer slots across chunks within a call
// (that's what enables cross-GPU per-chunk overlap).
static constexpr size_t GGML_CUDA_AR_MAX_BYTES = 16 * 1024 * 1024; // 16 MB

// Per-chunk size inside the kernel (BF16 wire bytes).  Smaller = finer-grain
// cross-GPU pipeline but more sync overhead per AR.  Compile-time so we can
// shmoo it.
#ifndef GGML_CUDA_AR_HYBRID_CHUNK_BYTES
#define GGML_CUDA_AR_HYBRID_CHUNK_BYTES (512 * 1024) // 512 KB
#endif // GGML_CUDA_AR_HYBRID_CHUNK_BYTES

// Per-AR token stride.  Each AR reserves this many consecutive token values:
// one per inner sync (post-D2H of each chunk).  256 leaves room for up to
// 256 chunks per AR -- plenty for any chunk size we'd realistically shmoo.
static constexpr int GGML_CUDA_AR_TOKEN_STRIDE = 256;

// Copyengine path: largest tensor accepted on this path; sets host_large /
// dev_tmp allocation size.
static constexpr size_t GGML_CUDA_AR_COPY_MAX_BYTES = 32 * 1024 * 1024; // 32 MB

// AR wire size at which the copyengine path takes over from the kernel
// path.  Override via GGML_CUDA_AR_COPY_THRESHOLD.
static constexpr size_t GGML_CUDA_AR_COPY_THRESHOLD_DEFAULT = 16 * 1024 * 1024; // 16 MB
// Per-call CE chunk-size heuristic: chunk_bytes = clamp(nbytes / 4, MIN, MAX).
// The /4 keeps ~4 chunks in flight at any moment (good D2H/H2D overlap with
// the peer); the clamps cover the cases where nbytes/4 is too small (per-
// memcpy fixed cost dominates) or too large (chunk-level pipelining stalls).
// Env var GGML_CUDA_AR_COPY_CHUNK_BYTES can override with a fixed value.
static constexpr size_t GGML_CUDA_AR_COPY_CHUNK_BYTES_HEURISTIC_MIN = 512 * 1024;       // 512 KB
static constexpr size_t GGML_CUDA_AR_COPY_CHUNK_BYTES_HEURISTIC_MAX = 2 * 1024 * 1024;  // 2 MB
// Absolute floor that an env-var override is allowed to set; this caps the
// per-slot copy-event array.  256 KB -> up to 128 chunks per 32 MB tensor.
static constexpr size_t GGML_CUDA_AR_COPY_CHUNK_BYTES_MIN = 256 * 1024;
// Maximum number of slots host_large is sliced into; equals (host_large_bytes
// / chunk_bytes_min).  Typical operation uses 16 slots (32 MB / 2 MB).  Sets
// the size of the per-(GPU, slot) write/read event arrays.
static constexpr int GGML_CUDA_AR_COPY_MAX_SLOTS =
    static_cast<int>((GGML_CUDA_AR_COPY_MAX_BYTES + GGML_CUDA_AR_COPY_CHUNK_BYTES_MIN - 1) /
                    GGML_CUDA_AR_COPY_CHUNK_BYTES_MIN);

struct ggml_cuda_ar_event_slot {
    cudaEvent_t app = nullptr;  // upstream computation complete
    cudaEvent_t ker = nullptr;  // end-of-AR (slot-pool wraparound)
};

// Mapped pinned host allocation: cudaHostAlloc + cudaHostGetDevicePointer
// in one place, with the host handle preserved for cudaFreeHost.  Used where
// the CPU never touches the buffer -- only the device reads/writes via the
// mapped device pointer.  Required on systems where cudaDevAttrCanUseHost-
// PointerForRegisteredMem is 0 and the host pointer can't be used as a
// device pointer.
struct ggml_cuda_ar_host_mapping {
    uint8_t * host = nullptr;   // cudaFreeHost handle; also the H-side ptr for cudaMemcpyAsync
    uint8_t * dev  = nullptr;   // device-side pointer for kernels / cudaMemset

    cudaError_t alloc(size_t bytes) {
        cudaError_t rc = cudaHostAlloc(reinterpret_cast<void **>(&host), bytes,
                                       cudaHostAllocPortable | cudaHostAllocMapped);
        if (rc != cudaSuccess) {
            host = nullptr;
            return rc;
        }
        rc = cudaHostGetDevicePointer(reinterpret_cast<void **>(&dev), host, 0);
        if (rc != cudaSuccess) {
            cudaFreeHost(host);
            host = nullptr;
            dev  = nullptr;
        }
        return rc;
    }

    void free() {
        if (host) {
            cudaFreeHost(host);
            host = nullptr;
            dev  = nullptr;
        }
    }
};

struct ggml_cuda_ar_pipeline {
    int      n_devices;
    int      devices[GGML_CUDA_MAX_DEVICES];
    size_t   buf_bytes;    // bytes per device in host_buf[]
    size_t   copy_bytes;   // bytes per device in host_large[] / dev_tmp[]
    size_t   copy_threshold;
    size_t   copy_chunk_bytes;
    size_t   bf16_threshold; // tensors >= this size (bytes) are reduced via FP32->BF16 round-trip; 0 disables
    size_t   hybrid_chunk_bytes; // per-chunk size inside the kernel; runtime override of compile-time GGML_CUDA_AR_HYBRID_CHUNK_BYTES
    uint64_t call_count;
    bool     dispatch_logged;    // one-shot: print chosen path + params at first AR
    bool     copy_required_warned;

    // Per-device resources.
    ggml_cuda_ar_host_mapping host_buf[GGML_CUDA_MAX_DEVICES];   // pinned staging (chunked kernel)
    ggml_cuda_ar_host_mapping host_large[GGML_CUDA_MAX_DEVICES]; // pinned staging (copyengine)
    char *                    dev_tmp[GGML_CUDA_MAX_DEVICES];    // device scratch for copyengine path
    cudaStream_t              streams[GGML_CUDA_MAX_DEVICES];    // per-device AR stream (copyengine path)
    ggml_cuda_ar_event_slot   ev_pool[GGML_CUDA_MAX_DEVICES][GGML_CUDA_AR_POOL_SIZE];

    // Copyengine path: per-(GPU, slot) write/read events.  Reused across
    // waves WITHIN one AR (cross-GPU per-chunk fence) and across ARs
    // (host_large slot-reuse fence -- peer must finish reading our
    // host_large[s] before we overwrite it on the next AR / next wave).
    // Single AR stream per GPU serializes everything else; these are the
    // only cross-GPU synchronization points the copy path needs.
    cudaEvent_t write_ev[GGML_CUDA_MAX_DEVICES][GGML_CUDA_AR_COPY_MAX_SLOTS];
    cudaEvent_t read_ev [GGML_CUDA_MAX_DEVICES][GGML_CUDA_AR_COPY_MAX_SLOTS];

    // Arrival ring: ARRIVAL_STRIDE bytes between adjacent ints.  Mapped pinned
    // memory; CPU never reads/writes -- only the kernel and cudaMemset.
    // Use ggml_cuda_ar_arrival_ptr() to index.
    ggml_cuda_ar_host_mapping arrival;
};

// Base pointer for the (slot, rank) per-block token block.  The kernel adds
// blockIdx.x * (ARRIVAL_STRIDE/sizeof(int)) internally to land on its own slot.
static int * ggml_cuda_ar_arrival_ptr(const ggml_cuda_ar_pipeline * p, int slot, int rank) {
    const size_t offset = ((size_t)slot * p->n_devices + rank) *
                          GGML_CUDA_AR_KERNEL_BLOCKS * GGML_CUDA_AR_ARRIVAL_STRIDE;
    return reinterpret_cast<int *>(p->arrival.dev + offset);
}

static uint64_t ggml_cuda_ar_env_u64(const char * name, uint64_t default_value) {
    const char * value = getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return default_value;
    }

    char * end = nullptr;
    const unsigned long long parsed = strtoull(value, &end, 10);
    return end != value ? (uint64_t) parsed : default_value;
}

struct ggml_cuda_ar_slot_info {
    int slot;
    int token;
};

static ggml_cuda_ar_slot_info ggml_cuda_ar_acquire_slot(ggml_cuda_ar_pipeline * p) {
    // The cross-device slot-reuse fence is issued at the AR launch site as
    // a cudaStreamWaitEvent on the peer's prior ev.ker for this slot --
    // non-host-blocking, so the host can keep queuing work while the GPU
    // waits.
    const int slot = static_cast<int>(p->call_count % GGML_CUDA_AR_POOL_SIZE);
    p->call_count++;
    return { slot, (int) (p->call_count * GGML_CUDA_AR_TOKEN_STRIDE) };
}

// Per-AR copyengine chunk size: env-var override if set, else heuristic
// (clamp(nbytes/4, HEURISTIC_MIN, HEURISTIC_MAX)).
static size_t ggml_cuda_ar_chunk_bytes(const ggml_cuda_ar_pipeline * p, size_t nbytes) {
    if (p->copy_chunk_bytes > 0) {
        return p->copy_chunk_bytes;
    }
    return std::min(GGML_CUDA_AR_COPY_CHUNK_BYTES_HEURISTIC_MAX,
                    std::max(GGML_CUDA_AR_COPY_CHUNK_BYTES_HEURISTIC_MIN, nbytes / 4));
}

static void ggml_cuda_ar_wait_for_compute(
        ggml_cuda_ar_pipeline * p, ggml_backend_cuda_context * cuda_ctx, int rank, int slot) {
    ggml_cuda_ar_event_slot & ev = p->ev_pool[rank][slot];
    CUDA_CHECK(cudaEventRecord(ev.app, cuda_ctx->stream()));
    CUDA_CHECK(cudaStreamWaitEvent(p->streams[rank], ev.app));
}

// ---------------------------------------------------------------------------
// Init / free
// ---------------------------------------------------------------------------

ggml_cuda_ar_pipeline * ggml_cuda_ar_pipeline_init(const int * devices, size_t n_devices) {

    if (n_devices != 2) {
        GGML_LOG_DEBUG("%s: internal AllReduce only supports n_devices=2 (got %zu); "
                       "falling back\n", __func__, n_devices);
        return nullptr;
    }

    // The chunked kernel uses __nanosleep, which is sm70+ (Volta+).
    for (size_t i = 0; i < n_devices; ++i) {
        const int cc = ggml_cuda_info().devices[devices[i]].cc;
        if (cc < GGML_CUDA_CC_VOLTA) {
            GGML_LOG_DEBUG("%s: internal AllReduce requires compute capability >= %d "
                           "(device %d has cc=%d); falling back\n",
                           __func__, GGML_CUDA_CC_VOLTA, devices[i], cc);
            return nullptr;
        }
    }

    auto * p = new ggml_cuda_ar_pipeline{};
    p->n_devices        = n_devices;
    p->copy_bytes       = GGML_CUDA_AR_COPY_MAX_BYTES;
    p->copy_threshold   = ggml_cuda_ar_env_u64("GGML_CUDA_AR_COPY_THRESHOLD", GGML_CUDA_AR_COPY_THRESHOLD_DEFAULT);
    // 0 = use the per-call heuristic (default).  Non-zero env value forces a
    // fixed chunk size for diagnostics, with a floor at COPY_CHUNK_BYTES_MIN.
    p->copy_chunk_bytes = ggml_cuda_ar_env_u64("GGML_CUDA_AR_COPY_CHUNK_BYTES", 0);
    if (p->copy_chunk_bytes > 0 && p->copy_chunk_bytes < GGML_CUDA_AR_COPY_CHUNK_BYTES_MIN) {
        GGML_LOG_WARN("%s: GGML_CUDA_AR_COPY_CHUNK_BYTES=%zu below minimum %zu; clamping\n",
                      __func__, p->copy_chunk_bytes, GGML_CUDA_AR_COPY_CHUNK_BYTES_MIN);
        p->copy_chunk_bytes = GGML_CUDA_AR_COPY_CHUNK_BYTES_MIN;
    }
    // Default 1: BF16 round-trip is always on for F32 inputs (any non-zero
    // ne).  Set GGML_CUDA_AR_BF16_THRESHOLD=0 to disable, or to a larger
    // byte threshold to opt out for small tensors.
    p->bf16_threshold   = ggml_cuda_ar_env_u64("GGML_CUDA_AR_BF16_THRESHOLD", 1);
    // Per-chunk size inside the kernel: env override of compile-time default.
    // Smaller = finer pipeline grain at higher per-chunk sync cost; larger =
    // fewer chunks but worse cross-GPU phase overlap.
    p->hybrid_chunk_bytes = ggml_cuda_ar_env_u64("GGML_CUDA_AR_HYBRID_CHUNK_BYTES",
                                                 GGML_CUDA_AR_HYBRID_CHUNK_BYTES);
    for (size_t i = 0; i < n_devices; ++i) {
        p->devices[i] = devices[i];
    }

    // Per-device streams and event pools.  One AR stream per device on the
    // copyengine path: stream-sequential ordering serializes ARs and gives
    // wave-to-wave dev_tmp slot reuse for free, leaving only the cross-GPU
    // per-chunk events (write_ev / read_ev) as explicit synchronization.
    for (size_t i = 0; i < n_devices; ++i) {
        ggml_cuda_set_device(p->devices[i]);

        if (cudaStreamCreateWithFlags(&p->streams[i], cudaStreamNonBlocking) != cudaSuccess) {
            GGML_LOG_ERROR("%s: cudaStreamCreateWithFlags failed for device %d\n",
                           __func__, p->devices[i]);
            ggml_cuda_ar_pipeline_free(p);
            return nullptr;
        }

        for (int s = 0; s < GGML_CUDA_AR_POOL_SIZE; ++s) {
            bool ok =
                cudaEventCreateWithFlags(&p->ev_pool[i][s].app, cudaEventDisableTiming) == cudaSuccess &&
                cudaEventCreateWithFlags(&p->ev_pool[i][s].ker, cudaEventDisableTiming) == cudaSuccess;
            if (!ok) {
                GGML_LOG_ERROR("%s: cudaEventCreate failed for device %d slot %d\n",
                               __func__, p->devices[i], s);
                ggml_cuda_ar_pipeline_free(p);
                return nullptr;
            }
        }

        for (int s = 0; s < GGML_CUDA_AR_COPY_MAX_SLOTS; ++s) {
            bool ok =
                cudaEventCreateWithFlags(&p->write_ev[i][s], cudaEventDisableTiming) == cudaSuccess &&
                cudaEventCreateWithFlags(&p->read_ev [i][s], cudaEventDisableTiming) == cudaSuccess;
            if (!ok) {
                GGML_LOG_ERROR("%s: cudaEventCreate (chunk events) failed for device %d slot %d\n",
                               __func__, p->devices[i], s);
                ggml_cuda_ar_pipeline_free(p);
                return nullptr;
            }
        }

        // Pre-record events that may be waited on by the very first AR before
        // a real record has been issued: ev.ker (slot-pool wraparound) for
        // every pool slot, and read_ev (cross-AR host_large slot-reuse fence
        // -- the first AR has no prior owner, so the wait must see an
        // already-fired event).  write_ev doesn't need pre-recording: it is
        // only ever waited on after its matching D2H+record has been
        // host-issued in the same wave.
        for (int s = 0; s < GGML_CUDA_AR_POOL_SIZE; ++s) {
            CUDA_CHECK(cudaEventRecord(p->ev_pool[i][s].ker, p->streams[i]));
        }
        for (int s = 0; s < GGML_CUDA_AR_COPY_MAX_SLOTS; ++s) {
            CUDA_CHECK(cudaEventRecord(p->read_ev[i][s], p->streams[i]));
        }
    }

    // Arrival ring: cache-line padded so each GPU's int is on its own line.
    const size_t arrival_bytes =
        (size_t)GGML_CUDA_AR_POOL_SIZE * n_devices *
        GGML_CUDA_AR_KERNEL_BLOCKS * GGML_CUDA_AR_ARRIVAL_STRIDE;
    if (p->arrival.alloc(arrival_bytes) != cudaSuccess) {
        GGML_LOG_ERROR("%s: alloc for arrival ring failed (%zu bytes)\n",
                       __func__, arrival_bytes);
        ggml_cuda_ar_pipeline_free(p);
        return nullptr;
    }
    ggml_cuda_set_device(p->devices[0]);
    if (cudaMemset(p->arrival.dev, 0, arrival_bytes) != cudaSuccess) {
        GGML_LOG_ERROR("%s: cudaMemset for arrival ring failed (%zu bytes)\n",
                       __func__, arrival_bytes);
        ggml_cuda_ar_pipeline_free(p);
        return nullptr;
    }

    // Per-device pinned staging buffers -- POOL_SIZE-deep ring so the chunked-
    // kernel can write the next slot's data while the peer is still reading
    // the previous slot's. Indexed by (slot * buf_bytes) at the call site.
    p->buf_bytes = GGML_CUDA_AR_MAX_BYTES;
    const size_t host_buf_total = (size_t) GGML_CUDA_AR_POOL_SIZE * p->buf_bytes;
    for (size_t i = 0; i < n_devices; ++i) {
        if (p->host_buf[i].alloc(host_buf_total) != cudaSuccess) {
            GGML_LOG_ERROR("%s: alloc for staging failed (%zu bytes)\n",
                           __func__, host_buf_total);
            ggml_cuda_ar_pipeline_free(p);
            return nullptr;
        }
    }

    // Copyengine path: pinned host staging + device scratch, both sliced into
    // a ring of N_SLOTS = (copy_bytes / chunk_bytes) chunk-sized slots.  Slot
    // reuse within an AR is per-chunk (host_large) and per-wave (dev_tmp);
    // cross-AR sync rides on the slot pool's ev.ker as before.
    for (size_t i = 0; i < n_devices; ++i) {
        ggml_cuda_set_device(p->devices[i]);
        if (p->host_large[i].alloc(p->copy_bytes) != cudaSuccess) {
            GGML_LOG_ERROR("%s: alloc for large staging failed (%zu bytes)\n",
                           __func__, p->copy_bytes);
            ggml_cuda_ar_pipeline_free(p);
            return nullptr;
        }
        if (cudaMalloc(reinterpret_cast<void **>(&p->dev_tmp[i]), p->copy_bytes) != cudaSuccess) {
            GGML_LOG_ERROR("%s: cudaMalloc for copy scratch failed (%zu bytes) on device %d\n",
                           __func__, p->copy_bytes, p->devices[i]);
            ggml_cuda_ar_pipeline_free(p);
            return nullptr;
        }
    }

    GGML_LOG_INFO("%s: initialized AllReduce pipeline: %zu GPUs, "
                  "%zu KB chunked kernel staging + %zu MB copyengine staging per GPU\n",
                  __func__, n_devices, p->buf_bytes >> 10, p->copy_bytes >> 20);

    return p;
}

void ggml_cuda_ar_pipeline_free(ggml_cuda_ar_pipeline * p) {
    if (!p) {
        return;
    }

    // Drain all in-flight work before tearing down resources.
    for (int i = 0; i < p->n_devices; ++i) {
        if (p->streams[i]) {
            ggml_cuda_set_device(p->devices[i]);
            cudaStreamSynchronize(p->streams[i]);
        }
    }

    for (int i = 0; i < p->n_devices; ++i) {
        p->host_buf[i].free();
        p->host_large[i].free();
        if (p->dev_tmp[i]) {
            ggml_cuda_set_device(p->devices[i]);
            cudaFree(p->dev_tmp[i]);
        }
        ggml_cuda_set_device(p->devices[i]);
        for (int s = 0; s < GGML_CUDA_AR_POOL_SIZE; ++s) {
            if (p->ev_pool[i][s].app) { cudaEventDestroy(p->ev_pool[i][s].app); }
            if (p->ev_pool[i][s].ker) { cudaEventDestroy(p->ev_pool[i][s].ker); }
        }
        for (int s = 0; s < GGML_CUDA_AR_COPY_MAX_SLOTS; ++s) {
            if (p->write_ev[i][s]) { cudaEventDestroy(p->write_ev[i][s]); }
            if (p->read_ev [i][s]) { cudaEventDestroy(p->read_ev [i][s]); }
        }
        if (p->streams[i]) { cudaStreamDestroy(p->streams[i]); }
    }
    p->arrival.free();
    delete p;
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

// Asymmetric copy_engine path: data sent over PCIe in T_src precision (one
// element of nbytes per ne element); accumulated locally into a T_dst buffer.
// When T_src == T_dst this is the original homogeneous reduction.  When they
// differ (e.g. BF16 wire / F32 accumulator) the add kernel rounds dst through
// T_src for bit-equivalence between GPUs and we skip the otherwise-needed
// post-conversion entirely.
//
// Pipeline layout
// ---------------
//  * One AR stream per device.  Within a wave, we issue all N_SLOTS D2Hs
//    back-to-back, then all N_SLOTS H2Ds, then a single add_kernel covering
//    the wave -- in nSight this looks like a block of D2H rectangles
//    followed by a block of H2D rectangles on each GPU's copy engine, which
//    is the "32 MB read, 32 MB write" pattern PCIe likes.  Stream-sequential
//    ordering on the AR stream gives wave-to-wave dev_tmp slot reuse and
//    cross-AR ordering for free, so the only explicit synchronization is
//    cross-GPU per-chunk events.
//  * host_large[i] is sliced into N_SLOTS = (host_large_size / chunk_bytes)
//    chunk-sized slots; dev_tmp[i] uses the same indexing.  In typical
//    operation N_SLOTS = 16 (32 MB / 2 MB) -- 2 events per slot per GPU
//    gives the user-visible "64 events for hostmem management".
//  * Per-(GPU, slot) events:
//      write_ev[s] - device i's D2H[s] done.  Peer waits before its H2D[s].
//      read_ev[s]  - device i's H2D[s] done.  Peer waits before its next D2H
//                    that will overwrite host_large[s] (next wave within this
//                    AR, or wave 0 of the next AR).
//
// read_ev[s] and ev.ker are pre-recorded at pipeline init so the first AR's
// cross-AR waits and slot-pool wraparound see an already-fired event.
// write_ev doesn't need pre-recording: it is only ever waited on after its
// matching D2H+record has been host-issued in the same wave.
template <typename T_src, typename T_dst>
static bool ggml_cuda_ar_allreduce_copy_impl(
        ggml_cuda_ar_pipeline * p,
        ggml_backend_t        * backends,
        T_src * const           src_buf[GGML_CUDA_MAX_DEVICES],
        T_dst * const           dst_buf[GGML_CUDA_MAX_DEVICES],
        const bool              compute[GGML_CUDA_MAX_DEVICES],
        int64_t                 ne,
        size_t                  nbytes) {
    GGML_ASSERT(p->n_devices == 2);
    GGML_ASSERT(ne <= std::numeric_limits<int>::max());

    const size_t chunk_bytes = ggml_cuda_ar_chunk_bytes(p, nbytes);
    GGML_ASSERT(chunk_bytes > 0);

    const int n_chunks = (int) ((nbytes + chunk_bytes - 1) / chunk_bytes);
    const int n_slots  = (int) (p->copy_bytes / chunk_bytes);
    const int n_waves  = (n_chunks + n_slots - 1) / n_slots;
    GGML_ASSERT(n_slots  >= 1);
    GGML_ASSERT(n_slots  <= GGML_CUDA_AR_COPY_MAX_SLOTS);

    const auto [slot, _token] = ggml_cuda_ar_acquire_slot(p);

    ggml_backend_cuda_context * cuda_ctx[2] = {};
    for (int i = 0; i < 2; ++i) {
        cuda_ctx[i] = static_cast<ggml_backend_cuda_context *>(backends[i]->context);
        GGML_ASSERT(cuda_ctx[i]->device == p->devices[i]);
    }

    // Cross-AR fence: peer's POOL_SIZE-prior AR must be done before we start.
    // Mostly a safety belt -- single-stream serializes copy-path ARs on this
    // GPU naturally, and the per-slot read_ev waits below cover host_large
    // slot reuse against any path peer used.  Pre-fired at init for the first
    // POOL_SIZE ARs.
    for (int i = 0; i < 2; ++i) {
        const int peer = 1 - i;
        ggml_cuda_set_device(p->devices[i]);
        CUDA_CHECK(cudaStreamWaitEvent(p->streams[i], p->ev_pool[peer][slot].ker, 0));
    }

    // Wait for upstream compute on each device before any AR-stream work
    // touches the source tensor.
    for (int i = 0; i < 2; ++i) {
        ggml_cuda_set_device(p->devices[i]);
        ggml_cuda_ar_wait_for_compute(p, cuda_ctx[i], i, slot);
    }

    // Inactive-shard zeroing on the AR stream so it precedes the D2Hs that
    // read src_buf.
    for (int i = 0; i < 2; ++i) {
        if (!compute[i]) {
            ggml_cuda_set_device(p->devices[i]);
            CUDA_CHECK(cudaMemsetAsync(src_buf[i], 0, nbytes, p->streams[i]));
        }
    }

    // Drive both devices' wave pipelines in lockstep.  Per wave we run three
    // passes; within a pass we issue for both devices interleaved so the
    // host-side issue order is correct for cross-device cudaStreamWaitEvent
    // (a wait captures the event's state AT ISSUE TIME, so peer's record
    // must be host-issued before our wait).
    //
    //   pass A: for each chunk in wave: cross-AR slot-reuse wait (peer's
    //           read_ev[s]) + D2H + record write_ev[s]
    //   pass B: for each chunk in wave: wait peer.write_ev[s] + H2D +
    //           record read_ev[s]
    //   pass C: launch the wave's add_kernel (covers all N_SLOTS chunks of
    //           this wave at once)
    for (int w = 0; w < n_waves; ++w) {
        const int    chunk_lo  = w * n_slots;
        const int    chunk_hi  = std::min(chunk_lo + n_slots, n_chunks);
        const size_t wave_off  = (size_t) chunk_lo * chunk_bytes;
        const size_t wave_end  = std::min((size_t) chunk_hi * chunk_bytes, nbytes);
        const size_t wave_size = wave_end - wave_off;

        // Pass A: D2Hs.  Each chunk's D2H waits on peer's read_ev[s] so we
        // don't overwrite host_large[s] before peer is done reading it (from
        // the prior wave or, on wave 0, from the prior AR -- pre-fired at
        // init for the very first AR).
        for (int c = chunk_lo; c < chunk_hi; ++c) {
            const int    s          = c % n_slots;
            const size_t off_src    = (size_t) c * chunk_bytes;
            const size_t off_slot   = (size_t) s * chunk_bytes;
            const size_t this_bytes = std::min(chunk_bytes, nbytes - off_src);

            for (int i = 0; i < 2; ++i) {
                const int peer = 1 - i;
                ggml_cuda_set_device(p->devices[i]);
                CUDA_CHECK(cudaStreamWaitEvent(p->streams[i], p->read_ev[peer][s], 0));
                CUDA_CHECK(cudaMemcpyAsync(
                    p->host_large[i].host + off_slot,
                    reinterpret_cast<char *>(src_buf[i]) + off_src,
                    this_bytes, cudaMemcpyDeviceToHost, p->streams[i]));
                CUDA_CHECK(cudaEventRecord(p->write_ev[i][s], p->streams[i]));
            }
        }

        // Pass B: H2Ds.  Each H2D waits on peer's matching write_ev[s] (just
        // host-issued in pass A); reads from peer's host_large[s] into our
        // dev_tmp[s].
        for (int c = chunk_lo; c < chunk_hi; ++c) {
            const int    s          = c % n_slots;
            const size_t off_src    = (size_t) c * chunk_bytes;
            const size_t off_slot   = (size_t) s * chunk_bytes;
            const size_t this_bytes = std::min(chunk_bytes, nbytes - off_src);

            for (int i = 0; i < 2; ++i) {
                const int peer = 1 - i;
                ggml_cuda_set_device(p->devices[i]);
                CUDA_CHECK(cudaStreamWaitEvent(p->streams[i], p->write_ev[peer][s], 0));
                CUDA_CHECK(cudaMemcpyAsync(
                    p->dev_tmp[i] + off_slot,
                    p->host_large[peer].host + off_slot,
                    this_bytes, cudaMemcpyHostToDevice, p->streams[i]));
                CUDA_CHECK(cudaEventRecord(p->read_ev[i][s], p->streams[i]));
            }
        }

        // Pass C: wave-level add_kernel on the AR stream.  Stream-sequential
        // ordering ensures it runs after all H2Ds; the next wave's D2Hs (and
        // their dev_tmp[s] reuse) are stream-sequential after the kernel.
        GGML_ASSERT((wave_size % sizeof(T_src)) == 0);
        const int wave_ne = (int) (wave_size / sizeof(T_src));
        for (int i = 0; i < 2; ++i) {
            ggml_cuda_set_device(p->devices[i]);
            const int block_size = 256;
            int n_blocks = (wave_ne + block_size - 1) / block_size;
            if (n_blocks > 1024) {
                n_blocks = 1024;
            }
            ggml_cuda_ar_add_kernel<T_dst, T_src><<<n_blocks, block_size, 0, p->streams[i]>>>(
                dst_buf[i] + wave_off / sizeof(T_src),
                reinterpret_cast<const T_src *>(p->dev_tmp[i]),
                wave_ne);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    // End-of-AR: record ev.ker on the AR stream, and have the compute stream
    // wait so subsequent ops see the AR result.  ev.ker also serves as the
    // POOL_SIZE-back fence for future ARs (kernel or copy path).
    for (int i = 0; i < 2; ++i) {
        ggml_cuda_set_device(p->devices[i]);
        CUDA_CHECK(cudaEventRecord(p->ev_pool[i][slot].ker, p->streams[i]));
        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx[i]->stream(), p->ev_pool[i][slot].ker, 0));
    }

    return true;
}

bool ggml_cuda_ar_allreduce(
        ggml_cuda_ar_pipeline * p,
        ggml_backend_t        * backends,
        ggml_tensor           ** tensors) {
    GGML_ASSERT(p != nullptr);

    const int n = p->n_devices;
    GGML_ASSERT(n == 2);

    const ggml_type input_type = tensors[0]->type;
    GGML_ASSERT(input_type == GGML_TYPE_F32 || input_type == GGML_TYPE_F16 || input_type == GGML_TYPE_BF16);

    const int64_t ne = ggml_nelements(tensors[0]);
    GGML_ASSERT(ne > 0);

    const size_t   input_nbytes = ggml_nbytes(tensors[0]);

    // BF16 round-trip: F32 inputs >= bf16_threshold are converted to BF16 for
    // the reduction (chunked or copyengine), halving on-wire bytes. Matches
    // NCCL's behaviour. The pre-conversion zeroes inactive shards so the
    // inner paths see them as already-prepared compute tensors.
    const bool use_bf16 =
        input_type == GGML_TYPE_F32 &&
        p->bf16_threshold > 0 &&
        input_nbytes >= p->bf16_threshold;

    const ggml_type kernel_type = use_bf16 ? GGML_TYPE_BF16 : input_type;
    const size_t    type_size   = ggml_type_size(kernel_type);
    GGML_ASSERT(p->buf_bytes >= type_size);
    const size_t    nbytes      = (size_t) ne * type_size;

    bool compute_flag[GGML_CUDA_MAX_DEVICES] = {};
    for (int i = 0; i < n; ++i) {
        compute_flag[i] = (tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) != 0;
    }

    // Path selection: copy_threshold controls the tuning boundary, but the
    // copy engine path is mandatory once the AR no longer fits in one
    // chunked kernel staging slot.  The kernel splits the AR into n_chunks of
    // HYBRID_CHUNK_BYTES with per-chunk D2H/H2D pipeline sync.  Copyengine
    // handles arbitrary AR sizes via per-chunk ring-buffer reuse of
    // host_large and per-wave reuse of dev_tmp.
    const bool copy_selected_by_threshold =
        p->copy_threshold > 0 &&
        nbytes >= p->copy_threshold;
    const bool copy_required_for_size = nbytes > p->buf_bytes;
    const bool use_copy_engine = copy_selected_by_threshold || copy_required_for_size;

    if (copy_required_for_size && !copy_selected_by_threshold && !p->copy_required_warned) {
        p->copy_required_warned = true;
        GGML_LOG_WARN("%s: internal AllReduce tensor (%zu bytes) exceeds chunked kernel staging "
                      "(%zu bytes); forcing copy engine path despite GGML_CUDA_AR_COPY_THRESHOLD=%zu\n",
                      __func__, nbytes, p->buf_bytes, p->copy_threshold);
    }

    // One-shot diagnostic: log the path + sizing chosen for the FIRST AR call,
    // so it's easy to verify which path is in use without rebuilding.  Subsequent
    // calls don't log to keep stdout uncluttered.
    if (!p->dispatch_logged) {
        p->dispatch_logged = true;
        const char * path = use_copy_engine ? "copyengine" : "chunked kernel";
        const size_t kernel_chunk_max  = p->hybrid_chunk_bytes / type_size;
        const size_t kernel_n_chunks   = (ne + kernel_chunk_max - 1) / kernel_chunk_max;
        fprintf(stderr,
                "ggml_cuda_ar: first AR -- ne=%lld nbytes=%zu use_bf16=%d wire=%s "
                "path=%s copy_threshold=%zu hybrid_chunk_bytes=%zu "
                "(if kernel: chunk_max=%zu n_chunks=%zu)\n",
                (long long) ne, nbytes, (int) use_bf16,
                ggml_type_name(kernel_type), path,
                p->copy_threshold, p->hybrid_chunk_bytes,
                kernel_chunk_max, kernel_n_chunks);
    }

    // BF16 inactive-shard zeroing: when use_bf16 is on, the combined kernel
    // (chunked kernel path) and the combined add kernel (copy_engine path)
    // both accumulate into the F32 tensor data directly, so an inactive
    // shard's accumulator must start at zero.
    if (use_bf16) {
        for (int i = 0; i < n; ++i) {
            if (!compute_flag[i]) {
                auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(backends[i]->context);
                GGML_ASSERT(cuda_ctx->device == p->devices[i]);
                ggml_cuda_set_device(p->devices[i]);
                CUDA_CHECK(cudaMemsetAsync(tensors[i]->data, 0, (size_t) ne * sizeof(float), cuda_ctx->stream()));
            }
        }
    }

    // Pre-convert F32 -> BF16 into bf16_tmp ONLY for the copy_engine + use_bf16
    // path; the chunked kernel path's combined kernel does the conversion
    // inline as it writes to host_buf.
    ggml_cuda_pool_alloc<nv_bfloat16> bf16_tmp[GGML_CUDA_MAX_DEVICES];
    void * copy_src_ptr[GGML_CUDA_MAX_DEVICES] = {};

    if (use_copy_engine && use_bf16) {
        to_bf16_cuda_t to_bf16 = ggml_get_to_bf16_cuda(GGML_TYPE_F32);
        for (int i = 0; i < n; ++i) {
            auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(backends[i]->context);
            GGML_ASSERT(cuda_ctx->device == p->devices[i]);
            bf16_tmp[i].pool = &cuda_ctx->pool();
            bf16_tmp[i].alloc(ne);
            ggml_cuda_set_device(p->devices[i]);
            if (compute_flag[i]) {
                to_bf16(tensors[i]->data, bf16_tmp[i].get(), ne, cuda_ctx->stream());
                CUDA_CHECK(cudaGetLastError());
            } else {
                CUDA_CHECK(cudaMemsetAsync(bf16_tmp[i].get(), 0, nbytes, cuda_ctx->stream()));
            }
            copy_src_ptr[i] = bf16_tmp[i].get();
        }
    }

    bool ok = true;
    if (use_copy_engine) {
        // After up-front BF16 conversion, the tmp buffers already hold the
        // (possibly zeroed-for-inactive) data, so the inner path can treat
        // every shard as compute.
        bool inner_compute[GGML_CUDA_MAX_DEVICES];
        for (int i = 0; i < n; ++i) {
            inner_compute[i] = use_bf16 ? true : compute_flag[i];
        }

        // Dispatch into copy_impl with explicit src/dst types.  When use_bf16
        // is on, the wire type is BF16 (src = bf16_tmp) and the accumulator
        // is F32 (dst = tensors[i]->data); the combined add kernel rounds dst
        // through BF16 for bit-equivalence and writes F32 directly, so no
        // post-conversion is needed.  Otherwise src == dst (same native type).
        if (use_bf16) {
            GGML_ASSERT(kernel_type == GGML_TYPE_BF16);
            nv_bfloat16 * src[GGML_CUDA_MAX_DEVICES] = {};
            float       * dst[GGML_CUDA_MAX_DEVICES] = {};
            for (int i = 0; i < n; ++i) {
                src[i] = static_cast<nv_bfloat16 *>(copy_src_ptr[i]);
                dst[i] = static_cast<float *>(tensors[i]->data);
            }
            ok = ggml_cuda_ar_allreduce_copy_impl<nv_bfloat16, float>(
                p, backends, src, dst, inner_compute, ne, (size_t) ne * sizeof(nv_bfloat16));
        } else {
            switch (kernel_type) {
                case GGML_TYPE_F32: {
                    float * buf[GGML_CUDA_MAX_DEVICES] = {};
                    for (int i = 0; i < n; ++i) {
                        buf[i] = static_cast<float *>(tensors[i]->data);
                    }
                    ok = ggml_cuda_ar_allreduce_copy_impl<float, float>(
                        p, backends, buf, buf, inner_compute, ne, (size_t) ne * sizeof(float));
                    break;
                }
                case GGML_TYPE_BF16: {
                    nv_bfloat16 * buf[GGML_CUDA_MAX_DEVICES] = {};
                    for (int i = 0; i < n; ++i) {
                        buf[i] = static_cast<nv_bfloat16 *>(tensors[i]->data);
                    }
                    ok = ggml_cuda_ar_allreduce_copy_impl<nv_bfloat16, nv_bfloat16>(
                        p, backends, buf, buf, inner_compute, ne, (size_t) ne * sizeof(nv_bfloat16));
                    break;
                }
                case GGML_TYPE_F16: {
                    half * buf[GGML_CUDA_MAX_DEVICES] = {};
                    for (int i = 0; i < n; ++i) {
                        buf[i] = static_cast<half *>(tensors[i]->data);
                    }
                    ok = ggml_cuda_ar_allreduce_copy_impl<half, half>(
                        p, backends, buf, buf, inner_compute, ne, (size_t) ne * sizeof(half));
                    break;
                }
                default:
                    GGML_ASSERT(false);
            }
        }
    } else {
        // Kernel path: single launch per device.  The kernel internally splits
        // the AR into n_chunks of HYBRID_CHUNK_BYTES and uses per-chunk
        // pipeline sync; n_chunks=1 is the small-AR fast path.  Runs entirely
        // on the caller's compute stream -- AR is a barrier here, so same-
        // stream ordering subsumes the cross-stream event handshake that the
        // copyengine path needs and avoids its scheduling overhead.  ev.ker
        // is recorded for two purposes: end-of-AR sync against subsequent
        // compute work, and slot-reuse protection (cross-device wait below).
        const int    chunk_max       = (int) (p->hybrid_chunk_bytes / type_size);
        const int    n_chunks        = (int) ((ne + chunk_max - 1) / chunk_max);
        const size_t input_type_size = ggml_type_size(input_type);

        const auto [slot, token] = ggml_cuda_ar_acquire_slot(p);

        // Slot-reuse fence: peer's prior AR for this slot must finish reading
        // our host_buf[i][slot] before we launch the new AR that overwrites
        // it.  Pre-recorded at init so the first POOL_SIZE ARs see an already-
        // fired event (no actual wait).  Stream-side wait, not host-blocking.
        //
        // BOTH waits must happen BEFORE any of the new AR's kernels record
        // their ev.ker.  If we interleaved wait/launch/record per device, the
        // second device's wait would see the FIRST device's just-recorded
        // event (the in-flight new AR) instead of the prior occupant -- a
        // circular dependency with the in-kernel peer signal -> deadlock.
        for (int i = 0; i < n; ++i) {
            const int peer = 1 - i;
            auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(backends[i]->context);
            GGML_ASSERT(cuda_ctx->device == p->devices[i]);
            ggml_cuda_set_device(p->devices[i]);
            CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx->stream(), p->ev_pool[peer][slot].ker, 0));
        }

        for (int i = 0; i < n; ++i) {
            const int peer = 1 - i;  // valid for n == 2 only
            ggml_cuda_set_device(p->devices[i]);
            auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(backends[i]->context);
            GGML_ASSERT(cuda_ctx->device == p->devices[i]);
            cudaStream_t stream = cuda_ctx->stream();

            char * data = static_cast<char *>(tensors[i]->data);

            // Match NCCL/meta-backend semantics: inactive shards contribute
            // zeros.  BF16 path has already zeroed the tensor up-front above.
            if (!compute_flag[i] && !use_bf16) {
                CUDA_CHECK(cudaMemsetAsync(data, 0, (size_t) ne * input_type_size, stream));
            }

#define LAUNCH_AR_KERNEL(T_dst, T_wire) \
            ggml_cuda_ar_kernel<T_dst, T_wire><<<dim3(GGML_CUDA_AR_KERNEL_BLOCKS), dim3(256), 0, stream>>>( \
                reinterpret_cast<const T_dst *>(data), \
                reinterpret_cast<T_dst *>(data), \
                reinterpret_cast<T_wire *>(p->host_buf[i].dev    + (size_t) slot * p->buf_bytes), \
                reinterpret_cast<const T_wire *>(p->host_buf[peer].dev + (size_t) slot * p->buf_bytes), \
                static_cast<int>(ne), \
                chunk_max, \
                n_chunks, \
                ggml_cuda_ar_arrival_ptr(p, slot, i), \
                ggml_cuda_ar_arrival_ptr(p, slot, peer), \
                token)

            if (use_bf16) {
                GGML_ASSERT(input_type == GGML_TYPE_F32);
                LAUNCH_AR_KERNEL(float, nv_bfloat16);
            } else {
                switch (input_type) {
                    case GGML_TYPE_F32:  LAUNCH_AR_KERNEL(float,         float);         break;
                    case GGML_TYPE_F16:  LAUNCH_AR_KERNEL(half,          half);          break;
                    case GGML_TYPE_BF16: LAUNCH_AR_KERNEL(nv_bfloat16, nv_bfloat16); break;
                    default: GGML_ASSERT(false);
                }
            }

#undef LAUNCH_AR_KERNEL
            CUDA_CHECK(cudaGetLastError());

            CUDA_CHECK(cudaEventRecord(p->ev_pool[i][slot].ker, stream));
        }
    }

    return ok;
}

#else // defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)

// HIP and MUSA lack the host-mapped pinned-memory APIs (cudaHostAllocPortable
// / cudaHostAllocMapped / cudaHostGetDevicePointer) and __nanosleep that this
// implementation relies on, so the internal AllReduce is a CUDA-only feature.
// The dispatcher in ggml-cuda.cu treats a nullptr pipeline as "init failed"
// and silently falls back to the meta backend's generic AllReduce.
ggml_cuda_ar_pipeline * ggml_cuda_ar_pipeline_init(const int *, size_t) {
    return nullptr;
}
void ggml_cuda_ar_pipeline_free(ggml_cuda_ar_pipeline *) {
}
bool ggml_cuda_ar_allreduce(ggml_cuda_ar_pipeline *, ggml_backend_t *, ggml_tensor **) {
    return false;
}

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
