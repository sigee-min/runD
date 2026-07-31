#include "../../domain.hpp"
#include "local.hpp"
#include "source/op.hpp"
#include "source/wide.hpp"

#include <kernel/program/compute/reduce/wide.hpp>

#include <string>
#include <string_view>

namespace rund::node::accel::detail {
namespace {

void AppendCanonicalWideReduce(std::string &source,
                               const rund::kernel::ReduceOp op,
                               const char *const op_name,
                               const char *const type, const char *const suffix,
                               const bool signed_domain) {
  source += "kernel void rund_compute_reduce_";
  source += op_name;
  source += "_";
  source += suffix;
  source += "(device const ";
  source += type;
  source += R"MSL(* input [[buffer(0)]],
    device RundWide* partial [[buffer(1)]],
    device )MSL";
  source += type;
  source += R"MSL(* output [[buffer(2)]],
    device atomic_uint* status [[buffer(3)]],
    constant ReduceParams& params [[buffer(4)]],
    device const uint* logical_count [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
  threadgroup RundWide sums[RUND_REDUCE_BLOCK_SIZE];
  RundWide acc = rund_wide_zero();
  if (params.initial_pass != 0u) {
    const ulong resident_count = params.count_words == 2u
        ? (ulong(logical_count[1]) << 32u) | ulong(logical_count[0])
        : (params.count_words == 1u ? ulong(logical_count[0]) : params.input_count);
    const ulong active_count = min(resident_count, params.input_count);
    if (resident_count > params.input_count && tid == 0u && group == 0u) {
      atomic_fetch_or_explicit(&status[0], 2u, memory_order_relaxed);
    }
    const ulong stride = params.grid_size * ulong(RUND_REDUCE_BLOCK_SIZE);
)MSL";
  if (op == rund::kernel::ReduceOp::CountNonzero) {
    source += R"MSL(    RundPair count = rund_pair_zero();
    for (ulong index = ulong(group) * ulong(RUND_REDUCE_BLOCK_SIZE) + ulong(tid);
         index < active_count; index += stride) {
      count = rund_pair_add(count, input[index] != 0 ? 1u : 0u, 0u);
    }
    acc = rund_wide_pair_unsigned(count);
)MSL";
  } else if (std::string_view{suffix}.ends_with("32")) {
    source +=
        R"MSL(    for (ulong index = ulong(group) * ulong(RUND_REDUCE_BLOCK_SIZE) + ulong(tid);
         index < active_count;) {
)MSL";
    source += "      RundPair narrow = rund_pair_zero();\n";
    source += R"MSL(      uint items = 0u;
      do {
)MSL";
    source +=
        signed_domain
            ? "        narrow = rund_pair_add(narrow, uint(input[index]), "
              "input[index] < 0 ? 0xffffffffu : 0u);\n"
            : "        narrow = rund_pair_add(narrow, uint(input[index]), "
              "0u);\n";
    source += R"MSL(        index += stride;
        ++items;
      } while (items < RUND_REDUCE_NARROW_CHUNK && index < active_count);
      acc = rund_wide_add(acc, )MSL";
    source += signed_domain ? "rund_wide_pair_signed(narrow));\n"
                            : "rund_wide_pair_unsigned(narrow));\n";
    source += "    }\n";
  } else {
    source +=
        R"MSL(    for (ulong index = ulong(group) * ulong(RUND_REDUCE_BLOCK_SIZE) + ulong(tid);
         index < active_count; index += stride) {
      acc = rund_wide_add(acc, rund_wide_)MSL";
    source += signed_domain ? "i" : "u";
    source += "64";
    source += "(input[index]));\n";
    source += "    }\n";
  }
  source += R"MSL(  } else {
    for (ulong index = ulong(tid); index < params.input_count;
         index += ulong(RUND_REDUCE_BLOCK_SIZE)) {
      acc = rund_wide_add(acc, partial[index]);
    }
  }
  sums[tid] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint width = RUND_REDUCE_BLOCK_SIZE;
  while (width > 1u) {
    const uint next = (width + 1u) >> 1u;
    if (tid < width - next) {
      sums[tid] = rund_wide_add(sums[tid], sums[tid + next]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    width = next;
  }
  if (tid != 0u) { return; }
  acc = sums[0];
  if (params.final_pass == 0u) {
    partial[ulong(group)] = acc;
    return;
  }
  if (!rund_wide_fits_)MSL";
  source += signed_domain ? "i" : "u";
  source += std::string_view{suffix}.ends_with("64") ? "64" : "32";
  source += R"MSL((acc)) {
    atomic_fetch_or_explicit(&status[0], 1u, memory_order_relaxed);
  }
  output[0] = )MSL";
  source += type;
  source += std::string_view{suffix}.ends_with("64")
                ? "(rund_wide_low64(acc));\n}\n"
                : "(rund_wide_low32(acc));\n}\n";
}

[[nodiscard]] std::string
MetalCanonicalWideReduceSource(const rund::kernel::ReduceOp op,
                               const bool signed_domain,
                               const rund::kernel::u64 block_size) {
  const char *const op_name = MetalReduceOpName(op);
  std::string source = R"MSL(
#include <metal_stdlib>
using namespace metal;
#define RUND_REDUCE_BLOCK_SIZE )MSL";
  source += std::to_string(block_size);
  source += "\n#define RUND_REDUCE_NARROW_CHUNK ";
  source += std::to_string(rund::kernel::kReduceNarrowChunkItems);
  source += R"MSL(
#define RUND_REDUCE_I64 long
#define RUND_REDUCE_U64 ulong
struct ReduceParams {
  ulong input_offset;
  ulong output_offset;
  ulong input_count;
  ulong grid_size;
  uint final_pass;
  uint initial_pass;
  uint count_words;
};
)MSL";
  source.append(rund::kernel::ReduceWideSource);
  AppendCanonicalWideReduce(source, op, op_name, signed_domain ? "int" : "uint",
                            "u32", signed_domain);
  AppendCanonicalWideReduce(source, op, op_name,
                            signed_domain ? "long" : "ulong", "u64",
                            signed_domain);
  return source;
}

} // namespace

std::string MetalReduceSource(const rund::kernel::ReduceOp op,
                              const rund::kernel::u64 block_size,
                              const rund::kernel::ComputeDomain domain) {
  const bool signed_domain = IsSignedDomain(domain);
  if (op == rund::kernel::ReduceOp::Sum ||
      op == rund::kernel::ReduceOp::CountNonzero) {
    return MetalCanonicalWideReduceSource(op, signed_domain, block_size);
  }
  const std::string block = std::to_string(block_size);
  const char *const op_name = MetalReduceOpName(op);
  std::string source = R"MSL(
#include <metal_stdlib>
using namespace metal;
#define RUND_REDUCE_BLOCK_SIZE )MSL";
  source += block;
  source += R"MSL(
struct ReduceParams {
  ulong input_offset;
  ulong output_offset;
  ulong input_count;
  ulong grid_size;
  uint final_pass;
  uint initial_pass;
  uint count_words;
};
kernel void )MSL";
  source += "rund_compute_reduce_";
  source += op_name;
  source += "_u32";
  source += R"MSL((
    device const uint* input [[buffer(0)]],
    device uint* partial [[buffer(1)]],
    device uint* output [[buffer(2)]],
    device atomic_uint* status [[buffer(3)]],
    constant ReduceParams& params [[buffer(4)]],
    device const uint* logical_count [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
  threadgroup ulong sums[RUND_REDUCE_BLOCK_SIZE];
  threadgroup uint overflows[RUND_REDUCE_BLOCK_SIZE];
  const ulong local = ulong(tid);
  const ulong index = ulong(group) * ulong(RUND_REDUCE_BLOCK_SIZE) + local;
  const ulong resident_count = params.count_words == 2u
      ? (ulong(logical_count[1]) << 32u) | ulong(logical_count[0])
      : (params.count_words == 1u ? ulong(logical_count[0]) : params.input_count);
  const ulong active_count = params.initial_pass != 0u
      ? min(resident_count, params.input_count) : params.input_count;
  if (params.initial_pass != 0u && resident_count > params.input_count && tid == 0u && group == 0u) {
    atomic_fetch_or_explicit(&status[0], 2u, memory_order_relaxed);
  }
)MSL";
  if (op == rund::kernel::ReduceOp::Min || op == rund::kernel::ReduceOp::Max) {
    source += "  if (params.initial_pass != 0u && active_count == 0ul && "
              "tid == 0u && group == 0u) { atomic_fetch_or_explicit("
              "&status[0], 4u, memory_order_relaxed); }\n";
  }
  source += MetalReduceInitU32(op, signed_domain);
  source += R"MSL(
  overflows[tid] = 0u;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint width = RUND_REDUCE_BLOCK_SIZE;
  while (width > 1u) {
    const uint next = (width + 1u) >> 1u;
    if (tid < width - next) {
      const ulong rhs = sums[tid + next];
)MSL";
  source += MetalReduceCombine(op, signed_domain, false);
  source += MetalReduceOverflowU32(op, signed_domain);
  source += R"MSL(
      overflows[tid] |= overflows[tid + next];
      sums[tid] = combined;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    width = next;
  }
  if (tid == 0u) {
    if (overflows[0] != 0u) { atomic_fetch_or_explicit(&status[0], 1u, memory_order_relaxed); }
    if (params.final_pass != 0u) { output[0] = uint(sums[0]); }
    else { partial[params.output_offset + ulong(group)] = uint(sums[0]); }
  }
}
kernel void )MSL";
  AppendMetalReduceU64(source, op, signed_domain, op_name);
  return source;
}

} // namespace rund::node::accel::detail
