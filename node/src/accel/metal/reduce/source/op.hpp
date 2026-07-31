#pragma once

#include "../local.hpp"

namespace rund::node::accel::detail {

[[nodiscard]] inline const char *
MetalReduceOpName(const rund::kernel::ReduceOp op) noexcept {
  if (op == rund::kernel::ReduceOp::CountNonzero) {
    return "count_nonzero";
  }
  if (op == rund::kernel::ReduceOp::Min) {
    return "min";
  }
  if (op == rund::kernel::ReduceOp::Max) {
    return "max";
  }
  return "sum";
}

[[nodiscard]] inline const char *
MetalReduceInitU32(const rund::kernel::ReduceOp op,
                   const bool signed_domain) noexcept {
  if (op == rund::kernel::ReduceOp::CountNonzero) {
    return "  sums[tid] = index < active_count ? (params.initial_pass != "
           "0u ? (input[params.input_offset + index] != 0u ? 1ul : 0ul) : "
           "ulong(input[params.input_offset + index])) : 0ul;\n";
  }
  if (op == rund::kernel::ReduceOp::Min) {
    return signed_domain
               ? "  sums[tid] = index < active_count ? "
                 "ulong(long(int(input[params.input_offset + index]))) : "
                 "0x7ffffffful;\n"
               : "  sums[tid] = index < active_count ? "
                 "ulong(input[params.input_offset + index]) : 0xfffffffful;\n";
  }
  if (op == rund::kernel::ReduceOp::Max && signed_domain) {
    return "  sums[tid] = index < active_count ? "
           "ulong(long(int(input[params.input_offset + index]))) : "
           "ulong(long(-2147483647 - 1));\n";
  }
  return signed_domain && op == rund::kernel::ReduceOp::Sum
             ? "  sums[tid] = index < active_count ? "
               "ulong(long(int(input[params.input_offset + index]))) : 0ul;\n"
             : "  sums[tid] = index < active_count ? "
               "ulong(input[params.input_offset + index]) : 0ul;\n";
}

[[nodiscard]] inline const char *
MetalReduceInitU64(const rund::kernel::ReduceOp op,
                   const bool signed_domain) noexcept {
  if (op == rund::kernel::ReduceOp::CountNonzero) {
    return "  sums[tid] = index < active_count ? (params.initial_pass != "
           "0u ? (input[params.input_offset + index] != 0ul ? 1ul : 0ul) : "
           "input[params.input_offset + index]) : 0ul;\n";
  }
  if (op == rund::kernel::ReduceOp::Min) {
    return signed_domain
               ? "  sums[tid] = index < active_count ? "
                 "input[params.input_offset + index] : 0x7ffffffffffffffful;\n"
               : "  sums[tid] = index < active_count ? "
                 "input[params.input_offset + index] : ulong(-1);\n";
  }
  if (op == rund::kernel::ReduceOp::Max && signed_domain) {
    return "  sums[tid] = index < active_count ? "
           "input[params.input_offset + index] : 0x8000000000000000ul;\n";
  }
  return "  sums[tid] = index < active_count ? input[params.input_offset "
         "+ index] : 0ul;\n";
}

[[nodiscard]] inline const char *
MetalReduceCombine(const rund::kernel::ReduceOp op, const bool signed_domain,
                   const bool wide) noexcept {
  if (op == rund::kernel::ReduceOp::Min) {
    if (signed_domain && wide) {
      return "      const ulong combined = "
             "as_type<ulong>(min(as_type<long>(sums[tid]), "
             "as_type<long>(rhs)));\n";
    }
    if (signed_domain) {
      return "      const ulong combined = ulong(long(min(int(sums[tid]), "
             "int(rhs))));\n";
    }
    return "      const ulong combined = min(sums[tid], rhs);\n";
  }
  if (op == rund::kernel::ReduceOp::Max) {
    if (signed_domain && wide) {
      return "      const ulong combined = "
             "as_type<ulong>(max(as_type<long>(sums[tid]), "
             "as_type<long>(rhs)));\n";
    }
    if (signed_domain) {
      return "      const ulong combined = ulong(long(max(int(sums[tid]), "
             "int(rhs))));\n";
    }
    return "      const ulong combined = max(sums[tid], rhs);\n";
  }
  return "      const ulong combined = sums[tid] + rhs;\n";
}

[[nodiscard]] inline const char *
MetalReduceOverflowU32(const rund::kernel::ReduceOp op,
                       const bool signed_domain) noexcept {
  if (op == rund::kernel::ReduceOp::Min || op == rund::kernel::ReduceOp::Max) {
    return "";
  }
  return signed_domain
             ? "      if (long(combined) < long(-2147483647 - 1) || "
               "long(combined) > 2147483647l) { overflows[tid] = 1u; }\n"
             : "      if (combined < sums[tid] || combined > 0xfffffffful) { "
               "overflows[tid] = 1u; }\n";
}

[[nodiscard]] inline const char *
MetalReduceOverflowU64(const rund::kernel::ReduceOp op,
                       const bool signed_domain) noexcept {
  if (op == rund::kernel::ReduceOp::Min || op == rund::kernel::ReduceOp::Max) {
    return "";
  }
  if (!signed_domain) {
    return "      if (combined < sums[tid]) { overflows[tid] = 1u; }\n";
  }
  if (op == rund::kernel::ReduceOp::CountNonzero) {
    return "      if ((combined & 0x8000000000000000ul) != 0ul) { "
           "overflows[tid] = 1u; }\n";
  }
  return "      if ((((sums[tid] ^ rhs) & 0x8000000000000000ul) == 0ul) && "
         "(((sums[tid] ^ combined) & 0x8000000000000000ul) != 0ul)) { "
         "overflows[tid] = 1u; }\n";
}

} // namespace rund::node::accel::detail
