#include "local.hpp"

#include <new>
#include <stdexcept>
#include <utility>

namespace rund::kernel::compute_lowering_detail {

[[nodiscard]] ParsedIR
ParseComputeIRUnchecked(const ComputeIR &ir,
                        const std::vector<u8> &canonical_bytes) {
  Reader reader{canonical_bytes};
  std::string schema;
  if (!reader.read_string(schema) || schema != "rund.compute.ir") {
    return RejectParsed("compute_ir_malformed");
  }

  ParsedIR parsed{};
  u8 rounding = 0u;
  u8 overflow = 0u;
  u8 approximation = 0u;
  if (!reader.read_string(parsed.name) || !reader.read_u8(parsed.scalar_mode) ||
      !reader.read_u8(parsed.fixed_format.integer_bits) ||
      !reader.read_u8(parsed.fixed_format.fraction_bits) ||
      !reader.read_u8(rounding) || !reader.read_u8(overflow) ||
      !reader.read_u8(approximation)) {
    return RejectParsed("compute_ir_malformed");
  }
  parsed.fixed_format.rounding = static_cast<ComputeRounding>(rounding);
  parsed.fixed_format.overflow = static_cast<ComputeOverflow>(overflow);
  parsed.fixed_format.approximation =
      static_cast<ComputeApproximation>(approximation);
  if (parsed.scalar_mode != DomainModeFor(ir.scalar, ir.domain)) {
    return RejectParsed("compute_ir_scalar_mismatch");
  }
  if (parsed.fixed_format != ir.fixed_format ||
      (ir.domain == ComputeDomain::Fixed
           ? !ComputeFixedFormatValid(ir.scalar, parsed.fixed_format)
           : !ComputeFixedFormatAbsent(parsed.fixed_format))) {
    return RejectParsed("compute_ir_numeric_policy_mismatch");
  }

  u32 binding_count = 0u;
  if (!reader.read_u32(binding_count)) {
    return RejectParsed("compute_ir_malformed");
  }
  if (binding_count > kMaxComputeBindingCount ||
      static_cast<std::size_t>(binding_count) >
          reader.remaining() / kMinBindingBytes) {
    return RejectParsed("compute_ir_binding_count_invalid");
  }
  parsed.bindings.reserve(binding_count);
  for (u32 index = 0u; index < binding_count; ++index) {
    if (const char *const reason = AppendParsedBinding(reader, parsed);
        reason != nullptr) {
      return RejectParsed(reason);
    }
  }

  u32 node_count = 0u;
  if (!reader.read_u32(node_count) || node_count == 0u) {
    return RejectParsed("compute_ir_malformed");
  }
  if (node_count > kMaxComputeNodeCount ||
      static_cast<std::size_t>(node_count) >
          reader.remaining() / kSerializedNodeBytes) {
    return RejectParsed("compute_ir_node_count_invalid");
  }
  parsed.nodes.reserve(node_count);
  u32 write_count = 0u;
  for (u32 index = 0u; index < node_count; ++index) {
    if (const char *const reason =
            AppendParsedNode(reader, parsed, ir.scalar, index, write_count);
        reason != nullptr) {
      return RejectParsed(reason);
    }
  }

  if (!reader.done()) {
    return RejectParsed("compute_ir_malformed");
  }
  if (write_count == 0u) {
    return RejectParsed("compute_write_missing");
  }
  parsed.ok = true;
  parsed.reason = "ok";
  return parsed;
}

namespace {

[[nodiscard]] ParsedIR GuardComputeIRParse(auto &&parse) {
  try {
    return std::forward<decltype(parse)>(parse)();
  } catch (const std::bad_alloc &) {
    return RejectParsed("compute_ir_capacity");
  } catch (const std::length_error &) {
    return RejectParsed("compute_ir_capacity");
  }
}

} // namespace

[[nodiscard]] ParsedIR ParseComputeIR(const ComputeIR &ir,
                                      const std::vector<u8> &canonical_bytes) {
  return GuardComputeIRParse(
      [&]() { return ParseComputeIRUnchecked(ir, canonical_bytes); });
}

[[nodiscard]] ParsedIR ParseComputeIR(const ComputeIR &ir) {
  return ParseComputeIR(ir, ir.canonical_bytes);
}

} // namespace rund::kernel::compute_lowering_detail
