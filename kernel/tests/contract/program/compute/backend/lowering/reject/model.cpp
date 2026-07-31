#include "contract/program/compute/backend/lowering/reject/model.hpp"

namespace program_compute_contract::lowering_reject {

namespace {

void AppendBinding(std::vector<rund::kernel::u8> &bytes,
                   const rund::kernel::u8 kind, const std::string_view name,
                   const IntegerDomain mode) {
  using namespace backend_lowering_support;
  AppendU8(bytes, kind);
  AppendU8(bytes, rund::kernel::compute_lowering_detail::DomainModeFor(
                      mode.scalar, mode.domain));
  AppendBytes(bytes, name);
  AppendU32(bytes,
            mode.scalar == rund::kernel::ComputeScalar::Lane64 ? 8u : 4u);
  AppendU8(bytes, 0u);
  AppendU32(bytes, 0u);
}

} // namespace

rund::kernel::ComputeIR IntegerIr(const rund::kernel::IrOp op,
                                  const rund::kernel::u32 arity,
                                  const IntegerDomain mode) {
  using namespace backend_lowering_support;
  std::vector<rund::kernel::u8> bytes;
  AppendBytes(bytes, "rund.compute.ir");
  AppendBytes(bytes, "forged-integer-domain-op");
  AppendU8(bytes, rund::kernel::compute_lowering_detail::DomainModeFor(
                      mode.scalar, mode.domain));
  AppendIntegerNumericPolicy(bytes);
  AppendU32(bytes, arity + 1u);
  constexpr std::array names{"first", "second", "third"};
  for (rund::kernel::u32 operand = 0u; operand < arity; ++operand) {
    AppendBinding(bytes, 2u, names[operand], mode);
  }
  AppendBinding(bytes, 3u, "output", mode);

  AppendU32(bytes, arity + 2u);
  for (rund::kernel::u32 operand = 0u; operand < arity; ++operand) {
    AppendNode(bytes, rund::kernel::IrOp::Read, 0u, 0u, operand);
  }
  AppendNode(bytes, op, 1u, arity >= 2u ? 2u : 0u, arity >= 3u ? 3u : 0u);
  AppendNode(bytes, rund::kernel::IrOp::Write, arity + 1u, 0u, arity);
  return RehashIr(rund::kernel::ComputeIR{
      .scalar = mode.scalar,
      .domain = mode.domain,
      .canonical_bytes = std::move(bytes),
      .ok = true,
      .reason = "ok",
  });
}

bool SetNode(std::vector<rund::kernel::u8> &bytes,
             const rund::kernel::u32 one_based_node,
             const rund::kernel::IrOp op, const rund::kernel::u32 lhs,
             const rund::kernel::u32 rhs,
             const rund::kernel::u32 aux) noexcept {
  using namespace backend_lowering_support;
  std::size_t offset = 0u;
  if (one_based_node == 0u || !SkipLengthPrefixedBytes(bytes, offset) ||
      !SkipLengthPrefixedBytes(bytes, offset) ||
      !SkipBytes(bytes, offset, 6u)) {
    return false;
  }
  rund::kernel::u32 binding_count = 0u;
  if (!ReadU32(bytes, offset, binding_count)) {
    return false;
  }
  for (rund::kernel::u32 binding = 0u; binding < binding_count; ++binding) {
    if (!SkipBytes(bytes, offset, 2u) ||
        !SkipLengthPrefixedBytes(bytes, offset) ||
        !SkipBytes(bytes, offset, 4u) || !SkipBytes(bytes, offset, 1u) ||
        !SkipLengthPrefixedBytes(bytes, offset)) {
      return false;
    }
  }
  rund::kernel::u32 node_count = 0u;
  constexpr std::size_t kNodeBytes = 18u;
  if (!ReadU32(bytes, offset, node_count) || one_based_node > node_count ||
      node_count > (bytes.size() - offset) / kNodeBytes) {
    return false;
  }
  offset += static_cast<std::size_t>(one_based_node - 1u) * kNodeBytes;
  if (offset >= bytes.size()) {
    return false;
  }
  bytes[offset] = static_cast<rund::kernel::u8>(op);
  return WriteU32At(bytes, offset + 1u, lhs) &&
         WriteU32At(bytes, offset + 5u, rhs) &&
         WriteU32At(bytes, offset + 9u, aux);
}

bool SetFirstMode(std::vector<rund::kernel::u8> &bytes,
                  const rund::kernel::u8 mode) noexcept {
  using namespace backend_lowering_support;
  std::size_t offset = 0u;
  rund::kernel::u32 count = 0u;
  if (!SkipLengthPrefixedBytes(bytes, offset) ||
      !SkipLengthPrefixedBytes(bytes, offset) ||
      !SkipBytes(bytes, offset, 6u) || !ReadU32(bytes, offset, count) ||
      count == 0u || offset + 1u >= bytes.size()) {
    return false;
  }
  bytes[offset + 1u] = mode;
  return true;
}

bool SetMode(std::vector<rund::kernel::u8> &bytes,
             const rund::kernel::u32 target,
             const rund::kernel::u8 mode) noexcept {
  using namespace backend_lowering_support;
  std::size_t offset = 0u;
  rund::kernel::u32 count = 0u;
  if (!SkipLengthPrefixedBytes(bytes, offset) ||
      !SkipLengthPrefixedBytes(bytes, offset) ||
      !SkipBytes(bytes, offset, 6u) || !ReadU32(bytes, offset, count) ||
      target >= count) {
    return false;
  }
  for (rund::kernel::u32 binding = 0u; binding < count; ++binding) {
    if (offset + 1u >= bytes.size()) {
      return false;
    }
    if (binding == target) {
      bytes[offset + 1u] = mode;
      return true;
    }
    if (!SkipBytes(bytes, offset, 2u) ||
        !SkipLengthPrefixedBytes(bytes, offset) ||
        !SkipBytes(bytes, offset, 4u) || !SkipBytes(bytes, offset, 1u) ||
        !SkipLengthPrefixedBytes(bytes, offset)) {
      return false;
    }
  }
  return false;
}

rund::kernel::u32
FindNode(const rund::kernel::compute_lowering_detail::ParsedIR &parsed,
         const rund::kernel::IrOp op) noexcept {
  for (std::size_t index = 0u; index < parsed.nodes.size(); ++index) {
    if (static_cast<rund::kernel::IrOp>(parsed.nodes[index].op) == op) {
      return static_cast<rund::kernel::u32>(index + 1u);
    }
  }
  return 0u;
}

} // namespace program_compute_contract::lowering_reject
