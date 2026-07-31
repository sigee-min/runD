#pragma once

#include <kernel/program/compute/lowering/names.hpp>
#include <new>
#include <stdexcept>
#include <utility>

namespace rund::kernel {
namespace compute_lowering_detail {

class Reader {
public:
  explicit Reader(const std::vector<u8> &bytes) noexcept : bytes_(&bytes) {}

  [[nodiscard]] bool read_u8(u8 &value) noexcept {
    if (remaining() < 1u) {
      return false;
    }
    value = (*bytes_)[offset_];
    ++offset_;
    return true;
  }

  [[nodiscard]] bool read_u32(u32 &value) noexcept {
    if (remaining() < 4u) {
      return false;
    }
    value = static_cast<u32>((*bytes_)[offset_]) |
            (static_cast<u32>((*bytes_)[offset_ + 1u]) << 8u) |
            (static_cast<u32>((*bytes_)[offset_ + 2u]) << 16u) |
            (static_cast<u32>((*bytes_)[offset_ + 3u]) << 24u);
    offset_ += 4u;
    return true;
  }

  [[nodiscard]] bool read_bytes(std::vector<u8> &out) {
    u32 size = 0u;
    if (!read_u32(size) || remaining() < size) {
      return false;
    }
    out.assign(bytes_->begin() + static_cast<std::ptrdiff_t>(offset_),
               bytes_->begin() + static_cast<std::ptrdiff_t>(offset_ + size));
    offset_ += size;
    return true;
  }

  [[nodiscard]] bool read_string(std::string &out) {
    u32 size = 0u;
    if (!read_u32(size) || remaining() < size) {
      return false;
    }
    out.assign(reinterpret_cast<const char *>(bytes_->data() + offset_), size);
    offset_ += size;
    return true;
  }

  [[nodiscard]] bool done() const noexcept {
    return bytes_ != nullptr && offset_ == bytes_->size();
  }

  [[nodiscard]] std::size_t remaining() const noexcept {
    return bytes_ == nullptr || offset_ > bytes_->size()
               ? 0u
               : bytes_->size() - offset_;
  }

private:
  const std::vector<u8> *bytes_ = nullptr;
  std::size_t offset_ = 0u;
};

[[nodiscard]] inline bool BindingIs(const ParsedIR &parsed, const u32 index,
                                    const u8 kind) noexcept {
  return index < parsed.bindings.size() && parsed.bindings[index].kind == kind;
}

[[nodiscard]] inline bool DuplicateBindingName(const ParsedIR &parsed,
                                               const ParsedBinding &binding) {
  for (const ParsedBinding &existing : parsed.bindings) {
    if (existing.kind == binding.kind && existing.name == binding.name) {
      return true;
    }
  }
  return false;
}

[[nodiscard]] inline bool ValidNodeRef(const u32 node,
                                       const u32 current_node) noexcept {
  return node != 0u && node < current_node;
}

[[nodiscard]] inline ParsedIR RejectParsed(const char *const reason) {
  ParsedIR parsed{};
  parsed.reason = reason;
  return parsed;
}

struct ParsedBindingRead {
  ParsedBinding binding{};
  bool ok = false;
  const char *reason = "compute_ir_malformed";
};

[[nodiscard]] inline ParsedBindingRead ReadParsedBinding(Reader &reader) {
  ParsedBinding binding{};
  u8 floating_point = 0u;
  if (!reader.read_u8(binding.kind) || !reader.read_u8(binding.numeric_mode) ||
      !reader.read_string(binding.name) ||
      !reader.read_u32(binding.element_bytes) ||
      !reader.read_u8(floating_point) ||
      !reader.read_bytes(binding.value_bytes)) {
    return ParsedBindingRead{};
  }
  binding.floating_point_param = floating_point != 0u;
  return ParsedBindingRead{
      .binding = std::move(binding), .ok = true, .reason = "ok"};
}

[[nodiscard]] inline const char *
ValidateParsedBinding(const ParsedIR &parsed, const ParsedBinding &binding) {
  if (BindingKindName(binding.kind) == nullptr || binding.numeric_mode < 1u ||
      binding.numeric_mode > 6u || binding.element_bytes == 0u) {
    return "compute_ir_binding_invalid";
  }
  if (binding.kind == kParamBindingKind && binding.floating_point_param) {
    return "compute_param_float_unsupported";
  }
  if (binding.kind == kParamBindingKind &&
      binding.value_bytes.size() != binding.element_bytes) {
    return "compute_ir_param_size_mismatch";
  }
  if (binding.kind != kParamBindingKind && !binding.value_bytes.empty()) {
    return "compute_ir_binding_payload_invalid";
  }
  if (DuplicateBindingName(parsed, binding)) {
    return "compute_ir_binding_duplicate";
  }
  return nullptr;
}

[[nodiscard]] inline const char *AppendParsedBinding(Reader &reader,
                                                     ParsedIR &parsed) {
  ParsedBindingRead read = ReadParsedBinding(reader);
  if (!read.ok) {
    return read.reason;
  }
  if (const char *const reason = ValidateParsedBinding(parsed, read.binding);
      reason != nullptr) {
    return reason;
  }
  parsed.bindings.push_back(std::move(read.binding));
  return nullptr;
}

struct ParsedNodeRead {
  ParsedNode node{};
  bool ok = false;
  const char *reason = "compute_ir_malformed";
};

[[nodiscard]] inline ParsedNodeRead ReadParsedNode(Reader &reader) {
  ParsedNode node{};
  u8 rounding = 0u;
  u8 overflow = 0u;
  u8 approximation = 0u;
  if (!reader.read_u8(node.op) || !reader.read_u32(node.lhs) ||
      !reader.read_u32(node.rhs) || !reader.read_u32(node.aux) ||
      !reader.read_u8(node.fixed_format.integer_bits) ||
      !reader.read_u8(node.fixed_format.fraction_bits) ||
      !reader.read_u8(rounding) || !reader.read_u8(overflow) ||
      !reader.read_u8(approximation)) {
    return ParsedNodeRead{};
  }
  node.fixed_format.rounding = static_cast<ComputeRounding>(rounding);
  node.fixed_format.overflow = static_cast<ComputeOverflow>(overflow);
  node.fixed_format.approximation =
      static_cast<ComputeApproximation>(approximation);
  if (OpName(node.op) == nullptr) {
    return ParsedNodeRead{.node = node, .reason = "compute_ir_op_unsupported"};
  }
  if (!ComputeIntermediateFormatValid(node.fixed_format)) {
    return ParsedNodeRead{.node = node,
                          .reason = "compute_ir_numeric_policy_invalid"};
  }
  return ParsedNodeRead{.node = node, .ok = true, .reason = "ok"};
}

[[nodiscard]] inline const char *
ValidateParsedNodeOperands(const ParsedIR &parsed, const ParsedNode &node,
                           const ComputeScalar scalar,
                           const u32 current_node) noexcept {
  switch (static_cast<IrOp>(node.op)) {
  case IrOp::Param:
    if (node.lhs != 0u || node.rhs != 0u ||
        !BindingIs(parsed, node.aux, kParamBindingKind)) {
      return "compute_ir_node_invalid";
    }
    return nullptr;
  case IrOp::Read:
    if (node.lhs != 0u || node.rhs != 0u ||
        !BindingIs(parsed, node.aux, kReadBindingKind)) {
      return "compute_ir_node_invalid";
    }
    return nullptr;
  case IrOp::ReadAt:
    if (!BindingIs(parsed, node.lhs, kReadBindingKind) ||
        !BindingIs(parsed, node.aux, kReadBindingKind) ||
        node.lhs == node.aux || node.rhs == 0u ||
        parsed.bindings[node.lhs].numeric_mode != 4u ||
        parsed.bindings[node.lhs].element_bytes != sizeof(u32)) {
      return "compute_ir_node_invalid";
    }
    return nullptr;
  case IrOp::Constant:
    if (node.aux != 0u || (scalar == ComputeScalar::Lane32 && node.rhs != 0u)) {
      return "compute_ir_node_invalid";
    }
    return nullptr;
  case IrOp::Index:
    if (node.lhs != 0u || node.rhs != 0u || node.aux != 0u) {
      return "compute_ir_node_invalid";
    }
    return nullptr;
  case IrOp::Write:
    if (!ValidNodeRef(node.lhs, current_node) || !IrWriteModeValid(node.rhs) ||
        !BindingIs(parsed, node.aux, kWriteBindingKind)) {
      return "compute_ir_node_invalid";
    }
    if (parsed.scalar_mode == ScalarModeFor(scalar) &&
        node.rhs != static_cast<u32>(IrWriteMode::Value)) {
      return "compute_ir_node_invalid";
    }
    if (parsed.scalar_mode == ScalarModeFor(scalar) &&
        static_cast<IrOp>(parsed.nodes[node.lhs - 1u].op) != IrOp::Quantize) {
      return "compute_ir_quantize_required";
    }
    return nullptr;
  case IrOp::Neg:
  case IrOp::Abs:
  case IrOp::AbsMagnitude:
  case IrOp::Sign:
  case IrOp::PredicateNot:
  case IrOp::BitNot:
  case IrOp::NegPositiveFixed:
  case IrOp::Recip:
  case IrOp::Sqrt:
  case IrOp::Rsqrt:
  case IrOp::Sin:
  case IrOp::Cos:
  case IrOp::Tan:
  case IrOp::Exp:
  case IrOp::Log:
  case IrOp::Quantize:
    if (!ValidNodeRef(node.lhs, current_node) || node.rhs != 0u ||
        node.aux != 0u) {
      return "compute_ir_node_invalid";
    }
    return nullptr;
  case IrOp::Add:
  case IrOp::Sub:
  case IrOp::Mul:
  case IrOp::MulWrap:
  case IrOp::Min:
  case IrOp::Max:
  case IrOp::Eq:
  case IrOp::Lt:
  case IrOp::Le:
  case IrOp::Ne:
  case IrOp::Gt:
  case IrOp::Ge:
  case IrOp::PredicateAnd:
  case IrOp::PredicateOr:
  case IrOp::BitAnd:
  case IrOp::BitOr:
  case IrOp::BitXor:
  case IrOp::AddSat:
  case IrOp::AddSatUnsigned:
  case IrOp::SubSat:
  case IrOp::MulFixed:
  case IrOp::MulFixedScaled:
  case IrOp::MulUnsignedFixed:
  case IrOp::DivFixed:
  case IrOp::Atan2:
  case IrOp::DivSigned:
  case IrOp::DivUnsigned:
  case IrOp::MinUnsigned:
  case IrOp::MaxUnsigned:
  case IrOp::LtUnsigned:
  case IrOp::LeUnsigned:
  case IrOp::GtUnsigned:
  case IrOp::GeUnsigned:
    if (!ValidNodeRef(node.lhs, current_node) ||
        !ValidNodeRef(node.rhs, current_node) || node.aux != 0u) {
      return "compute_ir_node_invalid";
    }
    return nullptr;
  case IrOp::ShlConst:
  case IrOp::ShrLogicalConst:
  case IrOp::ShrArithmeticConst:
    if (!ValidNodeRef(node.lhs, current_node) || node.rhs != 0u) {
      return "compute_ir_node_invalid";
    }
    return node.aux >= ScalarBitWidth(scalar) ? "compute_shift_count_invalid"
                                              : nullptr;
  case IrOp::Clamp:
  case IrOp::ClampUnsigned:
  case IrOp::Select:
  case IrOp::MulAddFixed:
    if (!ValidNodeRef(node.lhs, current_node) ||
        !ValidNodeRef(node.rhs, current_node) ||
        !ValidNodeRef(node.aux, current_node)) {
      return "compute_ir_node_invalid";
    }
    return nullptr;
  }
  return "compute_ir_op_unsupported";
}

[[nodiscard]] inline const char *
AppendParsedNode(Reader &reader, ParsedIR &parsed, const ComputeScalar scalar,
                 const u32 index, u32 &write_count) {
  ParsedNodeRead read = ReadParsedNode(reader);
  if (!read.ok) {
    return read.reason;
  }
  const bool nonfixed_boundary_format =
      static_cast<IrOp>(read.node.op) == IrOp::Write &&
      read.node.rhs == static_cast<u32>(IrWriteMode::BoundaryMask);
  if (parsed.scalar_mode != ScalarModeFor(scalar) &&
      !ComputeFixedFormatAbsent(read.node.fixed_format) &&
      !nonfixed_boundary_format) {
    return "compute_ir_numeric_policy_mismatch";
  }
  const u32 current_node = index + 1u;
  if (const char *const reason =
          ValidateParsedNodeOperands(parsed, read.node, scalar, current_node);
      reason != nullptr) {
    return reason;
  }
  if (static_cast<IrOp>(read.node.op) == IrOp::Write) {
    ++write_count;
  }
  parsed.nodes.push_back(read.node);
  return nullptr;
}

[[nodiscard]] inline ParsedIR
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

[[nodiscard]] inline ParsedIR GuardComputeIRParse(auto &&parse) {
  try {
    return std::forward<decltype(parse)>(parse)();
  } catch (const std::bad_alloc &) {
    return RejectParsed("compute_ir_capacity");
  } catch (const std::length_error &) {
    return RejectParsed("compute_ir_capacity");
  }
}

[[nodiscard]] inline ParsedIR
ParseComputeIR(const ComputeIR &ir, const std::vector<u8> &canonical_bytes) {
  return GuardComputeIRParse(
      [&]() { return ParseComputeIRUnchecked(ir, canonical_bytes); });
}

[[nodiscard]] inline ParsedIR ParseComputeIR(const ComputeIR &ir) {
  return ParseComputeIR(ir, ir.canonical_bytes);
}

} // namespace compute_lowering_detail

} // namespace rund::kernel
