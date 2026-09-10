#include "local.hpp"

#include <utility>

namespace rund::kernel::compute_lowering_detail {

[[nodiscard]] bool BindingIs(const ParsedIR &parsed, const u32 index,
                             const u8 kind) noexcept {
  return index < parsed.bindings.size() && parsed.bindings[index].kind == kind;
}

[[nodiscard]] bool DuplicateBindingName(const ParsedIR &parsed,
                                        const ParsedBinding &binding) {
  for (const ParsedBinding &existing : parsed.bindings) {
    if (existing.kind == binding.kind && existing.name == binding.name) {
      return true;
    }
  }
  return false;
}

[[nodiscard]] bool ValidNodeRef(const u32 node,
                                const u32 current_node) noexcept {
  return node != 0u && node < current_node;
}

[[nodiscard]] ParsedIR RejectParsed(const char *const reason) {
  ParsedIR parsed{};
  parsed.reason = reason;
  return parsed;
}

[[nodiscard]] ParsedBindingRead ReadParsedBinding(Reader &reader) {
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

[[nodiscard]] const char *ValidateParsedBinding(const ParsedIR &parsed,
                                                const ParsedBinding &binding) {
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

[[nodiscard]] const char *AppendParsedBinding(Reader &reader,
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

} // namespace rund::kernel::compute_lowering_detail
