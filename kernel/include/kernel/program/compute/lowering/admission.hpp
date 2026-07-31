#pragma once

#include <kernel/program/compute/lowering/artifact/key.hpp>
#include <kernel/program/compute/lowering/parse.hpp>
#include <kernel/program/compute/lowering/validate.hpp>
#include <kernel/program/compute/retention.hpp>

#include <utility>

namespace rund::kernel::compute_lowering_detail {

// Internal parse owner shared by generic artifact emission and backend
// preparation. ParsedIR never becomes part of the public Compute artifact.
struct ComputeInputAdmission {
  ArtifactKey key{};
  ParsedIR parsed{};
  u32 parse_count = 0u;
  bool ok = false;
  const char *reason = "compute_lowering_invalid";

  [[nodiscard]] u64 retained_dynamic_memory_bytes() const noexcept {
    using compute_retained_detail::Add;
    using compute_retained_detail::StringExternalStorageBytes;
    using compute_retained_detail::VectorCapacityBytes;
    u64 bytes = StringExternalStorageBytes(parsed.name);
    bytes = Add(bytes, VectorCapacityBytes(parsed.bindings));
    for (const ParsedBinding &binding : parsed.bindings) {
      bytes = Add(bytes, StringExternalStorageBytes(binding.name));
      bytes = Add(bytes, VectorCapacityBytes(binding.value_bytes));
    }
    return Add(bytes, VectorCapacityBytes(parsed.nodes));
  }
};

[[nodiscard]] inline ComputeInputAdmission
RejectComputeInput(const ArtifactKey &key, const char *const reason) {
  return ComputeInputAdmission{.key = key, .reason = reason};
}

[[nodiscard]] inline const char *
ComputeInputHeaderFailure(const ComputeIR &ir, const ComputeApi api,
                          const std::vector<u8> &canonical_bytes,
                          const ArtifactKey &key) {
  if (!ComputeApiValid(api)) {
    return "compute_api_unsupported";
  }
  if (!ComputeScalarValid(ir.scalar)) {
    return "compute_scalar_unsupported";
  }
  if (!ir.ok) {
    return ir.reason;
  }
  if (canonical_bytes.empty()) {
    return "compute_ir_missing";
  }
  if ((ir.op_hash_hi == 0u && ir.op_hash_lo == 0u) ||
      ir.op_hash_hi != key.canonical_ir_hash_hi ||
      ir.op_hash_lo != key.canonical_ir_hash_lo) {
    return "compute_ir_hash_mismatch";
  }
  return nullptr;
}

[[nodiscard]] inline ComputeInputAdmission
AdmitComputeInputWithKey(const ComputeIR &ir, const ComputeApi api,
                         const std::vector<u8> &canonical_bytes,
                         const ArtifactKey &key) {
  if (const char *const reason =
          ComputeInputHeaderFailure(ir, api, canonical_bytes, key);
      reason != nullptr) {
    return RejectComputeInput(key, reason);
  }

  ParsedIR parsed = ParseComputeIR(ir, canonical_bytes);
  if (!parsed.ok) {
    const char *const reason = parsed.reason;
    return ComputeInputAdmission{.key = key,
                                 .parsed = std::move(parsed),
                                 .parse_count = 1u,
                                 .reason = reason};
  }
  if (const char *const reason = ValidateLowerableIR(parsed, ir.scalar);
      reason != nullptr) {
    return ComputeInputAdmission{.key = key,
                                 .parsed = std::move(parsed),
                                 .parse_count = 1u,
                                 .reason = reason};
  }
  return ComputeInputAdmission{.key = key,
                               .parsed = std::move(parsed),
                               .parse_count = 1u,
                               .ok = true,
                               .reason = "ok"};
}

[[nodiscard]] inline ComputeInputAdmission
AdmitComputeInput(const ComputeIR &ir, const ComputeApi api,
                  const std::vector<u8> &canonical_bytes) {
  const ArtifactKey key = MakeKey(ir, api, canonical_bytes);
  return AdmitComputeInputWithKey(ir, api, canonical_bytes, key);
}

// Generated canonical IR can carry the ParsedIR that was serialized. This
// path applies the same identity and lowerability admission without parsing
// those just-produced bytes back into a duplicate owner.
[[nodiscard]] inline ComputeInputAdmission
AdmitGeneratedComputeInput(const ComputeIR &ir, const ComputeApi api,
                           ParsedIR parsed) {
  const ArtifactKey key = MakeKey(ir, api);
  if (const char *const reason =
          ComputeInputHeaderFailure(ir, api, ir.canonical_bytes, key);
      reason != nullptr) {
    return RejectComputeInput(key, reason);
  }
  if (!parsed.ok) {
    const char *const reason = parsed.reason;
    return ComputeInputAdmission{
        .key = key, .parsed = std::move(parsed), .reason = reason};
  }
  if (parsed.scalar_mode != DomainModeFor(ir.scalar, ir.domain)) {
    return ComputeInputAdmission{.key = key,
                                 .parsed = std::move(parsed),
                                 .reason = "compute_ir_scalar_mismatch"};
  }
  if (parsed.fixed_format != ir.fixed_format) {
    return ComputeInputAdmission{.key = key,
                                 .parsed = std::move(parsed),
                                 .reason =
                                     "compute_ir_numeric_policy_mismatch"};
  }
  if (const char *const reason = ValidateLowerableIR(parsed, ir.scalar);
      reason != nullptr) {
    return ComputeInputAdmission{
        .key = key, .parsed = std::move(parsed), .reason = reason};
  }
  return ComputeInputAdmission{
      .key = key, .parsed = std::move(parsed), .ok = true, .reason = "ok"};
}

[[nodiscard]] inline ComputeInputAdmission
AdmitComputeInput(const ComputeIR &ir, const ComputeApi api) {
  return AdmitComputeInput(ir, api, ir.canonical_bytes);
}

} // namespace rund::kernel::compute_lowering_detail
