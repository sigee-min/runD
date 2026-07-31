#pragma once

#include <kernel/program/compute/artifact.hpp>
#include <kernel/program/compute/backend.hpp>
#include <kernel/program/compute/binding/model.hpp>
#include <kernel/program/compute/lowering/layout.hpp>
#include <kernel/program/compute/lowering/text.hpp>

#include <cstdint>
#include <string>

namespace rund::node::accel::detail {

[[nodiscard]] inline bool ReplaceOne(std::string &source,
                                     const std::string &needle,
                                     const std::string &replacement) {
  if (needle.empty()) {
    return false;
  }
  const std::size_t at = source.find(needle);
  if (at == std::string::npos ||
      source.find(needle, at + needle.size()) != std::string::npos) {
    return false;
  }
  source.replace(at, needle.size(), replacement);
  return true;
}

[[nodiscard]] inline rund::kernel::LoweringArtifact
SpecializeMap(const rund::kernel::LoweringArtifact &source,
              const rund::kernel::ComputePlan &plan,
              const rund::kernel::BindingSet &bindings,
              const std::uint64_t alignment = 1u) {
  rund::kernel::LoweringArtifact artifact = source;
  if (!artifact.ok ||
      artifact.metadata.binding_names.size() !=
          artifact.metadata.binding_accesses.size() ||
      alignment == 0u ||
      artifact.metadata.input_element_bytes.size() !=
          static_cast<std::size_t>(plan.input_buffer_count) ||
      artifact.metadata.output_element_bytes.size() !=
          static_cast<std::size_t>(plan.output_buffer_count)) {
    artifact.ok = false;
    artifact.reason = "compute_artifact_mismatch";
    return artifact;
  }
  std::size_t read = 0u;
  std::size_t write = 0u;
  for (std::size_t index = 0u;
       index < artifact.metadata.binding_accesses.size(); ++index) {
    const auto access = artifact.metadata.binding_accesses[index];
    const bool is_read = access == rund::kernel::ComputeBindingAccess::Read;
    const bool is_write = access == rund::kernel::ComputeBindingAccess::Write;
    if (!is_read && !is_write) {
      continue;
    }
    const rund::kernel::ResidentBufferRef *const ref =
        is_read ? bindings.resident_inputs.ref(read)
                : bindings.resident_outputs.ref(write);
    const std::uint64_t element_bytes =
        is_read ? artifact.metadata.input_element_bytes[read]
                : artifact.metadata.output_element_bytes[write];
    read += is_read ? 1u : 0u;
    write += is_write ? 1u : 0u;
    if (ref == nullptr || ref->element_bytes != element_bytes ||
        ref->stride_bytes < element_bytes) {
      artifact.ok = false;
      artifact.reason = "compute_resident_stride_invalid";
      return artifact;
    }
    const std::string symbol =
        std::string{is_read ? "read_" : "write_"} +
        rund::kernel::compute_lowering_detail::SafeIdentifier(
            artifact.metadata.binding_names[index]);
    const std::string base_symbol =
        rund::kernel::compute_lowering_detail::BindingBaseSymbol(symbol);
    const std::string stride_symbol =
        rund::kernel::compute_lowering_detail::BindingStrideSymbol(symbol);
    const std::string qualifier =
        artifact.key.api == rund::kernel::ComputeApi::Metal ? "constant uint "
                                                            : "const uint ";
    if (ref->stride_bytes != element_bytes) {
      const std::string old_declaration = qualifier + stride_symbol + " = " +
                                          std::to_string(element_bytes) + "u;";
      const std::string new_declaration = qualifier + stride_symbol + " = " +
                                          std::to_string(ref->stride_bytes) +
                                          "u;";
      if (!ReplaceOne(artifact.source_text, old_declaration, new_declaration)) {
        artifact.ok = false;
        artifact.reason = "compute_artifact_mismatch";
        return artifact;
      }
    }
    const std::uint64_t base = ref->offset_bytes % alignment;
    if (base != 0u) {
      const std::string old_declaration = qualifier + base_symbol + " = 0u;";
      const std::string new_declaration =
          qualifier + base_symbol + " = " + std::to_string(base) + "u;";
      if (!ReplaceOne(artifact.source_text, old_declaration, new_declaration)) {
        artifact.ok = false;
        artifact.reason = "compute_artifact_mismatch";
        return artifact;
      }
    }
  }
  if (read != plan.input_buffer_count || write != plan.output_buffer_count) {
    artifact.ok = false;
    artifact.reason = "compute_artifact_mismatch";
    return artifact;
  }
  return artifact;
}

} // namespace rund::node::accel::detail
