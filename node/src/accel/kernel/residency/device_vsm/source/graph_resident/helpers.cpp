#include "internal.hpp"

#include <algorithm>

namespace rund::node::accel::detail::device_vsm_graph_resident_source::detail {

const char *graph_resident_metal_type(
    const DeviceVsmGraphResidentProof &proof) noexcept {
  return proof.type.scalar == rund::kernel::ComputeScalar::Lane32 ? "uint"
                                                                   : "ulong";
}

const char *graph_resident_vulkan_type(
    const DeviceVsmGraphResidentProof &proof) noexcept {
  return proof.type.scalar == rund::kernel::ComputeScalar::Lane32 ? "uint"
                                                                   : "uint64_t";
}

const char *graph_resident_zero_literal(
    const DeviceVsmGraphResidentProof &proof) noexcept {
  return proof.type.scalar == rund::kernel::ComputeScalar::Lane32 ? "0u"
                                                                   : "0ul";
}

rund::kernel::compute_lowering_detail::ParsedIR merged_helpers(
    const std::span<const DeviceVsmGraphResidentStageSource> stages) {
  using ParsedIR = rund::kernel::compute_lowering_detail::ParsedIR;
  ParsedIR result{};
  result.name = "rund_graph_resident_helpers";
  result.ok = true;
  bool first = true;
  for (const DeviceVsmGraphResidentStageSource &stage : stages) {
    if (stage.input == nullptr || !stage.input->parsed.ok) {
      result.ok = false;
      return result;
    }
    if (first) {
      first = false;
      result.scalar_mode = stage.input->parsed.scalar_mode;
      result.fixed_format = stage.input->parsed.fixed_format;
    } else if (result.scalar_mode != stage.input->parsed.scalar_mode ||
               result.fixed_format != stage.input->parsed.fixed_format) {
      result.ok = false;
      return result;
    }
    result.nodes.insert(result.nodes.end(), stage.input->parsed.nodes.begin(),
                        stage.input->parsed.nodes.end());
  }
  return result;
}

const DeviceVsmGraphResidentResource *resource(
    const DeviceVsmGraphResidentProof &proof, const std::uint32_t id) noexcept {
  for (std::size_t index = 0u; index < proof.resource_count; ++index) {
    if (proof.resources[index].resource == id) {
      return &proof.resources[index];
    }
  }
  return nullptr;
}

std::string offset(const DeviceVsmGraphResidentRegion &bank_region,
                   const DeviceVsmGraphResidentRegion &cache_region,
                   const std::uint32_t frame_capacity, const std::string &slot,
                   const std::string &element) {
  return "((" + std::to_string(bank_region.first) + "u - " +
         std::to_string(cache_region.first) +
         "u) * config.payload_elements + (" + slot + " % " +
         std::to_string(frame_capacity) + "u) * config.payload_elements + (" +
         element + " % config.payload_elements))";
}

std::size_t external_count(
    const DeviceVsmGraphResidentProof &proof) noexcept {
  std::size_t count = 0u;
  for (std::size_t index = 0u; index < proof.resource_count; ++index) {
    const auto &row = proof.resources[index];
    if (row.role == 0u || row.role == 2u) {
      count = std::max(count, static_cast<std::size_t>(row.external_slot) + 1u);
    }
  }
  return count;
}

} // namespace rund::node::accel::detail::device_vsm_graph_resident_source::detail
