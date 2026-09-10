#include "../graph_resident.hpp"

namespace rund::node::accel::detail {

std::uint64_t device_vsm_graph_resident_digest(
    const DeviceVsmGraphResidentProof &proof) noexcept {
  std::uint64_t digest = 1469598103934665603ull;
  const auto mix = [&](const std::uint64_t value) noexcept {
    digest ^= value;
    digest *= 1099511628211ull;
  };
  mix(proof.valid);
  mix(proof.width);
  mix(proof.resource_count);
  mix(proof.stage_count);
  mix(proof.port_count);
  mix(proof.owner_binding_count);
  mix(static_cast<std::uint64_t>(proof.type.scalar));
  mix(static_cast<std::uint64_t>(proof.type.domain));
  mix(proof.type.element_bytes);
  for (std::size_t index = 0u; index < proof.resource_count; ++index) {
    const auto &row = proof.resources[index];
    mix(row.resource);
    mix(row.physical_id);
    mix(row.color);
    mix(row.producer);
    mix(row.first);
    mix(row.last);
    mix(row.owner_slot);
    mix(row.external_slot);
    mix(row.page_bytes);
    mix(row.logical_bytes);
    mix(row.role);
    mix(row.type);
    mix(row.format.integer_bits);
    mix(row.format.fraction_bits);
    mix(row.valid);
  }
  for (std::size_t index = 0u; index < proof.owner_binding_count; ++index) {
    const auto &row = proof.owners[index];
    mix(row.physical_id);
    mix(row.color);
    mix(row.view);
    for (std::size_t bank = 0u; bank < row.bank_count; ++bank) {
      mix(row.local_first[bank]);
    }
    mix(row.physical_page_bytes);
    mix(row.physical_type);
    mix(row.physical_tier);
    mix(row.physical_role);
    mix(row.physical_format.integer_bits);
    mix(row.physical_format.fraction_bits);
    mix(row.bank_count);
    mix(row.valid);
    for (std::size_t bank = 0u; bank < row.bank_count; ++bank) {
      mix(row.refs[bank].id);
      mix(row.refs[bank].bytes);
      mix(row.refs[bank].offset_bytes);
      mix(row.refs[bank].element_bytes);
      mix(row.refs[bank].stride_bytes);
      mix(row.refs[bank].count);
      mix(row.refs[bank].usage);
      mix(row.cache_regions[bank].first);
      mix(row.cache_regions[bank].count);
      mix(row.cache_regions[bank].tier);
      mix(row.cache_regions[bank].role);
      mix(row.bank_regions[bank].first);
      mix(row.bank_regions[bank].count);
      mix(row.bank_regions[bank].tier);
      mix(row.bank_regions[bank].role);
      mix(row.buffer_generations[bank]);
    }
  }
  for (std::size_t index = 0u; index < proof.stage_count; ++index) {
    const auto &row = proof.stages[index];
    mix(row.node);
    mix(row.port_count);
    mix(row.domain);
    mix(row.valid);
    for (std::size_t port = 0u; port < row.port_count; ++port) {
      mix(row.ports[port].resource);
      mix(row.ports[port].next_stage);
      mix(row.ports[port].program_port);
      mix(row.ports[port].access);
      mix(row.ports[port].valid);
    }
  }
  for (const std::uint32_t word : proof.page_map.words) {
    mix(word);
  }
  return digest == 0u ? 1u : digest;
}

} // namespace rund::node::accel::detail
