#include "identity.hpp"

#include <cstdint>

namespace rund::compute::detail::residency {
namespace {

class Hash final {
public:
  Hash() noexcept {
    text("rund.compute.pipeline.residency");
    number(9u);
  }

  void byte(const std::uint8_t value) noexcept {
    lo_ ^= value;
    lo_ *= 1099511628211ull;
    hi_ ^= static_cast<std::uint8_t>(value + 0x9du);
    hi_ *= 14029467366897019727ull;
  }

  void number(const std::uint64_t value) noexcept {
    for (unsigned shift = 0u; shift != 64u; shift += 8u) {
      byte(static_cast<std::uint8_t>(value >> shift));
    }
  }

  void text(const char *value) noexcept {
    while (*value != '\0') {
      byte(static_cast<std::uint8_t>(*value++));
    }
    byte(0u);
  }

  [[nodiscard]] Identity finish() const noexcept {
    return Identity{.hi = hi_, .lo = lo_};
  }

private:
  std::uint64_t hi_{7809847782465536322ull};
  std::uint64_t lo_{1469598103934665603ull};
};

} // namespace

Identity IdentifyResidencyPlan(const std::uint64_t page_bytes,
                               const std::uint64_t page_count,
                               const std::uint64_t frame_capacity,
                               const DirtyRange dirty,
                               const std::uint64_t dirty_bytes,
                               const std::uint64_t prefetch_distance) noexcept {
  Hash hash{};
  hash.text("stream");
  hash.number(page_bytes);
  hash.number(page_count);
  hash.number(frame_capacity);
  hash.number(dirty.offset);
  hash.number(dirty.bytes);
  hash.number(dirty_bytes);
  hash.number(prefetch_distance);
  return hash.finish();
}

Identity IdentifyResidencyPlan(
    const std::uint64_t page_count, const std::uint64_t frame_capacity,
    const std::uint64_t prefetch_distance,
    const std::span<const TiledGraphResource> resources,
    const std::span<const TiledGraphPhysicalClass> physical_classes,
    const std::span<const TiledGraphStage> stages,
    const std::uint64_t graph_fingerprint_hi,
    const std::uint64_t graph_fingerprint_lo) noexcept {
  Hash hash{};
  hash.text("tiled-graph");
  hash.number(page_count);
  hash.number(frame_capacity);
  hash.number(prefetch_distance);
  hash.number(resources.size());
  for (const TiledGraphResource &resource : resources) {
    hash.number(resource.resource);
    hash.number(resource.physical_id);
    hash.number(static_cast<std::uint8_t>(resource.type));
    hash.number(resource.format.integer_bits);
    hash.number(resource.format.fraction_bits);
    hash.number(static_cast<std::uint8_t>(resource.format.rounding));
    hash.number(static_cast<std::uint8_t>(resource.format.overflow));
    hash.number(static_cast<std::uint8_t>(resource.format.approximation));
    hash.number(static_cast<std::uint8_t>(resource.role));
    hash.number(resource.color);
    hash.number(resource.first_stage);
    hash.number(resource.last_stage);
    hash.number(resource.producer_stage);
    hash.number(resource.page_bytes);
    hash.number(resource.logical_bytes);
    hash.number(static_cast<std::uint8_t>(resource.kind));
    hash.number(static_cast<std::uint8_t>(resource.persistence));
  }
  hash.number(physical_classes.size());
  for (const TiledGraphPhysicalClass physical : physical_classes) {
    hash.number(physical.physical_id);
    hash.number(static_cast<std::uint8_t>(physical.role));
    hash.number(static_cast<std::uint8_t>(physical.type));
    hash.number(physical.format.integer_bits);
    hash.number(physical.format.fraction_bits);
    hash.number(static_cast<std::uint8_t>(physical.format.rounding));
    hash.number(static_cast<std::uint8_t>(physical.format.overflow));
    hash.number(static_cast<std::uint8_t>(physical.format.approximation));
    hash.number(physical.color);
    hash.number(physical.page_bytes);
  }
  hash.number(stages.size());
  for (const TiledGraphStage &stage : stages) {
    hash.number(stage.node);
    hash.number(static_cast<std::uint8_t>(stage.domain));
    hash.number(stage.active_count_input);
    hash.number(stage.ports.size());
    for (const TiledGraphPort port : stage.ports) {
      hash.number(port.resource);
      hash.number(static_cast<std::uint8_t>(port.access));
      hash.number(port.program_port);
      hash.number(port.next_stage);
    }
    hash.number(stage.same_batch_predecessors.size());
    for (const TiledGraphStageDependency predecessor :
         stage.same_batch_predecessors) {
      hash.number(predecessor.stage);
      hash.number(static_cast<std::uint8_t>(predecessor.phase));
    }
    hash.number(stage.prior_batch_predecessors.size());
    for (const TiledGraphStageDependency predecessor :
         stage.prior_batch_predecessors) {
      hash.number(predecessor.stage);
      hash.number(static_cast<std::uint8_t>(predecessor.phase));
    }
  }
  bool remapped = false;
  for (const TiledGraphResource &resource : resources) {
    remapped = remapped || !resource.remaps.empty();
  }
  if (remapped) {
    hash.text("graph-remap-v10");
    hash.number(graph_fingerprint_hi);
    hash.number(graph_fingerprint_lo);
    for (const TiledGraphResource &resource : resources) {
      if (resource.remaps.empty()) {
        continue;
      }
      hash.number(resource.resource);
      hash.number(resource.remaps.size());
      for (const GraphPageRemap remap : resource.remaps) {
        hash.number(remap.target_local);
        hash.number(remap.source_local);
        hash.number(static_cast<std::uint8_t>(remap.source_origin));
      }
    }
  }
  return hash.finish();
}

} // namespace rund::compute::detail::residency
