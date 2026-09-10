#include "../../state/assembly.hpp"
#include "internal.hpp"

#include <limits>
#include <vector>

namespace rund::compute::detail {

Status prove_sealed_repetitions(
    const PipelineBuildState &build,
    const std::span<const resource::Resource> resource_shapes,
    const std::span<const resource::Access> resource_accesses,
    const std::span<const std::uint8_t> external_resources,
    const std::span<const resource::Access> publication_accesses) {
  if (build.sealed_repetitions <= 1u) {
    return Status::success();
  }
  if (!build.state_pairs.empty()) {
    return Status::fail(Reason::PipelineTemporalDependency);
  }

  // Place every caller-owned write in invocation t before every caller-owned
  // read in invocation t + 1. resource::analyze remains the sole exact-range
  // authority. A W -> R witness is an observable temporal carry.
  std::vector<resource::Access> temporal_accesses;
  temporal_accesses.reserve(resource_accesses.size() +
                            publication_accesses.size());
  const auto append_mode = [&](const resource::AccessMode mode) {
    const auto append = [&](const resource::Access &source) {
      if (source.resource == 0u ||
          source.resource > external_resources.size()) {
        return Status::fail(Reason::PipelineInvalid);
      }
      if (source.mode != mode ||
          external_resources[source.resource - 1u] == 0u) {
        return Status::success();
      }
      if (temporal_accesses.size() >=
          std::numeric_limits<std::uint32_t>::max()) {
        return Status::fail(Reason::PipelineCapacity);
      }
      resource::Access access = source;
      access.node = static_cast<std::uint32_t>(temporal_accesses.size());
      temporal_accesses.push_back(access);
      return Status::success();
    };
    for (const resource::Access &access : resource_accesses) {
      const Status appended = append(access);
      if (!appended) {
        return appended;
      }
    }
    for (const resource::Access &access : publication_accesses) {
      const Status appended = append(access);
      if (!appended) {
        return appended;
      }
    }
    return Status::success();
  };
  const Status writes = append_mode(resource::AccessMode::Write);
  if (!writes) {
    return writes;
  }
  const Status reads = append_mode(resource::AccessMode::Read);
  if (!reads || temporal_accesses.empty()) {
    return reads;
  }

  auto temporal =
      resource::analyze(resource_shapes, temporal_accesses,
                        static_cast<std::uint32_t>(temporal_accesses.size()));
  if (!temporal) {
    return Status::fail(temporal.reason());
  }
  for (const resource::Barrier &barrier : temporal->barriers) {
    if (barrier.before == resource::AccessMode::Write &&
        barrier.after == resource::AccessMode::Read) {
      return Status::fail(Reason::PipelineTemporalDependency);
    }
  }
  return Status::success();
}

} // namespace rund::compute::detail
