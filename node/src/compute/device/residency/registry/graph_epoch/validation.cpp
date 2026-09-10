#include "../../registry.hpp"
#include "../internal.hpp"
#include "internal.hpp"

#include <kernel/core/checked.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

namespace rund::compute::detail::residency::graph_epoch_detail {
namespace {

[[nodiscard]] bool remap_at(const std::span<const GraphPageRemap> remaps,
                            const std::size_t target,
                            const std::size_t page_count,
                            const std::size_t frame_count,
                            std::size_t &source) noexcept {
  if (remaps.empty()) {
    source = target;
    return target < page_count;
  }
  if (remaps.size() != frame_count || target >= page_count) {
    return false;
  }
  const auto found = std::find_if(remaps.begin(), remaps.end(),
                                  [target](const GraphPageRemap remap) {
                                    return remap.target_local == target;
                                  });
  if (found == remaps.end() || found->source_local >= page_count) {
    return false;
  }
  source = found->source_origin == GraphPageOrigin::Begin
               ? found->source_local
               : page_count - 1u - found->source_local;
  return source < page_count;
}

[[nodiscard]] bool valid_remaps(const std::span<const GraphPageRemap> remaps,
                                const std::size_t page_count,
                                const std::size_t frame_count) noexcept {
  if (remaps.empty()) {
    return true;
  }
  if (frame_count == 0u || frame_count > PipelineLeafCapacity ||
      remaps.size() != frame_count || remaps.size() > GraphPageRemapCapacity) {
    return false;
  }
  std::array<bool, PipelineLeafCapacity> sources{};
  std::array<bool, PipelineLeafCapacity> targets{};
  for (const GraphPageRemap remap : remaps) {
    if (remap.source_local >= frame_count ||
        remap.target_local >= frame_count ||
        static_cast<std::uint8_t>(remap.source_origin) >
            static_cast<std::uint8_t>(GraphPageOrigin::End) ||
        sources[remap.source_local] || targets[remap.target_local]) {
      return false;
    }
    sources[remap.source_local] = true;
    targets[remap.target_local] = true;
  }
  for (std::size_t target = 0u; target < page_count; ++target) {
    std::size_t source = 0u;
    if (!remap_at(remaps, target, page_count, frame_count, source)) {
      return false;
    }
  }
  return true;
}

} // namespace

AuthorityResult Validation::requests(
    const std::span<const PageUse> uses,
    const std::span<const GraphPortRequest> ports,
    const std::size_t anchor_port, const std::uint64_t epoch,
    AdmissionDraft &draft) noexcept {
  if (ports.size() < 2u || ports.size() > TiledGraphPortCapacity ||
      anchor_port >= ports.size() || uses.empty()) {
    return AuthorityResult{.failure = AuthorityFailure::Invalid};
  }
  const std::size_t page_count = ports.front().use_count;
  if (page_count == 0u || page_count > PipelineLeafCapacity ||
      ports.size() > MaxUses / page_count ||
      uses.size() != ports.size() * page_count) {
    return AuthorityResult{.failure = page_count > PipelineLeafCapacity
                                      ? AuthorityFailure::Capacity
                                      : AuthorityFailure::Invalid};
  }

  draft.page_count = page_count;
  draft.port_count = ports.size();
  draft.has_remap = std::any_of(
      ports.begin(), ports.end(),
      [](const GraphPortRequest &request) { return !request.remaps.empty(); });
  std::array<bool, TiledGraphPortCapacity> read_program_ports{};
  std::array<bool, TiledGraphPortCapacity> write_program_ports{};
  std::size_t read_port_count = 0u;
  std::size_t write_port_count = 0u;
  std::size_t remap_total = 0u;
  for (std::size_t port_index = 0u; port_index < ports.size(); ++port_index) {
    const GraphPortRequest &request = ports[port_index];
    const std::size_t expected_first = port_index * page_count;
    if (request.first_use != expected_first ||
        request.use_count != page_count || request.region.count == 0u ||
        request.region.count > PipelineLeafCapacity ||
        request.cache_region_count == 0u ||
        request.cache_region_count > request.cache_regions.size() ||
        (request.remaps.size() > GraphPageRemapCapacity - remap_total) ||
        !valid_remaps(request.remaps, page_count, request.region.count) ||
        request.materialization.resource == 0u ||
        request.materialization.key.backing == 0u ||
        request.materialization.key.page != 0u ||
        request.materialization.page_bytes == 0u ||
        request.materialization.page_count == 0u) {
      return AuthorityResult{.failure =
                                 request.region.count > PipelineLeafCapacity
                                     ? AuthorityFailure::Capacity
                                     : AuthorityFailure::Invalid};
    }
    remap_total += request.remaps.size();
    if (port_index != 0u &&
        (request.region.tier != ports.front().region.tier ||
         request.region.count != ports.front().region.count)) {
      return AuthorityResult{.failure = AuthorityFailure::Invalid};
    }
    bool execution_contained = false;
    for (std::size_t cache_index = 0u; cache_index < request.cache_region_count;
         ++cache_index) {
      const FrameRegion cache = request.cache_regions[cache_index];
      const std::size_t cache_end =
          static_cast<std::size_t>(cache.first) + cache.count;
      const std::size_t execution_end =
          static_cast<std::size_t>(request.region.first) + request.region.count;
      if (cache.count == 0u || cache.count > PipelineLeafCapacity ||
          cache.tier != request.region.tier ||
          cache.role != request.region.role) {
        return AuthorityResult{.failure = cache.count > PipelineLeafCapacity
                                          ? AuthorityFailure::Capacity
                                          : AuthorityFailure::Invalid};
      }
      execution_contained =
          (request.region.first >= cache.first && execution_end <= cache_end) ||
          execution_contained;
      for (std::size_t other_cache = 0u; other_cache < cache_index;
           ++other_cache) {
        const FrameRegion other = request.cache_regions[other_cache];
        const std::size_t other_end =
            static_cast<std::size_t>(other.first) + other.count;
        if (!(cache_end <= other.first || other_end <= cache.first)) {
          return AuthorityResult{.failure = AuthorityFailure::Invalid};
        }
      }
    }
    if (!execution_contained) {
      return AuthorityResult{.failure = AuthorityFailure::Invalid};
    }
    for (std::size_t prior = 0u; prior < port_index; ++prior) {
      const GraphPortRequest &other = ports[prior];
      if (request.materialization.resource == other.materialization.resource) {
        return AuthorityResult{.failure = AuthorityFailure::Invalid};
      }
      for (std::size_t cache_index = 0u;
           cache_index < request.cache_region_count; ++cache_index) {
        const FrameRegion cache = request.cache_regions[cache_index];
        const std::size_t cache_end =
            static_cast<std::size_t>(cache.first) + cache.count;
        for (std::size_t other_index = 0u;
             other_index < other.cache_region_count; ++other_index) {
          const FrameRegion other_cache = other.cache_regions[other_index];
          const std::size_t other_end =
              static_cast<std::size_t>(other_cache.first) + other_cache.count;
          if (!(cache_end <= other_cache.first ||
                other_end <= cache.first)) {
            return AuthorityResult{.failure = AuthorityFailure::Invalid};
          }
        }
      }
    }
    const Access access = uses[expected_first].access;
    const CacheDomain domain = request.materialization.key.domain;
    const bool legal_role =
        (request.region.role == FrameRole::Input && access == Access::Read &&
         domain == CacheDomain::Backing) ||
        (request.region.role == FrameRole::Intermediate &&
         (access == Access::Read || access == Access::Write) &&
         domain == CacheDomain::Transient) ||
        (request.region.role == FrameRole::Output && access == Access::Write &&
         (domain == CacheDomain::Transient || domain == CacheDomain::Backing));
    std::array<bool, TiledGraphPortCapacity> &program_ports =
        access == Access::Read ? read_program_ports : write_program_ports;
    std::size_t &program_port_count =
        access == Access::Read ? read_port_count : write_port_count;
    if (!legal_role || request.program_port >= TiledGraphPortCapacity ||
        program_ports[request.program_port] ||
        (port_index == anchor_port && access != Access::Read)) {
      return AuthorityResult{.failure = AuthorityFailure::Invalid};
    }
    program_ports[request.program_port] = true;
    ++program_port_count;
    for (std::size_t index = 0u; index < page_count; ++index) {
      const PageUse use = uses[expected_first + index];
      std::size_t source = 0u;
      std::uint64_t base_page = std::numeric_limits<std::uint64_t>::max();
      for (std::size_t candidate = 0u; candidate < page_count; ++candidate) {
        base_page =
            std::min(base_page, uses[expected_first + candidate].key.page);
      }
      std::uint64_t expected_page = 0u;
      if (!remap_at(request.remaps, index, page_count, request.region.count,
                    source) ||
          !kernel::checked::add(base_page, source, expected_page) ||
          use.access != access ||
          ((!draft.has_remap && use.key.page != uses[index].key.page) ||
           (draft.has_remap && use.key.page != expected_page)) ||
          !project_graph_use(use, request.materialization, access, epoch,
                             draft.projected[expected_first + index])) {
        return AuthorityResult{.failure = AuthorityFailure::Invalid};
      }
      for (std::size_t prior = 0u; prior < index; ++prior) {
        if (use.key == uses[expected_first + prior].key) {
          return AuthorityResult{.failure = AuthorityFailure::Invalid};
        }
      }
    }
  }
  for (std::size_t port = 0u; port < read_port_count; ++port) {
    if (!read_program_ports[port]) {
      return AuthorityResult{.failure = AuthorityFailure::Invalid};
    }
  }
  for (std::size_t port = 0u; port < write_port_count; ++port) {
    if (!write_program_ports[port]) {
      return AuthorityResult{.failure = AuthorityFailure::Invalid};
    }
  }
  return AuthorityResult{.failure = AuthorityFailure::None};
}

AuthorityResult Validation::regions(
    const Authority &authority,
    const std::span<const GraphPortRequest> ports) noexcept {
  const auto valid_region = [&authority](const FrameRegion region) noexcept {
    const std::size_t first = region.first;
    const std::size_t count = region.count;
    return count != 0u && first <= authority.frames_.size() &&
           count <= authority.frames_.size() - first &&
           std::all_of(
               authority.frames_.begin() + static_cast<std::ptrdiff_t>(first),
               authority.frames_.begin() +
                   static_cast<std::ptrdiff_t>(first + count),
               [region](const registry_model::Frame &frame) {
                 return frame.assigned && frame.tier == region.tier &&
                        frame.role == region.role;
               });
  };
  if (std::any_of(
          ports.begin(), ports.end(),
          [&valid_region](const GraphPortRequest &port) {
            return !valid_region(port.region) ||
                   std::any_of(
                       port.cache_regions.begin(),
                       port.cache_regions.begin() +
                           static_cast<std::ptrdiff_t>(
                               port.cache_region_count),
                       [&valid_region](const FrameRegion region) {
                         return !valid_region(region);
                       });
          })) {
    return AuthorityResult{.failure = AuthorityFailure::Invalid};
  }
  return AuthorityResult{.failure = AuthorityFailure::None};
}

} // namespace rund::compute::detail::residency::graph_epoch_detail
