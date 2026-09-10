#include "../plan.hpp"

#include "internal.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace rund::compute::detail::residency::execution {
namespace {

constexpr std::uint64_t FnvOffset = 1469598103934665603ull;
constexpr std::uint64_t FnvPrime = 1099511628211ull;

[[nodiscard]] constexpr std::uint64_t mix(std::uint64_t hash,
                                          std::uint64_t value) noexcept {
  for (std::size_t byte = 0u; byte < sizeof(value); ++byte) {
    hash ^= value & 0xffu;
    hash *= FnvPrime;
    value >>= 8u;
  }
  return hash;
}

[[nodiscard]] constexpr bool valid(const FrameRegion region,
                                   const FrameTier tier, const FrameRole role,
                                   const std::uint32_t count) noexcept {
  return region.tier == tier && region.role == role && region.count == count &&
         region.first <=
             std::numeric_limits<std::uint32_t>::max() - region.count;
}

[[nodiscard]] constexpr bool
valid_host_input(const FrameRegion region,
                 const std::uint32_t minimum) noexcept {
  return region.tier == FrameTier::Host && region.role == FrameRole::Input &&
         region.count >= minimum && region.count <= UseCapacity &&
         region.first <=
             std::numeric_limits<std::uint32_t>::max() - region.count;
}

[[nodiscard]] constexpr bool
valid_host_output(const FrameRegion region,
                  const std::uint32_t minimum) noexcept {
  return region.tier == FrameTier::Host && region.role == FrameRole::Output &&
         region.count >= minimum && region.count <= UseCapacity &&
         region.first <=
             std::numeric_limits<std::uint32_t>::max() - region.count;
}

[[nodiscard]] constexpr bool overlaps(const FrameRegion left,
                                      const FrameRegion right) noexcept {
  return left.tier == right.tier && left.first < right.first + right.count &&
         right.first < left.first + left.count;
}

[[nodiscard]] constexpr bool scalar_width(const std::uint64_t bytes) noexcept {
  return bytes == 1u || bytes == 2u || bytes == 4u || bytes == 8u;
}

[[nodiscard]] constexpr bool
empty(const GraphMaterialization &materialization) noexcept {
  return materialization.resource == 0u && materialization.key == CacheKey{} &&
         materialization.page_bytes == 0u && materialization.page_count == 0u &&
         materialization.boundary_extent == 0u;
}

[[nodiscard]] constexpr bool
valid_fill(const Materialization &value, const Access access,
           const std::uint64_t page_bytes) noexcept {
  switch (value.fill) {
  case FetchFill::None:
    return value.fill_element_bytes == 0u && value.fill_value == 0u;
  case FetchFill::ZeroInactiveTail:
    return access == Access::Read && value.read_prefix_bytes == 0u &&
           value.target_prefix_bytes == 0u && value.read_suffix_bytes == 0u &&
           value.payload_bytes == page_bytes &&
           value.fill_element_bytes == 0u && value.fill_value == 0u;
  case FetchFill::RepeatBoundary:
  case FetchFill::ConstantBoundary: {
    const std::uint64_t scalar = value.fill_element_bytes;
    if (access != Access::Read || !scalar_width(scalar) ||
        page_bytes % scalar != 0u || value.logical_bytes % scalar != 0u ||
        value.payload_bytes % scalar != 0u ||
        value.read_prefix_bytes % scalar != 0u ||
        value.target_prefix_bytes % scalar != 0u ||
        value.read_suffix_bytes % scalar != 0u ||
        (value.read_prefix_bytes == 0u && value.read_suffix_bytes == 0u) ||
        (value.fill == FetchFill::RepeatBoundary && value.fill_value != 0u)) {
      return false;
    }
    return scalar == 8u ||
           value.fill_value < (std::uint64_t{1u} << (scalar * 8u));
  }
  }
  return false;
}

[[nodiscard]] bool valid(const Materialization &value, const Access access,
                         const std::uint64_t page_count,
                         const std::uint32_t frame_capacity) noexcept {
  const GraphMaterialization &cache = value.cache;
  std::uint64_t capacity = 0u;
  std::uint64_t last_offset = 0u;
  std::uint64_t next_delta = 0u;
  std::uint64_t last_next_use = 0u;
  if (value.access != access || cache.resource == 0u ||
      cache.key.backing == 0u || cache.key.page != 0u ||
      cache.key.domain != CacheDomain::Backing || cache.page_bytes == 0u ||
      cache.page_count != page_count || value.logical_bytes == 0u ||
      value.payload_bytes == 0u ||
      !plan_internal::checked_mul(page_count, value.payload_bytes, capacity) ||
      !plan_internal::checked_mul(page_count - 1u, value.payload_bytes,
                                  last_offset) ||
      value.logical_bytes > capacity || value.logical_bytes <= last_offset ||
      value.read_prefix_bytes > value.target_prefix_bytes ||
      value.target_prefix_bytes > cache.page_bytes ||
      value.payload_bytes > cache.page_bytes - value.target_prefix_bytes ||
      value.read_suffix_bytes >
          cache.page_bytes - value.target_prefix_bytes - value.payload_bytes ||
      !valid_fill(value, access, cache.page_bytes) ||
      (access == Access::Read &&
       (!plan_internal::checked_mul(page_count - 1u, value.next_use_stride,
                                    next_delta) ||
        !plan_internal::checked_add(value.next_use_base, next_delta,
                                    last_next_use) ||
        last_next_use == NeverUse || value.next_use_base == 0u ||
        last_next_use <= (page_count - 1u) / frame_capacity)) ||
      value.dirty_origin >
          std::numeric_limits<std::uint64_t>::max() - value.logical_bytes) {
    return false;
  }
  return access == Access::Read
             ? value.next_use_base != NeverUse
             : value.next_use_base == NeverUse && value.next_use_stride == 0u &&
                   value.read_prefix_bytes == 0u &&
                   value.target_prefix_bytes == 0u &&
                   value.read_suffix_bytes == 0u &&
                   value.fill == FetchFill::None &&
                   value.fill_element_bytes == 0u && value.fill_value == 0u;
}

[[nodiscard]] bool valid_canonical_window(const Request &request) noexcept {
  const GraphMaterialization &canonical = request.canonical_input;
  if (empty(canonical)) {
    return true;
  }
  const Materialization &input = request.input;
  const GraphMaterialization &expanded = input.cache;
  if (input.fill_element_bytes == 0u ||
      input.logical_bytes % input.fill_element_bytes != 0u) {
    return false;
  }
  // Canonical identity follows the raw payload boundary. The expanded frame
  // has a separate clipped halo/fill boundary, even when the raw input ends
  // exactly on a canonical page. Keep both extents exact without conflating
  // their cache/materialization domains.
  const std::uint64_t active_elements =
      input.logical_bytes / input.fill_element_bytes;
  const std::uint64_t canonical_extent =
      input.logical_bytes % input.payload_bytes == 0u ? 0u : active_elements;
  const std::uint64_t required_sources = std::min<std::uint64_t>(
      request.page_count,
      static_cast<std::uint64_t>(request.frame_capacity) + 2u);
  return canonical.resource == expanded.resource &&
         canonical.key.backing == expanded.key.backing &&
         canonical.key.version == expanded.key.version &&
         canonical.key.page == 0u &&
         canonical.key.domain == CacheDomain::Backing &&
         canonical.key != expanded.key &&
         canonical.page_bytes == input.payload_bytes &&
         canonical.page_count == request.page_count &&
         canonical.boundary_extent == canonical_extent &&
         expanded.boundary_extent == active_elements &&
         input.access == Access::Read && input.read_prefix_bytes != 0u &&
         input.read_prefix_bytes == input.target_prefix_bytes &&
         input.read_prefix_bytes == input.read_suffix_bytes &&
         input.read_prefix_bytes <= input.payload_bytes &&
         (input.fill == FetchFill::RepeatBoundary ||
          input.fill == FetchFill::ConstantBoundary) &&
         required_sources <= WindowFootprintSourceCapacity &&
         request.host_input[0].count >= required_sources &&
         request.host_input[1].count >= required_sources;
}

[[nodiscard]] constexpr std::uint64_t hash(const std::uint64_t seed,
                                           const CacheKey key) noexcept {
  std::uint64_t result = mix(seed, key.backing);
  result = mix(result, key.version);
  result = mix(result, key.extent);
  result = mix(result, key.materialization_hi);
  result = mix(result, key.materialization_lo);
  result = mix(result, key.page);
  return mix(result, static_cast<std::uint8_t>(key.domain));
}

[[nodiscard]] constexpr std::uint64_t
hash(std::uint64_t seed, const GraphMaterialization &value) noexcept {
  seed = mix(seed, value.resource);
  seed = hash(seed, value.key);
  seed = mix(seed, value.page_bytes);
  seed = mix(seed, value.page_count);
  return mix(seed, value.boundary_extent);
}

[[nodiscard]] constexpr std::uint64_t hash(std::uint64_t seed,
                                           const FrameRegion region) noexcept {
  seed = mix(seed, static_cast<std::uint8_t>(region.tier));
  seed = mix(seed, static_cast<std::uint8_t>(region.role));
  seed = mix(seed, region.first);
  return mix(seed, region.count);
}

[[nodiscard]] constexpr std::uint64_t
hash(std::uint64_t seed, const Materialization &value) noexcept {
  seed = hash(seed, value.cache);
  seed = mix(seed, static_cast<std::uint8_t>(value.access));
  seed = mix(seed, value.logical_bytes);
  seed = mix(seed, value.payload_bytes);
  seed = mix(seed, value.read_prefix_bytes);
  seed = mix(seed, value.target_prefix_bytes);
  seed = mix(seed, value.read_suffix_bytes);
  seed = mix(seed, static_cast<std::uint8_t>(value.fill));
  seed = mix(seed, value.fill_element_bytes);
  seed = mix(seed, value.fill_value);
  seed = mix(seed, value.dirty_origin);
  seed = mix(seed, value.next_use_base);
  seed = mix(seed, value.next_use_stride);
  return mix(seed, value.retain_until);
}

} // namespace

SealResult seal(const Request &request) noexcept {
  SealResult result{};
  if (request.page_count == 0u || request.frame_capacity == 0u) {
    return result;
  }
  if (request.frame_capacity > UseCapacity) {
    result.failure = SealFailure::Capacity;
    return result;
  }
  const bool external_atomic = request.publication.generation != 0u &&
                               request.publication.capability != 0u;
  const bool malformed_capability = (request.publication.generation == 0u) !=
                                    (request.publication.capability == 0u);
  if (!valid(request.input, Access::Read, request.page_count,
             request.frame_capacity) ||
      !valid_canonical_window(request) ||
      !valid(request.output, Access::Write, request.page_count,
             request.frame_capacity) ||
      request.publication.backing != request.output.cache.key.backing ||
      request.publication.version != request.output.cache.key.version ||
      malformed_capability ||
      request.publication.extent.offset != request.output.dirty_origin ||
      request.publication.extent.bytes != request.output.logical_bytes) {
    return result;
  }
  for (std::size_t bank = 0u; bank < BankCapacity; ++bank) {
    if (!valid_host_input(request.host_input[bank], request.frame_capacity) ||
        !valid(request.device_input[bank], FrameTier::Device, FrameRole::Input,
               request.frame_capacity) ||
        !valid(request.device_output[bank], FrameTier::Device,
               FrameRole::Output, request.frame_capacity) ||
        !valid_host_output(request.host_output[bank], request.frame_capacity)) {
      return result;
    }
  }
  if (request.host_input[0].count != request.host_input[1].count ||
      request.host_output[0].count != request.host_output[1].count) {
    return result;
  }
  const std::array<FrameRegion, BankCapacity * 4u> regions{
      request.host_input[0],    request.host_input[1],
      request.device_input[0],  request.device_input[1],
      request.device_output[0], request.device_output[1],
      request.host_output[0],   request.host_output[1],
  };
  for (std::size_t left = 0u; left < regions.size(); ++left) {
    for (std::size_t right = left + 1u; right < regions.size(); ++right) {
      if (overlaps(regions[left], regions[right])) {
        return result;
      }
    }
  }
  const std::uint64_t epochs =
      request.page_count / request.frame_capacity +
      static_cast<std::uint64_t>(request.page_count % request.frame_capacity !=
                                 0u);
  if ((request.input.retain_until != NeverUse &&
       request.input.retain_until < epochs - 1u) ||
      (request.output.retain_until != NeverUse &&
       request.output.retain_until < epochs - 1u)) {
    return result;
  }
  std::uint64_t identity = mix(FnvOffset, 5u);
  identity = mix(identity, request.page_count);
  identity = mix(identity, request.frame_capacity);
  identity = mix(identity, request.prefetch_distance);
  identity = hash(identity, request.input);
  identity = hash(identity, request.canonical_input);
  identity = hash(identity, request.output);
  identity = mix(identity, request.publication.backing);
  identity = mix(identity, request.publication.version);
  identity = mix(identity, request.publication.generation);
  identity = mix(identity, request.publication.capability);
  identity = mix(identity, request.publication.extent.offset);
  identity = mix(identity, request.publication.extent.bytes);
  for (std::size_t bank = 0u; bank < BankCapacity; ++bank) {
    identity = hash(identity, request.host_input[bank]);
    identity = hash(identity, request.device_input[bank]);
    identity = hash(identity, request.device_output[bank]);
    identity = hash(identity, request.host_output[bank]);
  }
  result.plan.request_ = request;
  result.plan.epoch_count_ = epochs;
  result.plan.identity_ = identity == 0u ? 1u : identity;
  result.plan.external_all_or_none_ = external_atomic;
  result.failure = SealFailure::None;
  return result;
}

} // namespace rund::compute::detail::residency::execution
