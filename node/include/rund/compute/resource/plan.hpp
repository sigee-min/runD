#pragma once

#include <rund/compute/resource.hpp>
#include <rund/compute/status.hpp>

#include <cstdint>
#include <span>
#include <vector>

namespace rund::compute::resource {

struct Resource final {
  std::uint32_t id{};
  std::uint64_t bytes{};
  std::uint64_t alias_group{};
  std::uint64_t alias_offset_bytes{};
};

struct Access final {
  std::uint32_t node{};
  std::uint32_t resource{};
  AccessMode mode{AccessMode::Read};
  std::uint64_t offset_bytes{};
  // Contiguous byte-range form retained for the original resource-plan UX.
  // Leave this zero when authoring the exact strided form below.
  std::uint64_t size_bytes{};
  std::uint64_t element_bytes{};
  std::uint64_t element_count{};
  std::uint64_t stride_bytes{};
};

struct Lifetime final {
  std::uint32_t first_use{NoNode};
  std::uint32_t last_use{NoNode};
};

struct Dependency final {
  std::uint32_t before_node{};
  std::uint32_t after_node{};

  [[nodiscard]] constexpr bool
  operator==(const Dependency &) const noexcept = default;
};

struct Barrier final {
  std::uint64_t alias_group{};
  std::uint32_t before_resource{};
  std::uint32_t after_resource{};
  std::uint64_t offset_bytes{};
  std::uint64_t size_bytes{};
  std::uint64_t before_offset_bytes{};
  std::uint64_t before_element_bytes{};
  std::uint64_t before_element_count{};
  std::uint64_t before_stride_bytes{};
  std::uint64_t after_offset_bytes{};
  std::uint64_t after_element_bytes{};
  std::uint64_t after_element_count{};
  std::uint64_t after_stride_bytes{};
  std::uint32_t before_node{};
  std::uint32_t after_node{};
  AccessMode before{AccessMode::Read};
  AccessMode after{AccessMode::Read};

  [[nodiscard]] constexpr bool
  operator==(const Barrier &) const noexcept = default;
};

struct Plan final {
  std::vector<Lifetime> lifetimes;
  std::vector<Dependency> dependencies;
  std::vector<Barrier> barriers;
};

// Decide exact physical intersection for two validated logical-resource
// footprints. This is the canonical same-step admission predicate as well as
// the primitive used by analyze(); callers must not replace it with envelope
// overlap, which would falsely alias interleaved views.
[[nodiscard]] Result<bool> intersects(const Resource &left_resource,
                                      const Access &left,
                                      const Resource &right_resource,
                                      const Access &right) noexcept;

// Analyze canonical accesses in nondecreasing node order. An access is either
// the original contiguous [offset_bytes, offset_bytes + size_bytes) form, or
// the exact union of element_count half-open byte intervals
//
//   [offset + i * stride, offset + i * stride + element_bytes)
//
// for i in [0, element_count). Stride must be at least element width. Access
// offsets are relative to their logical resource; alias offsets place each
// logical resource in its alias group's byte address space. A dependency and
// its Barrier have the same index. The Barrier is one canonical exact witness
// for that ordered node pair, preferring a cross-resource conflict because it
// also proves alias-group visibility; further overlapping ranges cannot add
// ordering and are not retained. Barrier offset/size name the complete overlap
// interval when both footprints are contiguous. For a genuinely strided pair
// they name the canonical first overlap interval while the before/after
// footprint fields retain the complete witness. A zero-node plan is valid only
// with an empty access stream; it retains every declared resource with an
// explicit NoNode lifetime for canonical zero-work introspection.
[[nodiscard]] Result<Plan> analyze(std::span<const Resource> resources,
                                   std::span<const Access> accesses,
                                   std::uint32_t node_count);

} // namespace rund::compute::resource
