#pragma once

#include <rund/compute/fixed.hpp>
#include <rund/compute/graph/capacity.hpp>
#include <rund/compute/resource.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace rund::compute::graph {

struct Fingerprint final {
  std::uint64_t hi{};
  std::uint64_t lo{};

  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return hi != 0u || lo != 0u;
  }
  [[nodiscard]] constexpr bool
  operator==(const Fingerprint &) const noexcept = default;
};

enum class Value : unsigned char {
  I32,
  U32,
  I64,
  U64,
  Fixed,
};

enum class Visibility : unsigned char {
  Input,
  Output,
  Internal,
};

enum class Operation : unsigned char {
  Map,
  Scan,
  SegmentedScan,
  SegmentedReduce,
  Sort,
  Argsort,
  Compact,
  Gather,
  Histogram,
  Partition,
  Reduce,
  Scatter,
  Stencil,
  Transform,
  Matrix,
  Factor,
  Solve,
  Spectrum,
  ScatterReduce,
};

struct Resource final {
  std::uint32_t id{};
  Value type{Value::I32};
  std::uint8_t integer_bits{};
  std::uint8_t fraction_bits{};
  Rounding rounding{Rounding::NearestEven};
  Overflow overflow{Overflow::Saturate};
  Approximation approximation{Approximation::Exact};
  Visibility visibility{Visibility::Internal};
  std::uint64_t elements{};
  std::uint64_t element_bytes{};
  std::uint64_t bytes{};
  // Resource IDs for active-prefix, count-lineage, and pointwise destructive
  // alias proofs. Zero means that proof does not apply.
  std::uint32_t active{};
  std::uint32_t parent{};
  std::uint32_t source{};
  std::uint64_t alias_group{};
  std::uint64_t alias_offset_bytes{};
  // A reset resource begins its exact first-write frontier from all-byte zero.
  // This field identifies that writer; NoNode means no reset.
  std::uint32_t reset_node{::rund::compute::resource::NoNode};
  std::uint32_t first_use{::rund::compute::resource::NoNode};
  std::uint32_t last_use{::rund::compute::resource::NoNode};

  [[nodiscard]] constexpr bool requires_reset() const noexcept {
    return reset_node != ::rund::compute::resource::NoNode;
  }
};

struct Access final {
  std::uint32_t resource{};
  ::rund::compute::resource::AccessMode mode{
      ::rund::compute::resource::AccessMode::Read};
  std::uint64_t offset_bytes{};
  std::uint64_t size_bytes{};
  std::uint64_t element_bytes{};
  std::uint64_t element_count{};
  std::uint64_t stride_bytes{};
};

struct Node final {
  std::uint32_t index{};
  Operation operation{Operation::Map};
  std::uint64_t elements{};
  std::vector<Access> accesses{};
  std::vector<std::uint32_t> dependencies{};
};

struct Barrier final {
  // One deterministic witness for an executable visibility boundary. Info
  // retains at most one row per after_node; Node::dependencies owns the full
  // ordered predecessor set.
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
  ::rund::compute::resource::AccessMode before{
      ::rund::compute::resource::AccessMode::Read};
  ::rund::compute::resource::AccessMode after{
      ::rund::compute::resource::AccessMode::Read};
};

struct MemoryPlan final {
  // Authored sum, maximum closed-interval live sum, retained physical sum,
  // and retained Buffer-owner count for graph-internal storage.
  std::uint64_t logical_bytes{};
  std::uint64_t live_bytes{};
  std::uint64_t physical_bytes{};
  std::uint64_t allocation_count{};
  // Exact payload bytes and canonical physical reset ranges applied once per
  // logical Program invocation. Backend command batching is runtime evidence.
  std::uint64_t reset_bytes{};
  std::uint64_t reset_count{};

  [[nodiscard]] constexpr bool
  operator==(const MemoryPlan &) const noexcept = default;
};

struct Info final {
  Fingerprint fingerprint{};
  std::uint64_t read_bytes{};
  // The checked pre-fusion schedule and the executable post-fusion schedule.
  // nodes stores exactly lowered_nodes rows. authored_nodes participates in
  // the fingerprint so a cached Program cannot publish another compiling
  // Flow's admitted schedule count.
  std::uint64_t authored_nodes{};
  std::uint64_t lowered_nodes{};
  MemoryPlan memory{};
  std::vector<Resource> resources{};
  std::vector<Node> nodes{};
  // Canonical boundary order, with at most nodes.size() - 1 rows.
  std::vector<Barrier> barriers{};
  std::vector<std::uint32_t> inputs{};
  std::vector<std::uint32_t> outputs{};
};

} // namespace rund::compute::graph
