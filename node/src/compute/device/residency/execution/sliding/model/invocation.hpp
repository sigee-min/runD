#pragma once

#include "../../../../../pipeline/residency/model.hpp"
#include "../../plan.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace rund::compute::detail::residency {

class Authority;
class SlidingOwner;
class ExecutionSlidingReceipt;

namespace execution {

inline constexpr std::size_t SlidingNativeCapacity = 4u;
inline constexpr std::size_t SlidingHostCapacity = UseCapacity;
inline constexpr std::size_t SlidingHostCellCapacity =
    BankCapacity * SlidingHostCapacity;
inline constexpr std::size_t SlidingPredecessorCapacity =
    TiledGraphDependencyCapacity;

enum class SlidingTopology : std::uint8_t { None, Direct, Graph };

struct SlidingCoordinate final {
  std::uint64_t ordinal{};
  std::uint64_t batch{};
  std::uint32_t stage{};
  [[nodiscard]] constexpr bool
  operator==(const SlidingCoordinate &) const noexcept = default;
};

struct SlidingProjection final {
  SlidingCoordinate coordinate{};
  std::uint64_t prefetch_epoch{};
  std::uint64_t ready_epoch{};
  std::uint64_t fetch_bytes{};
  std::uint64_t persist_bytes{};
  std::size_t use_count{};
  std::size_t fetch_count{};
  std::size_t persist_count{};
};

// Sole immutable PageUse and next-use topology consumed by Sliding.
class SlidingInvocation final {
public:
  [[nodiscard]] static SlidingInvocation
      direct(std::shared_ptr<const Plan>) noexcept;
  [[nodiscard]] static SlidingInvocation
  graph(std::shared_ptr<const residency::ResidencyPlan>, std::uint64_t,
        std::span<const std::uint64_t>) noexcept;

  [[nodiscard]] explicit operator bool() const noexcept;
  [[nodiscard]] SlidingTopology topology() const noexcept { return topology_; }
  [[nodiscard]] residency::Identity identity() const noexcept;
  [[nodiscard]] std::uint64_t count() const noexcept;
  [[nodiscard]] std::size_t scratch_capacity() const noexcept;
  [[nodiscard]] bool capacity_requirements(std::size_t &inputs,
                                           std::size_t &outputs) const noexcept;
  [[nodiscard]] bool host_ring_capacities(std::size_t &inputs,
                                          std::size_t &outputs) const noexcept;
  [[nodiscard]] bool host_input_requirement(std::size_t &inputs) const noexcept;
  [[nodiscard]] bool project(std::uint64_t, std::span<residency::PageUse>,
                             SlidingProjection &) const noexcept;
  [[nodiscard]] bool fetch_use(const SlidingProjection &,
                               std::span<const residency::PageUse>,
                               std::size_t) const noexcept;
  [[nodiscard]] bool persist_use(const SlidingProjection &,
                                 std::span<const residency::PageUse>,
                                 std::size_t) const noexcept;
  [[nodiscard]] bool use_bytes(const SlidingProjection &,
                               std::span<const residency::PageUse>, std::size_t,
                               std::uint64_t &) const noexcept;
  [[nodiscard]] bool
  predecessors(const SlidingProjection &,
               std::span<SlidingCoordinate, SlidingPredecessorCapacity>,
               std::size_t &) const noexcept;

private:
  friend class ::rund::compute::detail::residency::Authority;
  friend class ::rund::compute::detail::residency::SlidingOwner;
  SlidingTopology topology_{SlidingTopology::None};
  std::shared_ptr<const Plan> direct_{};
  std::shared_ptr<const residency::ResidencyPlan> graph_owner_{};
  residency::TiledGraphInvocation graph_{};
};

} // namespace execution
} // namespace rund::compute::detail::residency
