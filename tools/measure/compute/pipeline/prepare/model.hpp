#pragma once

#include "../model.hpp"

#include "src/compute/backend.hpp"
#include "src/compute/pipeline/state.hpp"

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace rund::measure::compute::preparation_memory {

inline constexpr std::size_t Maximum = 516096u;
inline constexpr std::size_t Tile = 8192u;
inline constexpr std::size_t Inner = 64u;
inline constexpr std::size_t SecondInner = 1u;
inline constexpr std::size_t OrdinaryIterations = 64u;
inline constexpr std::size_t Outer = Maximum / Tile;
inline constexpr std::size_t Domain = 64u;
inline constexpr std::size_t SeedScanMapPairs = 500u;
inline constexpr std::size_t PreparedTemplates =
    (Outer + 2u + 3u) + 1u + 3u + (Outer + SecondInner + 3u) + 1u;
inline constexpr std::size_t PreparedCommands =
    Outer * (Inner + 2u) + 1u + OrdinaryIterations +
    Outer * (SecondInner + 2u) + 1u;

static_assert(Maximum % Tile == 0u);
static_assert(Outer == 63u);
static_assert(PreparedTemplates == 140u);
static_assert(PreparedCommands == 4413u);

struct PreparationMemoryObservation final {
  ::rund::compute::PipelinePlan plan{};
  ::rund::compute::MemoryStats memory{};
  ::rund::compute::MemoryEntry largest_retained_group{};
  ::rund::node::accel::detail::PreparedPipelineMemory backend_memory{};
  ::rund::node::accel::detail::PreparedKernelPipelineReservation
      backend_limit{};
  ::rund::node::accel::detail::PreparedKernelPipelineReservation
      backend_consumed{};
  ::rund::compute::Code code{::rund::compute::Code::Ok};
  ::rund::compute::Reason reason{::rund::compute::Reason::Ok};
  ::rund::compute::Location location{};
  std::string_view error{};
  const char *status{"invalid"};
  double plan_wall_us{};
  std::uint64_t plan_current_rss_before{};
  std::uint64_t plan_current_rss_after{};
  std::uint64_t plan_rss_before{};
  std::uint64_t plan_rss_after{};
  double prepare_wall_us{};
  std::uint64_t prepare_current_rss_before{};
  std::uint64_t prepare_current_rss_after{};
  std::uint64_t prepare_rss_before{};
  std::uint64_t prepare_rss_after{};
  std::uint64_t prepare_host_allocation_count{};
  std::uint64_t prepare_host_allocation_bytes{};
  bool prepare_rss_within_committed_peak{};
  bool plan_contract{};
  bool short_budget_checked{};
  bool short_budget_rejected{};
  bool short_budget_no_allocation{};
  ::rund::compute::Reason short_budget_reason{::rund::compute::Reason::Ok};
  bool prepare_attempted{};
  bool prepare_ok{};
  bool plan_frozen{};
  bool memory_contract{};
  bool backend_observed{};
  bool backend_reservation_contract{};
  bool backend_telemetry_contract{};
  bool retained_group_available{};
  bool precise_failure{};
  bool contract{};
};

} // namespace rund::measure::compute::preparation_memory
