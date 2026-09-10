#pragma once

#include "src/compute/device/residency/execution/sliding.hpp"
#include "src/compute/device/residency/registry.hpp"
#include "src/compute/device/residency/registry/sliding_owner.hpp"
#include "src/compute/pipeline/residency/planner.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace rund_node_test_pipeline_residency::sliding_detail {

namespace execution = rund::compute::detail::residency::execution;
namespace residency = rund::compute::detail::residency;

struct GraphFixture final {
  std::shared_ptr<const residency::ResidencyPlan> owner{};
  std::array<std::uint64_t, 3u> bytes{};
};

[[nodiscard]] GraphFixture graph_fixture();

[[nodiscard]] residency::FrameRegion region(residency::FrameTier tier,
                                            residency::FrameRole role,
                                            std::uint32_t first,
                                            std::uint32_t count);

[[nodiscard]] std::shared_ptr<const execution::Plan>
direct_plan(std::uint64_t pages, std::uint32_t capacity,
            std::uint32_t host_capacity = execution::SlidingHostCapacity,
            std::uint64_t input_retain = residency::NeverUse,
            std::uint64_t input_logical_bytes = 0u,
            execution::FetchFill fill = execution::FetchFill::None,
            std::uint64_t input_page_bytes = 16u,
            std::uint64_t input_payload_bytes = 16u,
            std::uint64_t input_prefix_bytes = 0u,
            std::uint64_t input_suffix_bytes = 0u,
            std::uint64_t fill_element_bytes = 0u,
            bool canonical_window = false,
            std::uint64_t expanded_boundary_extent = 0u,
            std::uint64_t canonical_boundary_extent = 0u);

[[nodiscard]] bool register_direct(residency::Authority &authority,
                                   std::uint32_t capacity,
                                   std::uint32_t host_capacity);

struct EpochWork final {
  execution::SlidingProjection projection{};
  execution::SlidingTicket native{};
  std::shared_ptr<execution::SlidingNative> physical_native{};
  std::array<residency::PageUse, 2u * execution::UseCapacity> uses{};
};

[[nodiscard]] bool fetch(execution::Sliding &sliding, std::uint64_t epoch,
                         EpochWork &work,
                         execution::SlidingTicket *first = nullptr);

[[nodiscard]] bool fetch_bound(residency::Authority &authority,
                               const execution::Plan &plan,
                               execution::Sliding &sliding, std::uint64_t epoch,
                               EpochWork &work);

[[nodiscard]] bool admit(execution::Sliding &sliding, EpochWork &work);

[[nodiscard]] bool admit_bound(residency::Authority &authority,
                               const execution::Plan &plan,
                               execution::Sliding &sliding, EpochWork &work);

[[nodiscard]] bool drain_bound(residency::Authority &authority,
                               const execution::Plan &plan,
                               execution::Sliding &sliding, EpochWork &work,
                               bool persist);

[[nodiscard]] bool drain(execution::Sliding &sliding, EpochWork &work,
                         bool persist,
                         execution::SlidingTicket *retained = nullptr);

} // namespace rund_node_test_pipeline_residency::sliding_detail

namespace rund_node_test_pipeline_residency {
[[nodiscard]] int CheckSlidingLifecycle();
[[nodiscard]] int CheckSlidingGraphProjection();
[[nodiscard]] int CheckSlidingGraphReplay();
[[nodiscard]] int CheckSlidingTerminalUnknown();
[[nodiscard]] int CheckSlidingTerminalFrontier();
[[nodiscard]] int CheckSlidingTerminalFailures();
[[nodiscard]] int CheckSlidingGraphAdmission();
[[nodiscard]] int CheckSlidingTerminalSuffix();
[[nodiscard]] int CheckSlidingGraphTopology();
[[nodiscard]] int CheckSlidingModel();

} // namespace rund_node_test_pipeline_residency
