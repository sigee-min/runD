#pragma once

#include "../backing.hpp"
#include "../route.hpp"
#include "src/compute/virtual/graph_resident/workload.hpp"
#include "src/compute/virtual/state.hpp"

#include <rund/compute.hpp>
#include <rund/compute/virtual.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace rund_node_test_virtual::product::graph_resident_host {

using Workload = rund::compute::graph_resident_workload::Spec;
inline constexpr std::size_t InputCount = Workload::InputCount;
inline constexpr std::size_t StageCount = Workload::StageCount;
inline constexpr std::size_t FrameElements = Workload::FrameElements;
inline constexpr std::size_t PageCount = Workload::PageCount;
inline constexpr std::size_t ElementCount = Workload::ElementCount;
inline constexpr std::size_t BatchCount = Workload::BatchCount;
inline constexpr std::size_t LogicalBytes =
    ElementCount * sizeof(std::uint64_t);
inline constexpr std::size_t PageBytes = FrameElements * sizeof(std::uint64_t);

using Program = Workload::Program;
using Pipeline = rund::compute::VirtualPipeline<std::uint64_t(
    std::uint64_t, std::uint64_t, std::uint64_t)>;

struct Observation final {
  rund::compute::Status status =
      rund::compute::Status::fail(rund::compute::Reason::CompletionInvalid);
  rund::compute::Stats before{};
  rund::compute::Stats after{};
  std::uint64_t version_before{};
  std::uint64_t version_after{};
  std::uint64_t output_hash{};
  std::vector<std::uint64_t> output{};
  BackingFacts inputs[InputCount]{};
  BackingFacts output_facts{};
  RouteKind route_kind{RouteKind::Unknown};
  std::uint32_t owner_mask{};
  std::uint32_t owner_count{};
  bool output_match{};
  bool receipts_idle{};
  bool quarantine_empty{};
  bool distinct_backings{};
  bool callbacks_quiet{};
  rund::compute::detail::graph_reduce::FailLog failure{};
};

struct Case final {
  std::array<std::vector<std::uint64_t>, InputCount> values{};
  std::vector<std::uint64_t> expected{};
  std::array<std::shared_ptr<MemoryVirtualBacking>, InputCount> inputs{};
  std::shared_ptr<MemoryVirtualBacking> output{};
  Pipeline pipeline;
  std::shared_ptr<rund::compute::detail::VirtualPipelineState> state;
};

struct Preparation final {
  std::unique_ptr<Case> value{};
  int reason{};
};

[[nodiscard]] rund::compute::Result<Program>
build_program(const rund::compute::Device &);
[[nodiscard]] bool validate_program(const Program &) noexcept;
[[nodiscard]] Preparation prepare_case(const rund::compute::Device &);
[[nodiscard]] bool run_case(Case &, Observation &, rund::compute::Device &);
[[nodiscard]] bool capture_run(Case &, Observation &) noexcept;
[[nodiscard]] bool validate_case(const Case &, const Observation &) noexcept;

} // namespace rund_node_test_virtual::product::graph_resident_host
