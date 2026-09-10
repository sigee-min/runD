#pragma once

#include "../local.hpp"

#include "src/compute/device/residency/execution/stream.hpp"
#include "src/compute/device/residency/execution/window.hpp"
#include "src/compute/device/residency/registry.hpp"
#include "src/compute/pipeline/execution/window.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace rund_node_test_pipeline_residency::window {

namespace detail = rund::compute::detail;
namespace execution = detail::residency::execution;
namespace residency = detail::residency;
using rund::compute::Reason;
using rund::compute::Status;

[[nodiscard]] residency::FrameRegion region(residency::FrameTier tier,
                                            residency::FrameRole role,
                                            std::uint32_t first);

[[nodiscard]] execution::SealResult plan(std::uint64_t page_count = 4u);

[[nodiscard]] bool frames(residency::Authority &authority);

[[nodiscard]] execution::Release
release(const residency::ExecutionLease &lease, std::uint64_t epoch,
        Status status = Status::success(),
        execution::TerminalKind terminal = execution::TerminalKind::Known,
        bool dispatched = true, bool completed = true, bool may_write = true);

[[nodiscard]] detail::PipelineWindowRelease
pipeline(execution::Release value);

[[nodiscard]] execution::WindowEvidence
chunk(const residency::ExecutionLease &lease, std::uint64_t first,
      std::size_t count, Status status = Status::success(),
      execution::TerminalKind terminal = execution::TerminalKind::Known);

[[nodiscard]] int CheckStreamWindow();
[[nodiscard]] int CheckScheduleWindow();

} // namespace rund_node_test_pipeline_residency::window
