#pragma once

#include "../../../accel/kernel/residency/window.hpp"

#include "../../device/residency/execution/model.hpp"

#include "../../../accel/kernel/prepared/pipeline.hpp"
#include "../../../accel/kernel/prepared/interface/evidence.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace rund::compute::detail {

struct PipelineState;

struct PipelineExecutionAttempt final {
  std::uint64_t epoch{};
  std::uint64_t attempt_generation{};
  std::size_t issued_steps{};
  bool started{};
  bool signalled{};
};

struct PipelineExecutionTerminal final {
  Status status{Status::fail(Reason::PipelineInvalid)};
  std::uint64_t published_generation{};
  bool terminal_published{};
  bool reseeded{};
};

struct PipelineExecutionSnapshot final {
  std::array<std::uint64_t, residency::execution::BankCapacity> generation{};
  std::array<std::uint8_t, residency::execution::BankCapacity> parity{};
};

[[nodiscard]] const node::accel::detail::PreparedKernelPipeline *
prepared_pipeline_for(const PipelineState &, std::uint8_t parity) noexcept;

[[nodiscard]] Status snapshot_pipeline_execution(
    const std::array<std::shared_ptr<PipelineState>,
                     residency::execution::BankCapacity> &,
    PipelineExecutionSnapshot &) noexcept;

// Sole Compute Pipeline-attempt authority shared by bounded windows and
// arbitrary-Q schedules. These functions own PipelineState locking;
// native adapters never start, finish, publish, or reseed Compute state.
[[nodiscard]] Status begin_pipeline_execution_attempt(
    PipelineState &, const residency::execution::Node &,
    std::span<const std::uint32_t> locals, std::uint32_t control_generation,
    PipelineExecutionAttempt &) noexcept;

[[nodiscard]] PipelineExecutionTerminal complete_pipeline_execution_attempt(
    PipelineState &, PipelineExecutionAttempt &,
    const node::accel::detail::BackendResidencyWindowReceipt &,
    const node::accel::detail::PreparedPipelineEvidence &) noexcept;

void reject_pipeline_execution_attempt(PipelineState &,
                                       PipelineExecutionAttempt &,
                                       Status failure) noexcept;

} // namespace rund::compute::detail
