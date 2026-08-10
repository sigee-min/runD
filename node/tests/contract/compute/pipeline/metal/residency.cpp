#include "../local.hpp"

#include "src/accel/kernel/prepared.hpp"
#include "src/compute/backend.hpp"
#include "src/compute/pipeline/state.hpp"

#include <node/runtime/compute/access.hpp>
#include <rund/compute/virtual.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <span>
#include <vector>

namespace rund_node_test_pipeline {
namespace {

class AdmissionBacking final : public rund::compute::VirtualBacking {
public:
  explicit AdmissionBacking(const std::size_t bytes) : bytes_(bytes) {}

  [[nodiscard]] std::uint64_t size_bytes() const noexcept override {
    return bytes_.size();
  }

  [[nodiscard]] rund::compute::Status
  read(const std::uint64_t offset,
       const std::span<std::byte> output) noexcept override {
    ++reads_;
    if (offset > bytes_.size() || output.size() > bytes_.size() - offset) {
      return rund::compute::Status::fail(
          rund::compute::Reason::TransferInvalid);
    }
    std::memcpy(output.data(), bytes_.data() + offset, output.size());
    return rund::compute::Status::success();
  }

  [[nodiscard]] rund::compute::Status
  write(const std::uint64_t offset,
        const std::span<const std::byte> input) noexcept override {
    ++writes_;
    if (offset > bytes_.size() || input.size() > bytes_.size() - offset) {
      return rund::compute::Status::fail(
          rund::compute::Reason::TransferInvalid);
    }
    std::memcpy(bytes_.data() + offset, input.data(), input.size());
    return rund::compute::Status::success();
  }

  [[nodiscard]] std::uint64_t callbacks() const noexcept {
    return reads_ + writes_;
  }

private:
  std::vector<std::byte> bytes_;
  std::uint64_t reads_{};
  std::uint64_t writes_{};
};

[[nodiscard]] bool SameCurrent(const rund::compute::MemoryStats &left,
                               const rund::compute::MemoryStats &right) {
  return left.host.current == right.host.current &&
         left.frame.current == right.frame.current &&
         left.tile.current == right.tile.current &&
         left.resident.current == right.resident.current &&
         left.staging.current == right.staging.current &&
         left.device.current == right.device.current &&
         left.transfer.current == right.transfer.current;
}

[[nodiscard]] bool
AddedSubmissionBytes(const rund::compute::PipelinePlan &before,
                     const rund::compute::PipelinePlan &after,
                     const std::uint64_t bytes) {
  const auto added = [bytes](const std::uint64_t left,
                             const std::uint64_t right) {
    return left <= std::numeric_limits<std::uint64_t>::max() - bytes &&
           right == left + bytes;
  };
  return added(before.prepared_native_bytes, after.prepared_native_bytes) &&
         added(before.prepared_bytes, after.prepared_bytes) &&
         added(before.peak_bytes, after.peak_bytes) &&
         added(before.committed_peak_bytes, after.committed_peak_bytes) &&
         added(before.total_bytes, after.total_bytes) &&
         added(before.logical_bytes, after.logical_bytes) &&
         added(before.live_bytes, after.live_bytes) &&
         added(before.physical_bytes, after.physical_bytes);
}

[[nodiscard]] int CheckTransactionalStageRollback() {
  using namespace rund::compute;
  auto opened = open(Target::metal());
  if (!opened) {
    return opened.reason() == Reason::AdapterUnavailable ? 0 : 1;
  }
  Device device = std::move(opened).value();
  constexpr std::array<std::int32_t, 8u> input{1, 2, 3, 4, 5, 6, 7, 8};
  auto source = device.upload<std::int32_t>(input);
  auto state_buffer = device.buffer<std::int32_t>(input.size());
  auto program =
      on(device)
          .map<std::int32_t>("metal-residency-admission", input.size(),
                             [](auto value) { return value + 1; })
          .compile();
  if (!source || !state_buffer || !program) {
    return 2;
  }
  auto prepared = pipeline(device)
                      .state(*source, *state_buffer)
                      .then(*program, read(*source), write(*state_buffer))
                      .commit()
                      .prepare();
  if (!prepared) {
    return 3;
  }
  const std::shared_ptr<detail::PipelineState> state =
      detail::PipelineStateAccess::state(*prepared);
  if (state == nullptr || state->device == nullptr ||
      state->device->ops == nullptr ||
      state->device->ops->prepare_pipeline_residency == nullptr ||
      !state->transactional) {
    return 4;
  }
  bool supported = false;
  if (!rund::node::accel::detail::QueryPreparedKernelPipelineResidency(
          state->prepared, supported)) {
    return 5;
  }
  if (!supported) {
    return 0;
  }
  const PipelinePlan before_plan = prepared->plan();
  const MemoryStats before_memory = prepared->memory();
  const DevicePipelineMemoryReport before_report = device.pipeline_memory();
  if (before_report.available_bytes == 0u) {
    return 6;
  }
  rund::storage::Reservation capacity_gate =
      state->device->pipeline_memory_budget.reserve(
          before_report.available_bytes);
  std::shared_ptr<void> primary_candidate;
  std::uint64_t primary_bytes = 0u;
  const auto primary =
      rund::node::accel::detail::StagePreparedKernelPipelineResidency(
          state->prepared, primary_candidate, primary_bytes);
  rund::node::accel::detail::PreparedKernelPipeline invalid{};
  std::shared_ptr<void> alternate_candidate;
  std::uint64_t alternate_bytes = 0u;
  const auto alternate =
      rund::node::accel::detail::StagePreparedKernelPipelineResidency(
          invalid, alternate_candidate, alternate_bytes);
  primary_candidate.reset();
  alternate_candidate.reset();
  if (!capacity_gate.refund() || !primary || primary_bytes == 0u || alternate ||
      state->residency_submission_memory || prepared->plan() != before_plan ||
      !SameCurrent(prepared->memory(), before_memory) ||
      device.pipeline_memory().committed_bytes !=
          before_report.committed_bytes ||
      device.pipeline_memory().preparing_bytes != 0u) {
    return 7;
  }

  const Status retried = state->device->ops->prepare_pipeline_residency(*state);
  const std::uint64_t admitted =
      state->residency_submission_memory.usage().allocated_bytes;
  const PipelinePlan after_plan = prepared->plan();
  const MemoryStats after_memory = prepared->memory();
  const std::uint64_t publication =
      state->publication == nullptr
          ? 0u
          : state->publication->publication_memory.usage().allocated_bytes;
  const std::uint64_t total_admission =
      state->private_memory.usage().allocated_bytes + publication + admitted;
  if (!retried || admitted == 0u ||
      !state->residency_submission_memory.committed() ||
      !AddedSubmissionBytes(before_plan, after_plan, admitted) ||
      after_plan.physical_bytes != after_plan.peak_bytes ||
      after_plan.committed_peak_bytes != total_admission ||
      device.pipeline_memory().committed_bytes != total_admission ||
      after_memory.device.current != before_memory.device.current + admitted ||
      !prepared->run()) {
    return 8;
  }
  return 0;
}

[[nodiscard]] int CheckBudgetMinusOneRollback() {
  using namespace rund::compute;
  constexpr std::size_t elements = 67u;
  constexpr std::size_t page_elements = 16u;
  constexpr std::size_t bytes = elements * sizeof(std::int32_t);
  std::uint64_t exact_capacity = 0u;
  {
    auto opened = open(Target::metal());
    if (!opened) {
      return opened.reason() == Reason::AdapterUnavailable ? 0 : 1;
    }
    Device device = std::move(opened).value();
    auto program =
        on(device)
            .map<std::int32_t>("metal-residency-budget-probe", page_elements,
                               [](auto value) { return value + 1; })
            .compile();
    auto input_backing = std::make_shared<AdmissionBacking>(bytes);
    auto output_backing = std::make_shared<AdmissionBacking>(bytes);
    auto input = virtual_buffer<std::int32_t>(elements, input_backing);
    auto output = virtual_buffer<std::int32_t>(elements, output_backing);
    auto prepared =
        program && input && output
            ? virtual_pipeline(*program, *input, *output,
                               ResidencyConfig{.slots = 2u})
            : Result<VirtualPipeline<std::int32_t(std::int32_t)>>::fail(
                  Reason::PipelineInvalid);
    if (!prepared) {
      return 2;
    }
    exact_capacity = prepared->plan().committed_peak_bytes;
    if (exact_capacity == 0u ||
        device.pipeline_memory().committed_bytes != exact_capacity) {
      return 3;
    }
  }

  auto limited = open(Target::metal(),
                      DevicePipelineMemoryLimit{.bytes = exact_capacity - 1u});
  if (!limited) {
    return 4;
  }
  Device device = std::move(limited).value();
  auto program =
      on(device)
          .map<std::int32_t>("metal-residency-budget-minus-one", page_elements,
                             [](auto value) { return value + 1; })
          .compile();
  auto input_backing = std::make_shared<AdmissionBacking>(bytes);
  auto output_backing = std::make_shared<AdmissionBacking>(bytes);
  auto input = virtual_buffer<std::int32_t>(elements, input_backing);
  auto output = virtual_buffer<std::int32_t>(elements, output_backing);
  if (!program || !input || !output) {
    return 5;
  }
  const MemoryStats before_memory = device.memory();
  auto rejected =
      virtual_pipeline(*program, *input, *output, ResidencyConfig{.slots = 2u});
  const MemoryStats after_memory = device.memory();
  const DevicePipelineMemoryReport after_report = device.pipeline_memory();
  if (rejected || rejected.reason() != Reason::DevicePipelineMemoryCapacity ||
      !SameCurrent(before_memory, after_memory) ||
      after_report.committed_bytes != 0u ||
      after_report.preparing_bytes != 0u || input_backing->callbacks() != 0u ||
      output_backing->callbacks() != 0u) {
    return 6;
  }
  return 0;
}

} // namespace

int CheckMetalResidencyAdmission() {
  const int transactional = CheckTransactionalStageRollback();
  if (transactional != 0) {
    return transactional;
  }
  const int capacity = CheckBudgetMinusOneRollback();
  return capacity == 0 ? 0 : 100 + capacity;
}

} // namespace rund_node_test_pipeline
