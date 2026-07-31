#include <node/runtime/compute/access.hpp>
#include <rund/compute.hpp>

#include "../../../src/compute/device/state.hpp"
#include "../../../src/compute/job/state.hpp"
#include "../../../src/accel/kernel/batch/plan.hpp"
#include "../target/selection.hpp"
#include "allocation.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <optional>
#include <string_view>
#include <type_traits>
#include <vector>

namespace {

using rund::compute::Backend;
using rund::compute::Batch;
using rund::compute::Code;
using rund::compute::Reason;
using rund::compute::Stats;
using rund::compute::Status;

static_assert(Batch::capacity() == 64u);
static_assert(!std::is_copy_constructible_v<Batch>);
static_assert(std::is_nothrow_move_constructible_v<Batch>);

[[nodiscard]] bool SharedFieldsAreZero(const Stats &stats) noexcept {
  return stats.command_submits == 0u && stats.command_capacity == 0u &&
         stats.command_inflight_peak == 0u &&
         stats.command_capacity_rejections == 0u && stats.kernel_ns == 0u &&
         stats.kernel_samples == 0u && stats.submit_wait_ns == 0u;
}

struct PreparedBusyProbe final {
  Batch *batch = nullptr;
  Status result = Status::fail(Reason::BackendFailed);
  bool called = false;
};

[[nodiscard]] bool CheckPackedAliasPlan() {
  using rund::node::accel::detail::BatchMapAliasFree;
  using rund::node::accel::detail::BatchMapBindingCapacity;
  using rund::node::accel::detail::BatchMapCapacity;
  using rund::node::accel::detail::BatchMapView;

  std::array<BatchMapView, BatchMapCapacity> views{};
  std::uint64_t input_id = 1u;
  std::uint64_t output_id = 2049u;
  for (BatchMapView &view : views) {
    view.input_count = BatchMapBindingCapacity;
    view.output_count = BatchMapBindingCapacity;
    for (std::size_t index = 0u; index < BatchMapBindingCapacity; ++index) {
      view.input_ids[index] = input_id++;
      view.output_ids[index] = output_id++;
    }
  }
  const std::span<const BatchMapView> all{views};
  if (!BatchMapAliasFree(all, 0u, all.size())) {
    return false;
  }
  views.back().output_ids.back() = views.front().output_ids.front();
  if (BatchMapAliasFree(all, 0u, all.size())) {
    return false;
  }
  views.back().output_ids.back() = views.front().input_ids.front();
  return !BatchMapAliasFree(all, 0u, all.size());
}

void RunWhilePrepared(void *const raw) noexcept {
  auto &probe = *static_cast<PreparedBusyProbe *>(raw);
  probe.called = true;
  probe.result = probe.batch->run();
}

[[nodiscard]] int CheckTypedAdmission(const Backend backend) {
  Batch empty{};
  const auto empty_run = empty.run();
  if (empty_run || empty_run.reason() != Reason::BatchEmpty ||
      empty_run.code() != Code::Invalid ||
      empty_run.error() != "compute_batch_empty" ||
      empty_run.exit_code() != 1) {
    return 1;
  }

  constexpr std::array<std::int32_t, 1u> input{3};
  if (backend == Backend::Cpu) {
    auto program =
        rund::compute::on(rund::node::test_contract::target_for(backend))
            .map<std::int32_t>("batch-cpu", input.size(),
                               [](auto value) { return value + 1; })
            .compile();
    auto job = program
                   ? program->resident(input)
                   : decltype(program->resident(input))::fail(program.reason());
    if (!job) {
      return 2;
    }
    Batch cpu{};
    const auto rejected = cpu.add(*job);
    return !rejected && rejected.reason() == Reason::BatchCpuUnsupported &&
                   rejected.code() == Code::Unsupported &&
                   rejected.error() == "compute_batch_cpu_unsupported"
               ? 0
               : 3;
  }

  auto device =
      rund::compute::open(rund::node::test_contract::target_for(backend));
  auto other_device =
      rund::compute::open(rund::node::test_contract::target_for(backend));
  if (!device || !other_device) {
    return 4;
  }
  auto program = rund::compute::on(*device)
                     .map<std::int32_t>("batch-admit", input.size(),
                                        [](auto value) { return value + 1; })
                     .compile();
  auto other_program =
      rund::compute::on(*other_device)
          .map<std::int32_t>("batch-other", input.size(),
                             [](auto value) { return value + 1; })
          .compile();
  if (!program || !other_program) {
    return 5;
  }
  auto first = program->resident(input);
  auto other = other_program->resident(input);
  if (!first || !other) {
    return 6;
  }

  Batch duplicate{};
  if (!duplicate.add(*first)) {
    return 7;
  }
  const auto duplicate_result = duplicate.add(*first);
  if (duplicate_result || duplicate_result.reason() != Reason::BatchDuplicate ||
      duplicate_result.code() != Code::Binding ||
      duplicate_result.error() != "compute_batch_duplicate") {
    return 8;
  }
  const auto mismatch = duplicate.add(*other);
  if (mismatch || mismatch.reason() != Reason::BatchDeviceMismatch ||
      mismatch.code() != Code::Binding ||
      mismatch.error() != "compute_batch_device_mismatch") {
    return 9;
  }

  auto moved_source = program->resident(input);
  if (!moved_source) {
    return 10;
  }
  auto moved = std::move(*moved_source);
  Batch invalid{};
  const auto prepared = invalid.add(*moved_source);
  if (prepared || prepared.reason() != Reason::BatchPreparedInvalid ||
      prepared.code() != Code::Invalid ||
      prepared.error() != "compute_batch_prepared_invalid") {
    return 11;
  }

  using Job = decltype(moved);
  std::array<std::optional<Job>, Batch::capacity() + 1u> owners{};
  Batch full{};
  for (std::size_t index = 0u; index < owners.size(); ++index) {
    auto resident = program->resident(input);
    if (!resident) {
      std::fprintf(stderr,
                   "batch resident admission backend=%u index=%zu reason=%u "
                   "error=%.*s\n",
                   static_cast<unsigned>(backend), index,
                   static_cast<unsigned>(resident.reason()),
                   static_cast<int>(resident.error().size()),
                   resident.error().data());
      return 12;
    }
    owners[index].emplace(std::move(*resident));
    const auto added = full.add(*owners[index]);
    if (index < Batch::capacity()) {
      if (!added) {
        return 13;
      }
    } else if (added || added.reason() != Reason::BatchCapacity ||
               added.code() != Code::Capacity ||
               added.error() != "compute_batch_capacity") {
      return 14;
    }
  }
  return 0;
}

[[nodiscard]] int CheckExecution(const Backend backend) {
  constexpr std::array<std::int32_t, 4u> signed_input{1, 2, 3, 4};
  constexpr std::array<std::uint32_t, 3u> unsigned_input{5u, 6u, 7u};
  auto device =
      rund::compute::open(rund::node::test_contract::target_for(backend));
  if (!device) {
    return 20;
  }
  auto signed_program =
      rund::compute::on(*device)
          .map<std::int32_t>("batch-signed", signed_input.size(),
                             [](auto value) { return value * 2 + 1; })
          .compile();
  auto unsigned_program =
      rund::compute::on(*device)
          .map<std::uint32_t>("batch-unsigned", unsigned_input.size(),
                              [](auto value) { return value + 9u; })
          .compile();
  if (!signed_program || !unsigned_program) {
    return 21;
  }
  auto signed_job = signed_program->resident(signed_input);
  auto unsigned_job = unsigned_program->resident(unsigned_input);
  if (!signed_job || !unsigned_job) {
    return 22;
  }

  if (!signed_job->run() || !unsigned_job->run()) {
    return 23;
  }
  auto serial_signed = signed_job->read();
  auto serial_unsigned = unsigned_job->read();
  if (!serial_signed || !serial_unsigned) {
    return 24;
  }
  const Stats signed_serial = signed_job->stats();
  const Stats unsigned_serial = unsigned_job->stats();

  node_compute_allocation::Start();
  const auto signed_probe = signed_job->run();
  const auto unsigned_probe = unsigned_job->run();
  node_compute_allocation::Stop();
  if (!signed_probe || !unsigned_probe) {
    return 25;
  }
  const std::uint64_t serial_allowance = node_compute_allocation::Count();

  Batch batch{};
  node_compute_allocation::Start();
  const auto signed_added = batch.add(*signed_job);
  const auto unsigned_added = batch.add(*unsigned_job);
  node_compute_allocation::Stop();
  if (!signed_added || !unsigned_added ||
      node_compute_allocation::Count() != 0u || batch.size() != 2u ||
      batch.stats().backend != backend) {
    return 26;
  }
  if (!batch.run()) {
    return 27;
  }
  node_compute_allocation::Start();
  const auto warm = batch.run();
  node_compute_allocation::Stop();
  if (!warm || node_compute_allocation::Count() > serial_allowance) {
    std::fprintf(
        stderr, "batch allocation backend=%u batch=%llu serial=%llu\n",
        static_cast<unsigned>(backend),
        static_cast<unsigned long long>(node_compute_allocation::Count()),
        static_cast<unsigned long long>(serial_allowance));
    return 28;
  }

  const Stats shared = batch.stats();
  const Stats signed_batched_before_read = signed_job->stats();
  const Stats unsigned_batched_before_read = unsigned_job->stats();
  if (shared.backend != backend || shared.command_submits != 1u ||
      shared.dispatches != signed_batched_before_read.dispatches +
                               unsigned_batched_before_read.dispatches ||
      shared.graph_hash != 0u || shared.output_hash != 0u ||
      !SharedFieldsAreZero(signed_batched_before_read) ||
      !SharedFieldsAreZero(unsigned_batched_before_read) ||
      signed_batched_before_read.graph_hash != signed_serial.graph_hash ||
      unsigned_batched_before_read.graph_hash != unsigned_serial.graph_hash ||
      signed_batched_before_read.dispatches != signed_serial.dispatches ||
      unsigned_batched_before_read.dispatches != unsigned_serial.dispatches) {
    return 29;
  }

  auto batched_signed = signed_job->read();
  auto batched_unsigned = unsigned_job->read();
  if (!batched_signed || !batched_unsigned ||
      *batched_signed != *serial_signed ||
      *batched_unsigned != *serial_unsigned ||
      signed_job->stats().output_hash != signed_serial.output_hash ||
      unsigned_job->stats().output_hash != unsigned_serial.output_hash) {
    return 30;
  }

  const auto signed_state =
      rund::compute::detail::JobAccess::state(*signed_job);
  const Stats preserved = signed_job->stats();
  {
    std::lock_guard lock{signed_state->gate};
    signed_state->phase = rund::compute::detail::JobPhase::Running;
  }
  const auto busy = batch.run();
  {
    std::lock_guard lock{signed_state->gate};
    signed_state->phase = rund::compute::detail::JobPhase::Idle;
  }
  if (busy || busy.reason() != Reason::BatchBusy ||
      busy.error() != "compute_batch_busy" ||
      signed_job->stats().graph_hash != preserved.graph_hash ||
      signed_job->stats().output_hash != preserved.output_hash) {
    return 31;
  }

  const auto *const accel =
      rund::compute::detail::accel_device(*signed_state->program->device);
  if (accel == nullptr) {
    return 32;
  }
  const Stats batch_preserved = batch.stats();
  const Stats unsigned_preserved = unsigned_job->stats();
  std::array<const rund::node::accel::detail::PreparedKernelRun *, 1u> prepared{
      &signed_state->prepared};
  std::array<rund::AccelEvidence, 1u> evidence{};
  std::shared_ptr<void> direct_workspace{};
  PreparedBusyProbe probe{.batch = &batch};
  const auto direct = rund::node::accel::detail::RunPreparedKernelBatch(
      accel->context, prepared, evidence, direct_workspace, RunWhilePrepared,
      &probe);
  if (!direct.check.ok || !probe.called || probe.result ||
      probe.result.reason() != Reason::BatchBusy ||
      signed_job->stats().graph_hash != preserved.graph_hash ||
      signed_job->stats().output_hash != preserved.output_hash ||
      unsigned_job->stats().graph_hash != unsigned_preserved.graph_hash ||
      unsigned_job->stats().output_hash != unsigned_preserved.output_hash ||
      batch.stats().command_submits != batch_preserved.command_submits) {
    return 33;
  }

  Batch moved{std::move(batch)};
  auto signed_owner = std::move(*signed_job);
  constexpr std::array<std::int32_t, 4u> next_input{8, 7, 6, 5};
  if (batch.size() != 0u || moved.size() != 2u ||
      !signed_owner.write(next_input) || !moved.run()) {
    return 34;
  }
  auto next_output = signed_owner.read();
  if (!next_output ||
      *next_output != std::vector<std::int32_t>{17, 15, 13, 11} ||
      moved.stats().command_submits != 1u) {
    return 35;
  }
  return 0;
}

[[nodiscard]] int CheckOverflow(const Backend backend) {
  constexpr std::array<std::uint32_t, 2u> overflow{
      std::numeric_limits<std::uint32_t>::max(), 1u};
  constexpr std::array<std::uint32_t, 2u> valid{1u, 2u};
  auto device =
      rund::compute::open(rund::node::test_contract::target_for(backend));
  if (!device) {
    return 40;
  }
  auto scan_program = rund::compute::on(*device)
                          .map<std::uint32_t>("batch-overflow", overflow.size(),
                                              [](auto value) { return value; })
                          .scan(rund::compute::Scan::InclusiveSum)
                          .compile();
  auto map_program =
      rund::compute::on(*device)
          .map<std::uint32_t>("batch-survivor", valid.size(),
                              [](auto value) { return value + 1u; })
          .compile();
  if (!scan_program || !map_program) {
    return 41;
  }
  auto scan = scan_program->resident(overflow);
  auto survivor = map_program->resident(valid);
  if (!scan || !survivor) {
    return 42;
  }
  const auto serial = scan->run();
  if (serial || serial.reason() != Reason::ScanSumOverflow ||
      !scan->write(overflow)) {
    return 43;
  }
  const std::uint64_t serial_graph = scan->stats().graph_hash;

  Batch batch{};
  if (!batch.add(*scan) || !batch.add(*survivor)) {
    return 44;
  }
  const auto result = batch.run();
  if (result || result.reason() != Reason::ScanSumOverflow ||
      result.error() != "compute_scan_sum_overflow" ||
      batch.stats().command_submits != 1u ||
      scan->stats().command_submits != 0u ||
      survivor->stats().command_submits != 0u ||
      scan->stats().graph_hash != serial_graph) {
    return 45;
  }
  auto output = survivor->read();
  return output && *output == std::vector<std::uint32_t>{2u, 3u} ? 0 : 46;
}

[[nodiscard]] int CheckReset(const Backend backend) {
  auto device =
      rund::compute::open(rund::node::test_contract::target_for(backend));
  if (!device) {
    return 60;
  }
  auto program =
      rund::compute::on(*device)
          .map<std::uint32_t>("batch-reset", 1u,
                              [](auto value) { return value; })
          .scatter(1u, {.count = 2u})
          .compile();
  constexpr std::array<std::uint32_t, 1u> first_value{7u};
  constexpr std::array<std::uint32_t, 1u> first_index{0u};
  constexpr std::array<std::uint32_t, 1u> second_value{8u};
  constexpr std::array<std::uint32_t, 1u> second_index{1u};
  if (!program) {
    return 61;
  }
  auto first = program->resident(first_value, first_index);
  auto second = program->resident(second_value, second_index);
  Batch batch{};
  if (!first || !second || !batch.add(*first) || !batch.add(*second) ||
      !batch.run()) {
    return 62;
  }
  auto first_output = first->read();
  auto second_output = second->read();
  constexpr std::array<std::uint32_t, 2u> expected_first{7u, 0u};
  constexpr std::array<std::uint32_t, 2u> expected_second{0u, 8u};
  if (!first_output || !second_output ||
      !std::equal(first_output->begin(), first_output->end(),
                  expected_first.begin(), expected_first.end()) ||
      !std::equal(second_output->begin(), second_output->end(),
                  expected_second.begin(), expected_second.end()) ||
      batch.stats().reset_bytes != 4u * sizeof(std::uint32_t) ||
      batch.stats().reset_commands != 2u ||
      first->stats().reset_bytes != 2u * sizeof(std::uint32_t) ||
      first->stats().reset_commands != 1u ||
      second->stats().reset_bytes != 2u * sizeof(std::uint32_t) ||
      second->stats().reset_commands != 1u) {
    return 63;
  }

  constexpr std::array<std::uint32_t, 1u> next_first_value{9u};
  constexpr std::array<std::uint32_t, 1u> next_first_index{1u};
  constexpr std::array<std::uint32_t, 1u> next_second_value{10u};
  constexpr std::array<std::uint32_t, 1u> next_second_index{0u};
  if (!first->write(next_first_value, next_first_index) ||
      !second->write(next_second_value, next_second_index) || !batch.run()) {
    return 64;
  }
  first_output = first->read();
  second_output = second->read();
  constexpr std::array<std::uint32_t, 2u> expected_next_first{0u, 9u};
  constexpr std::array<std::uint32_t, 2u> expected_next_second{10u, 0u};
  if (!first_output || !second_output ||
      !std::equal(first_output->begin(), first_output->end(),
                  expected_next_first.begin(), expected_next_first.end()) ||
      !std::equal(second_output->begin(), second_output->end(),
                  expected_next_second.begin(), expected_next_second.end()) ||
      batch.stats().reset_bytes != 4u * sizeof(std::uint32_t) ||
      batch.stats().reset_commands != 2u) {
    return 65;
  }
  return 0;
}

} // namespace

int RunComputeBatchContract() {
  if (!CheckPackedAliasPlan()) {
    return 51;
  }
  bool accelerator = false;
  for (const Backend backend :
       rund::node::test_contract::selected_compute_backends()) {
    if (backend != Backend::Cpu && backend != Backend::Metal &&
        backend != Backend::Vulkan) {
      continue;
    }
    const int admission = CheckTypedAdmission(backend);
    if (admission != 0) {
      return admission;
    }
    if (backend == Backend::Cpu) {
      continue;
    }
    accelerator = true;
    const int execution = CheckExecution(backend);
    if (execution != 0) {
      return execution;
    }
    const int overflow = CheckOverflow(backend);
    if (overflow != 0) {
      return overflow;
    }
    const int reset = CheckReset(backend);
    if (reset != 0) {
      return reset;
    }
  }
  return accelerator || rund::node::test_contract::selected_compute_backends()
                                .size() == 1u
             ? 0
             : 50;
}
