#include "../../../compute/allocation.hpp"
#include "../support.hpp"
#include "host/check.hpp"

#include <rund/compute.hpp>
#include <rund/compute/session.hpp>
#include <rund/replay.hpp>
#include <rund/task/api.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string_view>
#include <vector>

namespace {

struct ReadyOrder final {
  std::uint64_t allocations = 0u;
  bool ok = false;
};

struct CommitToken final {
  std::atomic<std::uint32_t> *completed = nullptr;
  std::atomic<std::uint32_t> *commit_index = nullptr;
  std::array<std::uint32_t, 4u> *commit_order = nullptr;
  std::uint32_t value = 0u;
  bool armed = true;

  CommitToken(std::atomic<std::uint32_t> &completed_value,
              std::atomic<std::uint32_t> &commit_index_value,
              std::array<std::uint32_t, 4u> &commit_order_value,
              const std::uint32_t task_value) noexcept
      : completed(&completed_value), commit_index(&commit_index_value),
        commit_order(&commit_order_value), value(task_value) {}
  CommitToken(const CommitToken &) = delete;
  CommitToken &operator=(const CommitToken &) = delete;
  CommitToken(CommitToken &&other) noexcept
      : completed(other.completed), commit_index(other.commit_index),
        commit_order(other.commit_order), value(other.value),
        armed(other.armed) {
    other.armed = false;
  }
  CommitToken &operator=(CommitToken &&) = delete;
  ~CommitToken() {
    if (!armed) {
      return;
    }
    const std::uint32_t index =
        commit_index->fetch_add(1u, std::memory_order_relaxed);
    if (index < commit_order->size()) {
      (*commit_order)[index] = value;
    }
  }

  void operator()() const noexcept {
    completed->fetch_add(1u, std::memory_order_relaxed);
  }
};

[[nodiscard]] bool WarmReadyOrder() {
  std::array<rund::task::Handle, 4u> handles{};
  for (rund::task::Handle &handle : handles) {
    handle = rund::task::spawn("ready-order-warm", [] {});
  }
  return static_cast<bool>(rund::task::join_all(handles));
}

[[nodiscard]] ReadyOrder CheckReadyOrder(const std::uint32_t workers) {
  constexpr std::array<std::int32_t, 1u> input{7};
  auto program = rund::compute::on(rund::compute::Target::cpu(2u))
                     .map<std::int32_t>("ready-order", input.size(),
                                        [](auto value) { return value + 1; })
                     .compile();
  if (!program) {
    return {};
  }
  auto job = program->resident(input);
  if (!job) {
    return {};
  }

  rund::SessionConfig options = rund::node::test_contract::Options();
  options.id = 93u;
  options.scheduler.task_workers = workers;
  options.scheduler.task_capacity = 8u;
  options.scheduler.ready_queue_capacity = 8u;
  rund::Session session{};
  if (!session.open(options)) {
    return {};
  }
  if (!session.compute(*job).submit().wait()) {
    static_cast<void>(session.close());
    return {};
  }

  std::atomic<std::uint32_t> completed{0u};
  std::atomic<std::uint32_t> commit_index{0u};
  std::array<std::uint32_t, 4u> commit_order{};
  rund::task::Status joined{};
  bool computed = false;
  std::uint64_t allocations = 0u;
  const rund::Session::Result report = session.scope([&] {
    if (!WarmReadyOrder()) {
      return;
    }
    std::array<rund::task::Handle, 4u> handles{};
    for (std::size_t index = 0u; index < handles.size(); ++index) {
      handles[index] = rund::task::spawn(
          "ready-order", CommitToken{completed, commit_index, commit_order,
                                     static_cast<std::uint32_t>(index + 1u)});
    }
    node_compute_allocation::Start();
    const rund::compute::Completion completion =
        session.compute(*job).submit().wait();
    computed = static_cast<bool>(completion);
    node_compute_allocation::Stop();
    allocations = node_compute_allocation::Count();
    if (!computed) {
      std::fprintf(stderr, "ready order compute: %.*s\n",
                   static_cast<int>(completion.error().size()),
                   completion.error().data());
    }
    joined = rund::task::join_all(handles);
  });
  const bool closed = session.close().ok();
  constexpr std::array<std::uint32_t, 4u> expected{1u, 2u, 3u, 4u};
  const std::uint32_t completed_count =
      completed.load(std::memory_order_relaxed);
  const std::uint32_t committed_count =
      commit_index.load(std::memory_order_relaxed);
  const bool ok = report && computed && joined && closed && allocations == 0u &&
                  completed_count == 4u && committed_count == 4u &&
                  commit_order == expected;
  if (!ok) {
    std::fprintf(
        stderr,
        "ready order workers=%u report=%u computed=%u joined=%u closed=%u "
        "allocations=%llu completed=%u committed=%u order=%u,%u,%u,%u\n",
        workers, report ? 1u : 0u, computed ? 1u : 0u, joined ? 1u : 0u,
        closed ? 1u : 0u, static_cast<unsigned long long>(allocations),
        completed_count, committed_count, commit_order[0], commit_order[1],
        commit_order[2], commit_order[3]);
  }
  return ReadyOrder{
      .allocations = allocations,
      .ok = ok,
  };
}

[[nodiscard]] bool CheckCpuStepParity(rund::Session &runtime) {
  using namespace rund::compute;
  constexpr std::size_t count = 4u * 1024u + 17u;
  const std::vector<std::int32_t> input(count, 0);
  auto program = on(Target::cpu(2u))
                     .map<std::int32_t>("cpu-step-parity", count,
                                        [](auto value) { return value + 1; })
                     .filter([](auto value) { return value < 0; })
                     .scan(Scan::InclusiveSum)
                     .compile();
  if (!program) {
    return false;
  }
  auto blocking = program->resident(input);
  auto submitted = program->resident(input);
  if (!blocking || !submitted || !blocking->run()) {
    return false;
  }
  const Stats blocking_stats = blocking->stats();
  auto submission = runtime.compute(*submitted).submit();
  const rund::compute::Poll progress =
      submission.wait_for(std::chrono::nanoseconds::zero());
  if (!progress.submitted ||
      (progress.completed && progress.reason() != Reason::Ok)) {
    return false;
  }
  const rund::compute::Poll settled =
      progress.completed ? progress
                         : submission.wait_for(std::chrono::seconds{5});
  if (!settled.completed || settled.reason() != Reason::Ok) {
    return false;
  }
  const rund::compute::Completion completion = submission.wait();
  if (!completion) {
    return false;
  }
  const Stats submitted_stats = completion.stats();
  auto blocking_output = blocking->read();
  auto submitted_output = submitted->read();
  return blocking_output && submitted_output && blocking_output->empty() &&
         submitted_output->empty() &&
         blocking_stats.graph_hash == submitted_stats.graph_hash &&
         blocking_stats.graph_read_bytes == submitted_stats.graph_read_bytes &&
         blocking_stats.dispatches == submitted_stats.dispatches &&
         blocking_stats.worker_count == submitted_stats.worker_count &&
         blocking_stats.tile_count == submitted_stats.tile_count &&
         blocking_stats.tile_size == submitted_stats.tile_size &&
         blocking_stats.vector_chunks == submitted_stats.vector_chunks &&
         blocking_stats.tail_chunks == submitted_stats.tail_chunks;
}

[[nodiscard]] bool CheckServerReplay(rund::Session &server) {
  using namespace rund::compute;
  constexpr std::size_t count = 4u;
  auto device = open(server, Target::cpu(2u));
  if (!device) {
    std::fprintf(stderr, "server replay device: %.*s\n",
                 static_cast<int>(device.error().size()),
                 device.error().data());
    return false;
  }
  const auto backend = device->backend();
  if (!backend || *backend != Backend::Cpu) {
    std::fprintf(stderr, "server replay backend invalid\n");
    return false;
  }
  auto program =
      on(*device)
          .map<std::uint32_t>("server-native-map", count,
                              [](auto value) { return value * 3u + 1u; })
          .compile();
  if (!program) {
    std::fprintf(stderr, "server replay program: %.*s\n",
                 static_cast<int>(program.error().size()),
                 program.error().data());
    return false;
  }
  constexpr std::array<std::uint32_t, count> empty{};
  auto job = program->resident(empty);
  if (!job) {
    std::fprintf(stderr, "server replay job: %.*s\n",
                 static_cast<int>(job.error().size()), job.error().data());
    return false;
  }

  std::array canonical{std::byte{1u}, std::byte{2u}, std::byte{3u},
                       std::byte{4u}};
  std::uint64_t source_calls = 0u;
  std::uint64_t callback_calls = 0u;
  bool callback_ok = true;
  std::vector<std::uint32_t> recorded{};
  std::vector<std::uint32_t> replayed{};
  rund::replay::Binding replay{};
  auto source = [&](rund::replay::Writer &writer) -> std::uint64_t {
    ++source_calls;
    if (!writer.append(canonical)) {
      callback_ok = false;
    }
    return 19u;
  };
  const auto input =
      replay.input(rund::replay::Input{.id = 71u, .schema = 7001u}, source);
  const auto run = [&](rund::replay::Context &context, rund::Session &active) {
    ++callback_calls;
    const rund::replay::Value value = input.read(context);
    if (&active != &server || !value || value.sequence() != 19u ||
        value.size() != count) {
      callback_ok = false;
      return;
    }
    std::array<std::uint32_t, count> values{};
    for (std::size_t index = 0u; index < count; ++index) {
      values[index] = std::to_integer<std::uint32_t>(value.bytes()[index]);
    }
    if (!job->write(values) || !active.compute(*job).submit().wait()) {
      callback_ok = false;
      return;
    }
    auto output = job->read();
    if (!output) {
      callback_ok = false;
      return;
    }
    (callback_calls == 1u ? recorded : replayed) = std::move(*output);
  };

  const rund::replay::Record baseline = rund::replay::record(server, run);
  if (!baseline || !callback_ok || source_calls != 1u || callback_calls != 1u) {
    std::fprintf(stderr,
                 "server replay record: %.*s callback_ok=%u source=%llu "
                 "callback=%llu\n",
                 static_cast<int>(baseline.error().size()),
                 baseline.error().data(), callback_ok ? 1u : 0u,
                 static_cast<unsigned long long>(source_calls),
                 static_cast<unsigned long long>(callback_calls));
    return false;
  }
  canonical.fill(std::byte{0xffu});
  const rund::replay::Check checked = rund::replay::run(server, baseline, run);
  constexpr std::array<std::uint32_t, count> expected{4u, 7u, 10u, 13u};
  const bool ok = checked && checked.actual_hash() == baseline.hash() &&
                  callback_ok && source_calls == 1u && callback_calls == 2u &&
                  std::ranges::equal(recorded, expected) &&
                  std::ranges::equal(replayed, expected);
  if (!ok) {
    std::fprintf(stderr,
                 "server native replay: %.*s record=%llu actual=%llu "
                 "source_calls=%llu callback_calls=%llu\n",
                 static_cast<int>(checked.error().size()),
                 checked.error().data(),
                 static_cast<unsigned long long>(baseline.hash()),
                 static_cast<unsigned long long>(checked.actual_hash()),
                 static_cast<unsigned long long>(source_calls),
                 static_cast<unsigned long long>(callback_calls));
  }
  return ok;
}

} // namespace

int RunRuntimeComputeHostContract() {
  const ReadyOrder two_workers = CheckReadyOrder(2u);
  const ReadyOrder four_workers = CheckReadyOrder(4u);
  TEST_ASSERT(two_workers.ok);
  TEST_ASSERT(four_workers.ok);
  TEST_ASSERT(two_workers.allocations == 0u);
  TEST_ASSERT(four_workers.allocations == 0u);

  constexpr std::size_t kCount = 64u * 1024u + 17u;
  std::vector<std::int32_t> input(kCount);
  for (std::size_t index = 0u; index < input.size(); ++index) {
    input[index] = static_cast<std::int32_t>(index % 97u);
  }

  auto program =
      rund::compute::on(rund::compute::Target::cpu(2u))
          .map<std::int32_t>("node-host-twice", input.size(),
                             [](auto value) { return value * 2 + 5; })
          .compile();
  TEST_ASSERT(program);
  auto job = program->resident(input);
  TEST_ASSERT(job);

  rund::node::test_contract::TelemetryProbe telemetry{};
  rund::Session runtime{};
  TEST_ASSERT(
      runtime.open(rund::node::test_contract::Options(&telemetry)).ok());

  auto task = runtime.compute(*job).submit();
  const rund::compute::Poll submitted = task.poll();
  if (!submitted.submitted) {
    std::fprintf(stderr, "node cpu compute admission failed: %.*s\n",
                 static_cast<int>(submitted.error().size()),
                 submitted.error().data());
  }
  TEST_ASSERT(submitted.submitted);
  TEST_ASSERT(submitted.reason() == rund::compute::Reason::Ok);
  TEST_ASSERT(submitted.code() == rund::compute::Code::Ok);
  TEST_ASSERT(submitted.error().empty());
  const rund::compute::Completion result = task.wait();
  if (!result) {
    std::fprintf(stderr, "node cpu compute failed: %.*s\n",
                 static_cast<int>(result.error().size()),
                 result.error().data());
  }
  TEST_ASSERT(result);
  const rund::compute::Poll completed = task.poll();
  TEST_ASSERT(completed.submitted);
  TEST_ASSERT(completed.completed);
  TEST_ASSERT(completed.reason() == rund::compute::Reason::Ok);
  TEST_ASSERT(completed.code() == rund::compute::Code::Ok);
  TEST_ASSERT(completed.error().empty());
  TEST_ASSERT(result.reason() == rund::compute::Reason::Ok);
  TEST_ASSERT(result.error().empty());
  TEST_ASSERT(result.stats().backend == rund::compute::Backend::Cpu);
  TEST_ASSERT(result.stats().worker_count == 2u);
  TEST_ASSERT(result.stats().participating_workers >= 2u);
  TEST_ASSERT(result.stats().pipeline_compiles == 0u);
  TEST_ASSERT(result.stats().buffer_allocations == 0u);
  TEST_ASSERT(result.stats().download_events == 0u);
  TEST_ASSERT(telemetry.events == 1u);
  TEST_ASSERT(telemetry.event.source == rund::telemetry::Source::Compute);
  TEST_ASSERT(telemetry.event.level == rund::telemetry::Level::Basic);
  TEST_ASSERT(telemetry.event.session == 1u);
  TEST_ASSERT(telemetry.event.compute.backend == rund::compute::Backend::Cpu);
  TEST_ASSERT(telemetry.event.compute.code == rund::compute::Code::Ok);
  TEST_ASSERT(telemetry.event.compute.graph == result.stats().graph_hash);
  TEST_ASSERT(telemetry.event.compute.workers == 2u);
  TEST_ASSERT(telemetry.event.compute.active_workers >= 2u);
  TEST_ASSERT(telemetry.event.compute.tiles == result.stats().tile_count);
  TEST_ASSERT(telemetry.event.compute.dispatches == 1u);
  TEST_ASSERT(telemetry.event.compute.buffer_allocations == 0u);
  TEST_ASSERT(telemetry.event.detail.prepare_ns == 0u);
  TEST_ASSERT(telemetry.event.detail.work_ns == 0u);
  TEST_ASSERT(telemetry.event.detail.finish_ns == 0u);
  TEST_ASSERT(telemetry.event.replay.mode == rund::telemetry::Mode::None);
  TEST_ASSERT(telemetry.event.replay.plan ==
              rund::telemetry::Preparation::None);
  TEST_ASSERT(telemetry.event.replay.input_rows == 0u);
  TEST_ASSERT(telemetry.event.replay.input_bytes == 0u);
  TEST_ASSERT(telemetry.event.replay.choices == 0u);
  TEST_ASSERT(telemetry.event.replay.evidence_rows == 0u);
  TEST_ASSERT(telemetry.event.replay.evidence_bytes == 0u);
  TEST_ASSERT(telemetry.event.replay.retained_bytes == 0u);
  TEST_ASSERT(telemetry.event.replay.copied_bytes == 0u);
  TEST_ASSERT(telemetry.event.replay.physical_bytes == 0u);
  TEST_ASSERT(telemetry.event.replay.allocated_bytes == 0u);
  TEST_ASSERT(telemetry.event.replay.reserved_bytes == 0u);
  TEST_ASSERT(telemetry.event.replay.storage_growths == 0u);
  TEST_ASSERT(telemetry.event.replay.result_hash == 0u);
  TEST_ASSERT(telemetry.event.error().empty());
  TEST_ASSERT(telemetry.event.replay.code == rund::replay::Code::Ok);

  node_compute_allocation::Start();
  const rund::compute::Completion warm = runtime.compute(*job).submit().wait();
  node_compute_allocation::Stop();
  if (!warm || node_compute_allocation::Count() != 0u) {
    std::fprintf(
        stderr, "node cpu warm submit allocations=%llu reason=%.*s\n",
        static_cast<unsigned long long>(node_compute_allocation::Count()),
        static_cast<int>(warm.error().size()), warm.error().data());
  }
  TEST_ASSERT(warm);
  TEST_ASSERT(node_compute_allocation::Count() == 0u);

  TEST_ASSERT(compute_host_test::OutputMatches(*job, input));

  const std::vector<std::int32_t> empty;
  auto empty_program =
      rund::compute::on(rund::compute::Target::cpu(2u))
          .map<std::int32_t>("node-host-empty", 0u,
                             [](auto value) { return value + 1; })
          .compile();
  TEST_ASSERT(empty_program);
  auto empty_job = empty_program->resident(empty);
  TEST_ASSERT(empty_job);
  auto empty_task = runtime.compute(*empty_job).submit();
  const rund::compute::Completion empty_result = empty_task.wait();
  TEST_ASSERT(empty_result);
  const rund::compute::Poll empty_poll = empty_task.poll();
  TEST_ASSERT(empty_poll.submitted);
  TEST_ASSERT(empty_poll.completed);
  TEST_ASSERT(empty_result.stats().backend == rund::compute::Backend::Cpu);
  TEST_ASSERT(empty_result.stats().graph_hash != 0u);
  TEST_ASSERT(empty_result.stats().pipeline_compiles == 0u);
  TEST_ASSERT(empty_result.stats().buffer_allocations == 0u);
  TEST_ASSERT(empty_result.stats().download_events == 0u);
  const auto empty_output = empty_job->read();
  TEST_ASSERT(empty_output);
  TEST_ASSERT(empty_output->empty());
  TEST_ASSERT(empty_job->stats().output_hash != 0u);

  constexpr std::array<std::uint32_t, 2u> overflow{
      std::numeric_limits<std::uint32_t>::max(), 1u};
  constexpr std::array<std::uint32_t, 2u> recovered_input{1u, 2u};
  auto failure_program =
      rund::compute::on(rund::compute::Target::cpu(2u))
          .map<std::uint32_t>("node-host-overflow", overflow.size(),
                              [](auto value) { return value; })
          .scan(rund::compute::Scan::InclusiveSum)
          .compile();
  TEST_ASSERT(failure_program);
  auto failure_job = failure_program->resident(overflow);
  TEST_ASSERT(failure_job);
  const rund::compute::Completion overflow_result =
      runtime.compute(*failure_job).submit().wait();
  TEST_ASSERT(!overflow_result);
  TEST_ASSERT(overflow_result.reason() ==
              rund::compute::Reason::ScanSumOverflow);
  TEST_ASSERT(overflow_result.error() ==
              std::string_view{"compute_scan_sum_overflow"});
  TEST_ASSERT(failure_job->write(recovered_input));
  TEST_ASSERT(runtime.compute(*failure_job).submit().wait());
  const auto recovered = failure_job->read();
  TEST_ASSERT(recovered);
  TEST_ASSERT(*recovered == std::vector<std::uint32_t>({1u, 3u}));
  TEST_ASSERT(CheckCpuStepParity(runtime));
  TEST_ASSERT(CheckServerReplay(runtime));

  const rund::compute::Status completed_cancel = task.cancel();
  TEST_ASSERT(!completed_cancel);
  TEST_ASSERT(completed_cancel.error() ==
              std::string_view{"compute_already_completed"});

  TEST_ASSERT(compute_host_test::HasComputeTrace(runtime));

  const rund::compute::Poll invalid = rund::compute::Submission{}.poll();
  TEST_ASSERT(!invalid.submitted);
  TEST_ASSERT(!invalid.backend_submitted);
  TEST_ASSERT(invalid.completed);
  TEST_ASSERT(invalid.reason() == rund::compute::Reason::TaskInvalid);
  TEST_ASSERT(invalid.code() == rund::compute::Code::Invalid);
  TEST_ASSERT(invalid.error() == std::string_view{"compute_task_invalid"});

  TEST_ASSERT(runtime.close().ok());
  return 0;
}
