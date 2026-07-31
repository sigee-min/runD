#include <rund/compute.hpp>

#include "../target/selection.hpp"

#include "allocation.hpp"

#include <array>
#include <concepts>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <span>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

using ExactJob = rund::compute::Job<std::int32_t(std::int32_t)>;
using BoundedJob =
    rund::compute::Job<rund::compute::Bounded<std::uint32_t>(std::int32_t)>;
using OutputsJob =
    rund::compute::Job<rund::compute::Outputs<std::int32_t, std::uint32_t>(
        std::int32_t)>;

template <class Job, class Input>
concept Writes = requires(Job &job, std::span<const Input> input) {
  { job.write(input) } -> std::same_as<rund::compute::Status>;
};

template <class Job>
concept ReadsOne = requires(const Job &job) { job.read(); };

template <class Job>
concept ReadsIndexed = requires(const Job &job) { job.template read<0u>(); };

template <class Job>
concept ReadsAll = requires(const Job &job) { job.read_all(); };

static_assert(Writes<ExactJob, std::int32_t>);
static_assert(Writes<BoundedJob, std::int32_t>);
static_assert(Writes<OutputsJob, std::int32_t>);
static_assert(!Writes<ExactJob, std::uint32_t>);
static_assert(!Writes<BoundedJob, std::uint32_t>);
static_assert(!Writes<OutputsJob, std::uint32_t>);
static_assert(ReadsOne<ExactJob> && ReadsOne<BoundedJob>);
static_assert(!ReadsOne<OutputsJob>);
static_assert(!ReadsIndexed<ExactJob> && !ReadsIndexed<BoundedJob>);
static_assert(ReadsIndexed<OutputsJob>);
static_assert(!ReadsAll<ExactJob> && !ReadsAll<BoundedJob>);
static_assert(ReadsAll<OutputsJob>);

static_assert(std::same_as<decltype(std::declval<const ExactJob &>().read()),
                           rund::compute::Result<std::vector<std::int32_t>>>);
static_assert(std::same_as<decltype(std::declval<const BoundedJob &>().read()),
                           rund::compute::Result<std::vector<std::uint32_t>>>);
static_assert(std::same_as<
              decltype(std::declval<const OutputsJob &>().template read<0u>()),
              rund::compute::Result<std::vector<std::int32_t>>>);
static_assert(
    std::same_as<decltype(std::declval<const OutputsJob &>().read_all()),
                 rund::compute::Result<std::tuple<
                     std::vector<std::int32_t>, std::vector<std::uint32_t>>>>);

template <class Job>
inline constexpr bool IsResidentOwner =
    sizeof(Job) == sizeof(std::shared_ptr<void>) &&
    alignof(Job) == alignof(std::shared_ptr<void>) &&
    !std::is_default_constructible_v<Job> &&
    !std::is_copy_constructible_v<Job> && !std::is_copy_assignable_v<Job> &&
    std::is_nothrow_move_constructible_v<Job> &&
    std::is_nothrow_move_assignable_v<Job>;

static_assert(IsResidentOwner<ExactJob>);
static_assert(IsResidentOwner<BoundedJob>);
static_assert(IsResidentOwner<OutputsJob>);
static_assert(!std::constructible_from<rund::compute::Target,
                                       rund::compute::Backend, std::uint32_t>);

template <class Selector>
concept OpensCompute =
    requires(Selector selector) { rund::compute::open(selector); };

template <class Selector>
concept StartsComputeFlow =
    requires(Selector selector) { rund::compute::on(selector); };

static_assert(!OpensCompute<rund::compute::Backend>);
static_assert(!StartsComputeFlow<rund::compute::Backend>);

namespace {

int CheckAccelWarmRun(const rund::compute::Backend backend) {
  const std::array<std::int32_t, 4> input{1, 2, 3, 4};
  auto program =
      rund::compute::on(rund::node::test_contract::target_for(backend))
          .map<std::int32_t>("twice", input.size(),
                             [](auto x) { return x * 2 + 5; })
          .compile();
  if (!program) {
    std::fprintf(stderr, "accelerator job backend=%u compile reason=%.*s\n",
                 static_cast<unsigned>(backend),
                 static_cast<int>(program.error().size()),
                 program.error().data());
    return 1;
  }
  auto job = program->resident(input);
  if (!job) {
    return 2;
  }
  if (!job->run()) {
    return 3;
  }

  const rund::compute::Status warm_run = job->run();
  if (!warm_run) {
    std::fprintf(stderr, "accelerator job backend=%u warm run failed\n",
                 static_cast<unsigned>(backend));
    return 4;
  }
  const rund::compute::Stats warm = job->stats();
  if (warm.backend != backend || warm.pipeline_compiles != 0 ||
      warm.buffer_allocations != 0 || warm.uploaded_bytes != 0 ||
      warm.download_events != 0 || warm.dispatches != 1 ||
      warm.graph_hash == 0 || warm.output_hash != 0) {
    return 5;
  }
  auto output = job->read();
  if (!output || *output != std::vector<std::int32_t>{7, 9, 11, 13}) {
    return 6;
  }
  const rund::compute::Stats read_stats = job->stats();
  if (read_stats.download_events != 1u ||
      read_stats.downloaded_bytes != sizeof(input) ||
      read_stats.readback_ns <= warm.readback_ns) {
    std::fprintf(
        stderr,
        "accelerator read telemetry backend=%u events=%llu bytes=%llu "
        "readback=%llu before=%llu submit=%llu/%llu buffers=%llu/%llu\n",
        static_cast<unsigned>(backend),
        static_cast<unsigned long long>(read_stats.download_events),
        static_cast<unsigned long long>(read_stats.downloaded_bytes),
        static_cast<unsigned long long>(read_stats.readback_ns),
        static_cast<unsigned long long>(warm.readback_ns),
        static_cast<unsigned long long>(read_stats.command_submits),
        static_cast<unsigned long long>(warm.command_submits),
        static_cast<unsigned long long>(read_stats.buffer_allocations +
                                        read_stats.buffer_reuses),
        static_cast<unsigned long long>(warm.buffer_allocations +
                                        warm.buffer_reuses));
    return 22;
  }

  const std::array<std::int32_t, 4> second_input{5, 6, 7, 8};
  auto second_job = program->resident(second_input);
  if (!second_job || !second_job->run()) {
    return 7;
  }
  auto second_output = second_job->read();
  if (!second_output ||
      *second_output != std::vector<std::int32_t>{15, 17, 19, 21}) {
    return 8;
  }
  if (!job->run()) {
    return 9;
  }
  output = job->read();
  if (!output || *output != std::vector<std::int32_t>{7, 9, 11, 13}) {
    return 10;
  }

  const std::array<std::uint32_t, 4> scan_input{1, 2, 3, 4};
  auto scan_program =
      rund::compute::on(rund::node::test_contract::target_for(backend))
          .map<std::uint32_t>("adjust", scan_input.size(),
                              [](auto x) { return x + 1; })
          .scan(rund::compute::Scan::InclusiveSum)
          .compile();
  if (!scan_program) {
    return 11;
  }
  auto scan_job = scan_program->resident(scan_input);
  if (!scan_job || !scan_job->run()) {
    return 13;
  }
  const rund::compute::Status scan_warm = scan_job->run();
  if (!scan_warm) {
    std::fprintf(stderr, "scan backend=%u warm run failed\n",
                 static_cast<unsigned>(backend));
    return 12;
  }
  const rund::compute::Stats scan_warm_stats = scan_job->stats();
  if (scan_warm_stats.pipeline_compiles != 0u ||
      scan_warm_stats.buffer_allocations != 0u ||
      scan_warm_stats.uploaded_bytes != 0u ||
      scan_warm_stats.download_events != 0u ||
      scan_warm_stats.dispatches == 0u) {
    std::fprintf(
        stderr,
        "scan backend=%u warm reuse pipeline=%llu buffer=%llu upload=%llu "
        "download=%llu dispatch=%llu\n",
        static_cast<unsigned>(backend),
        static_cast<unsigned long long>(scan_warm_stats.pipeline_compiles),
        static_cast<unsigned long long>(scan_warm_stats.buffer_allocations),
        static_cast<unsigned long long>(scan_warm_stats.uploaded_bytes),
        static_cast<unsigned long long>(scan_warm_stats.download_events),
        static_cast<unsigned long long>(scan_warm_stats.dispatches));
    return 12;
  }
  auto scan_output = scan_job->read();
  if (!scan_output || *scan_output != std::vector<std::uint32_t>{2, 5, 9, 14}) {
    return 14;
  }
  if (backend != rund::compute::Backend::Vulkan) {
    return 0;
  }

  const std::array<std::uint32_t, 4> second_scan_input{5, 4, 3, 2};
  auto second_scan_job = scan_program->resident(second_scan_input);
  if (!second_scan_job || !second_scan_job->run()) {
    return 15;
  }
  auto second_scan_output = second_scan_job->read();
  if (!second_scan_output ||
      *second_scan_output != std::vector<std::uint32_t>{6, 11, 15, 18}) {
    return 16;
  }
  if (!scan_job->run()) {
    return 17;
  }
  const rund::compute::Stats first_warm = scan_job->stats();
  scan_output = scan_job->read();
  if (!scan_output || *scan_output != std::vector<std::uint32_t>{2, 5, 9, 14}) {
    return 18;
  }
  if (!second_scan_job->run()) {
    return 19;
  }
  const rund::compute::Stats second_warm = second_scan_job->stats();
  second_scan_output = second_scan_job->read();
  if (!second_scan_output ||
      *second_scan_output != std::vector<std::uint32_t>{6, 11, 15, 18}) {
    return 20;
  }
  for (const rund::compute::Stats *const stats : {&first_warm, &second_warm}) {
    if (stats->pipeline_compiles != 0 || stats->buffer_allocations != 0 ||
        stats->download_events != 0 || stats->dispatches != 2) {
      std::fprintf(stderr,
                   "vulkan scan warm pipeline_compile=%llu "
                   "buffer_allocation=%llu download_event=%llu "
                   "dispatches=%llu\n",
                   static_cast<unsigned long long>(stats->pipeline_compiles),
                   static_cast<unsigned long long>(stats->buffer_allocations),
                   static_cast<unsigned long long>(stats->download_events),
                   static_cast<unsigned long long>(stats->dispatches));
      return 21;
    }
  }
  return 0;
}

} // namespace

int RunComputeJobContract() {
  const std::array<std::int32_t, 4> input{1, 2, 3, 4};
  const std::vector<std::int32_t> expected{7, 9, 11, 13};
  auto program = rund::compute::on(rund::compute::Target::cpu())
                     .map<std::int32_t>("twice", input.size(),
                                        [](auto x) { return x * 2 + 5; })
                     .compile();
  if (!program) {
    return 2;
  }
  const std::array<std::int32_t, 3> short_input{1, 2, 3};
  auto rejected = program->resident(short_input);
  if (rejected || rejected.code() != rund::compute::Code::Binding ||
      rejected.error() != std::string_view{"compute_shape_mismatch"}) {
    return 3;
  }
  auto job = program->resident(input);
  if (!job) {
    return 4;
  }
  auto before_run = job->read();
  if (before_run || before_run.code() != rund::compute::Code::Execution ||
      before_run.error() != std::string_view{"compute_resident_not_run"}) {
    return 4;
  }
  const rund::compute::Stats before_stats = job->stats();
  if (!before_stats.available() ||
      before_stats.backend != rund::compute::Backend::Cpu ||
      before_stats.pipeline_compiles != 0 ||
      before_stats.buffer_allocations != 0 ||
      before_stats.uploaded_bytes != 0 || before_stats.download_events != 0 ||
      before_stats.dispatches != 0 || before_stats.graph_hash != 0 ||
      before_stats.output_hash != 0) {
    return 5;
  }
  if (!job->run()) {
    return 6;
  }
  node_compute_allocation::Start();
  const rund::compute::Status second = job->run();
  node_compute_allocation::Stop();
  if (!second || node_compute_allocation::Count() != 0u) {
    std::fprintf(
        stderr, "job warm heap allocations=%llu\n",
        static_cast<unsigned long long>(node_compute_allocation::Count()));
    return 7;
  }
  const rund::compute::Stats warm = job->stats();
  if (warm.backend != rund::compute::Backend::Cpu ||
      warm.pipeline_compiles != 0 || warm.buffer_allocations != 0 ||
      warm.uploaded_bytes != 0 || warm.download_events != 0 ||
      warm.dispatches != 1 || warm.graph_hash == 0 || warm.output_hash != 0) {
    std::fprintf(stderr,
                 "job warm pipeline_compile=%llu buffer_allocation=%llu "
                 "download_event=%llu dispatch=%llu graph=%llu "
                 "output=%llu\n",
                 static_cast<unsigned long long>(warm.pipeline_compiles),
                 static_cast<unsigned long long>(warm.buffer_allocations),
                 static_cast<unsigned long long>(warm.download_events),
                 static_cast<unsigned long long>(warm.dispatches),
                 static_cast<unsigned long long>(warm.graph_hash),
                 static_cast<unsigned long long>(warm.output_hash));
    return 8;
  }
  auto output = job->read();
  if (!output || *output != expected) {
    return 9;
  }
  const rund::compute::Stats read = job->stats();
  if (read.download_events != 1 || read.output_hash == 0) {
    return 10;
  }

  ExactJob owned = std::move(*job);
  const rund::compute::Stats unavailable_stats = job->stats();
  const rund::compute::MemoryStats unavailable_memory = job->memory();
  std::array<rund::compute::MemoryEntry, 1u> unavailable_entries{};
  const rund::compute::MemorySnapshot unavailable_snapshot =
      job->memory_snapshot(unavailable_entries);
  node_compute_allocation::Start();
  const rund::compute::Stats owned_stats = owned.stats();
  const rund::compute::MemoryStats owned_memory = owned.memory();
  node_compute_allocation::Stop();
  if (unavailable_stats.available() || unavailable_memory.available() ||
      unavailable_stats.backend != rund::compute::Backend::Unavailable ||
      unavailable_memory.backend != rund::compute::Backend::Unavailable ||
      unavailable_snapshot.summary.available() ||
      unavailable_snapshot.written != 0u || unavailable_snapshot.total != 0u ||
      !owned_stats.available() || !owned_memory.available() ||
      node_compute_allocation::Count() != 0u) {
    return 11;
  }

  for (const rund::compute::Backend backend :
       rund::node::test_contract::selected_accelerators()) {
    if (const int result = CheckAccelWarmRun(backend); result != 0) {
      return 20 + static_cast<int>(backend) * 10 + result;
    }
  }
  return 0;
}
