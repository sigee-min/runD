#include "../../support.hpp"

#include "src/compute/host.hpp"

#include <kernel/dispatch/worker/backend.hpp>
#include <node/runtime/compute/access.hpp>
#include <rund/compute.hpp>

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <string_view>
#include <vector>

namespace rund::node::test_contract {
namespace {

constexpr kernel::u32 kWorkers = 2u;

kernel::u32 WorkerCount(void *) noexcept { return kWorkers; }

kernel::WorkerAffinityPolicy Affinity(void *) noexcept {
  return kernel::WorkerAffinityPolicy::Static;
}

kernel::WorkerBackendCapabilities Capabilities(void *) noexcept {
  return kernel::WorkerBackendCapabilities{
      .backend_width = kWorkers,
      .width_matches_request = true,
      .supports_static_partitions = true,
      .supports_async_partitions = true,
      .supports_static_tile_map = true,
      .supports_claim_free_static_tiles = true,
      .supports_no_alloc_worker_stats = true,
      .supports_strict_fp_fold = true,
      .is_nested = false,
      .affinity_is_truth = true,
      .affinity_truth_level = kernel::WorkerTruthLevel::Verified,
      .affinity_policy = kernel::WorkerAffinityPolicy::Static,
  };
}

bool NotNested(void *) noexcept { return false; }

bool Submit(void *, const kernel::Partition *const partitions,
            const kernel::u32 count, const kernel::WorkerTask task,
            kernel::WorkerStats *,
            kernel::WorkerSubmission *const submission) noexcept {
  if (partitions == nullptr || !task || submission == nullptr ||
      submission->completion.invoke == nullptr) {
    return false;
  }
  for (kernel::u32 index = 0u; index < count; ++index) {
    task.invoke(task.context, partitions[index]);
  }
  submission->completion.invoke(submission->completion.context, true);
  return true;
}

kernel::WorkerBackend InlineBackend() noexcept {
  static std::uint8_t identity{};
  return kernel::WorkerBackend{
      .context = &identity,
      .worker_count = WorkerCount,
      .affinity_policy = Affinity,
      .capabilities = Capabilities,
      .is_nested = NotNested,
      .submit_partitions = Submit,
  };
}

struct Ready final {
  std::atomic_bool *cancel{};
  std::uint32_t cancel_after{};
  std::uint32_t count{};
};

void Signal(void *const raw) noexcept {
  auto *const ready = static_cast<Ready *>(raw);
  if (ready == nullptr) {
    return;
  }
  ++ready->count;
  if (ready->cancel != nullptr && ready->count == ready->cancel_after) {
    ready->cancel->store(true, std::memory_order_release);
  }
}

template <class Program>
bool CancelAt(Program &program, const std::span<const std::int32_t> input,
              const std::uint32_t cancel_after, const bool cancel_before) {
  auto job = program.resident(input);
  if (!job) {
    return false;
  }
  const auto state = compute::detail::JobAccess::state(*job);
  if (!compute::detail::queue_job(state)) {
    return false;
  }
  std::atomic_bool cancel{cancel_before};
  Ready ready{.cancel = &cancel, .cancel_after = cancel_after};
  const compute::Status submitted = compute::detail::submit_cpu_job_on(
      state, InlineBackend(), kWorkers, &cancel, &ready, Signal);
  if (!submitted) {
    std::fprintf(stderr, "epoch submit failed: %.*s\n",
                 static_cast<int>(submitted.error().size()),
                 submitted.error().data());
    return false;
  }
  for (std::uint32_t epoch = 0u; epoch < 8u; ++epoch) {
    const compute::detail::CpuJobProgress progress =
        compute::detail::advance_cpu_job_on(state, InlineBackend(), &cancel,
                                            &ready, Signal);
    switch (progress.disposition()) {
    case compute::detail::CpuJobProgressDisposition::Failed:
      if (progress.status().error() != std::string_view{"compute_cancelled"}) {
        std::fprintf(stderr, "epoch progress failed: %.*s ready=%u\n",
                     static_cast<int>(progress.status().error().size()),
                     progress.status().error().data(), ready.count);
      }
      return progress.status().error() == std::string_view{"compute_cancelled"};
    case compute::detail::CpuJobProgressDisposition::Complete:
      return false;
    case compute::detail::CpuJobProgressDisposition::Pending:
      break;
    }
  }
  return false;
}

} // namespace

int CheckComputeCancelEpochs() {
  constexpr std::size_t count = 4097u;
  const std::vector<std::int32_t> input(count, 3);
  auto tile_program =
      compute::on(compute::Target::cpu(kWorkers))
          .map<std::int32_t>("cancel-tile-epoch", count,
                             [](auto value) { return value * 3 + 1; })
          .compile();
  if (!tile_program) {
    return 11;
  }

  auto step_program =
      compute::on(compute::Target::cpu(kWorkers))
          .map<std::int32_t>("cancel-graph-step", count,
                             [](auto value) { return value * 3 + 1; })
          .reduce(compute::Reduce::Sum)
          .compile();
  if (!step_program) {
    return 12;
  }

  auto scan_program = compute::on(compute::Target::cpu(kWorkers))
                          .map<std::int32_t>("cancel-collective-pass", count,
                                             [](auto value) { return value; })
                          .scan(compute::Scan::InclusiveSum)
                          .compile();
  if (!scan_program) {
    return 13;
  }

  for (std::uint32_t repeat = 0u; repeat < 32u; ++repeat) {
    if (!CancelAt(*tile_program, input, 0u, true)) {
      return 14;
    }
    if (!CancelAt(*tile_program, input, 1u, false)) {
      return 15;
    }
    if (!CancelAt(*step_program, input, 2u, false)) {
      return 16;
    }
    if (!CancelAt(*scan_program, input, 2u, false)) {
      return 17;
    }
  }
  return 0;
}

} // namespace rund::node::test_contract
