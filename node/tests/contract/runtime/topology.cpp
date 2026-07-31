#include "test/assert.hpp"

#include <kernel/dispatch/kernel.hpp>
#include <node/runtime/backend.hpp>
#include <rund/session.hpp>

#include "../../../src/runtime/resource/discovery.hpp"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace {

struct BackendEvidence final {
  std::uint32_t width = 2u;
  rund::kernel::WorkerTruthLevel affinity =
      rund::kernel::WorkerTruthLevel::Verified;
  rund::kernel::WorkerTruthLevel capacity =
      rund::kernel::WorkerTruthLevel::Verified;
  std::vector<std::uint32_t> capacity_milli{1000u, 1000u};
};

rund::kernel::u32 WorkerCount(void *const raw) noexcept {
  const auto *const evidence = static_cast<const BackendEvidence *>(raw);
  return evidence == nullptr ? 0u : evidence->width;
}

rund::kernel::WorkerAffinityPolicy Affinity(void *) noexcept {
  return rund::kernel::WorkerAffinityPolicy::Static;
}

rund::kernel::WorkerBackendCapabilities Capabilities(void *const raw) noexcept {
  const auto *const evidence = static_cast<const BackendEvidence *>(raw);
  if (evidence == nullptr) {
    return {};
  }
  return rund::kernel::WorkerBackendCapabilities{
      .backend_width = evidence->width,
      .width_matches_request = true,
      .supports_static_partitions = true,
      .supports_static_tile_map = true,
      .supports_claim_free_static_tiles = true,
      .supports_no_alloc_worker_stats = true,
      .supports_strict_fp_fold = true,
      .affinity_is_truth =
          evidence->affinity == rund::kernel::WorkerTruthLevel::Verified,
      .affinity_truth_level = evidence->affinity,
      .worker_capacity_milli = evidence->capacity_milli.data(),
      .worker_capacity_count =
          static_cast<rund::kernel::u32>(evidence->capacity_milli.size()),
      .worker_capacity_truth_level = evidence->capacity,
      .affinity_policy = rund::kernel::WorkerAffinityPolicy::Static,
  };
}

bool Execute(void *, const rund::kernel::Partition *const partitions,
             const rund::kernel::u32 partition_count,
             const rund::kernel::WorkerTask task,
             rund::kernel::WorkerStats *) noexcept {
  if (partitions == nullptr || !task) {
    return false;
  }
  for (rund::kernel::u32 index = 0u; index < partition_count; ++index) {
    task.invoke(task.context, partitions[index]);
  }
  return true;
}

rund::kernel::WorkerBackend Backend(BackendEvidence &evidence) noexcept {
  return rund::kernel::WorkerBackend{
      .context = &evidence,
      .worker_count = WorkerCount,
      .affinity_policy = Affinity,
      .capabilities = Capabilities,
      .execute_partitions = Execute,
  };
}

[[nodiscard]] rund::node::runtime_detail::resource::Result
Admit(BackendEvidence &evidence,
      const rund::node::runtime_detail::resource::Request &request) {
  return rund::node::runtime_detail::resource::Resolve(
      rund::node::select_backend(Backend(evidence), evidence.width), request);
}

} // namespace

int RunRuntimeTopologyContract() {
  rund::Session built_in{};
  rund::SessionConfig built_in_options{};
  built_in_options.workers = 3u;
  built_in_options.scheduler.task_workers = 3u;
  built_in_options.require_verified_numa = true;
  built_in_options.require_verified_affinity = true;
  built_in_options.require_verified_worker_capacity = true;
  TEST_ASSERT(built_in.open(built_in_options));
  const rund::Resources built_in_resources = built_in.resources();
  TEST_ASSERT(built_in_resources);
  TEST_ASSERT(built_in_resources.workers == 3u);
  TEST_ASSERT(built_in_resources.topology.numa ==
              rund::EvidenceTruth::Verified);
  TEST_ASSERT(built_in_resources.topology.affinity ==
              rund::EvidenceTruth::Verified);
  TEST_ASSERT(built_in_resources.topology.worker_capacity ==
              rund::EvidenceTruth::Verified);
  TEST_ASSERT(built_in_resources.worker_capacity_milli.size() == 3u);
  TEST_ASSERT(
      std::all_of(built_in_resources.worker_capacity_milli.begin(),
                  built_in_resources.worker_capacity_milli.end(),
                  [](const std::uint32_t value) { return value > 0u; }));
  TEST_ASSERT(built_in.close());

  BackendEvidence verified{};
  BackendEvidence hinted_affinity{};
  hinted_affinity.affinity = rund::kernel::WorkerTruthLevel::HintOnly;
  BackendEvidence short_capacity{};
  short_capacity.capacity_milli.pop_back();
  BackendEvidence zero_capacity{};
  zero_capacity.capacity_milli.back() = 0u;
  BackendEvidence unknown_capacity{};
  unknown_capacity.capacity = rund::kernel::WorkerTruthLevel::Unknown;

  TEST_ASSERT(Admit(verified,
                    {.workers = verified.width, .require_verified_numa = true})
                  .code == rund::ReasonCode::VerifiedTopologyRequired);
  TEST_ASSERT(Admit(hinted_affinity, {.workers = hinted_affinity.width,
                                      .require_verified_affinity = true})
                  .code == rund::ReasonCode::AffinityTruthUnavailable);
  TEST_ASSERT(Admit(short_capacity, {.workers = short_capacity.width,
                                     .require_verified_capacity = true})
                  .code == rund::ReasonCode::WorkerCapacityTruthUnavailable);
  TEST_ASSERT(Admit(zero_capacity, {.workers = zero_capacity.width,
                                    .require_verified_capacity = true})
                  .code == rund::ReasonCode::WorkerCapacityTruthUnavailable);
  TEST_ASSERT(Admit(unknown_capacity, {.workers = unknown_capacity.width,
                                       .require_verified_capacity = true})
                  .code == rund::ReasonCode::WorkerCapacityTruthUnavailable);
  TEST_ASSERT(
      rund::node::select_backend(rund::kernel::WorkerBackend{}, 2u).code ==
      rund::ReasonCode::BackendInvalid);

  rund::Session runtime{};
  rund::SessionConfig options{};
  options.id = 2u;
  options.workers = 2u;
  options.scheduler.task_workers = 4u;
  options.require_verified_numa = true;
  options.require_verified_affinity = true;
  options.require_verified_worker_capacity = true;
  TEST_ASSERT(runtime.open(options));
  const rund::Resources runtime_resources = runtime.resources();
  TEST_ASSERT(runtime_resources);
  TEST_ASSERT(runtime_resources.workers == 2u);
  TEST_ASSERT(runtime_resources.worker_capacity_milli.size() == 2u);
  TEST_ASSERT(runtime.close());
  return 0;
}
