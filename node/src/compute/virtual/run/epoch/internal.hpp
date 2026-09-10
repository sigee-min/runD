#pragma once

#include "../../stats.hpp"
#include "../cache.hpp"
#include "../epoch.hpp"
#include "../transaction.hpp"

#include "../../../device/residency/pool.hpp"
#include "../../../device/residency/registry/transaction_owner.hpp"
#include "../../../pipeline/local.hpp"
#include "../../../pipeline/run/clock.hpp"
#include "../../../pipeline/transfer/batch.hpp"

#include <rund/compute/pipeline/runtime.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace rund::compute::detail {

// This is a per-epoch scratch record only.  It borrows the prepared Pipeline,
// Pool, backing, and Authority-owned leases; it retains no frame table,
// policy, journal, or publication state after the epoch returns.
struct VirtualEpochContext final {
  VirtualPipelineState &state;
  VirtualBacking &input;
  VirtualBacking &output;
  const VirtualRunProjection &run;
  std::uint64_t epoch;
  std::array<bool, 2u> &prefetch_pending;
  Stats &stats;
  ::rund::node::hash_detail::Fnv &output_hash;
  VirtualReduction *reduction;
  VirtualScan *scan;
  VirtualRunTransaction *transaction;

  VirtualEpochProjection projected{};
  std::shared_ptr<PipelineState> *selected{};
  PipelineState *pipeline{};
  residency::Pool *pool{};
  std::uint32_t bank{};
  std::size_t consume_lane{};
  std::size_t epoch_pages{};
  bool cleanup_failed{};
  residency::PrefetchReceipt prefetched{};
  std::uint64_t host_token{};
  std::array<residency::CacheUse, PipelineLeafCapacity> input_uses{};
  std::array<residency::CacheUse, PipelineLeafCapacity> output_uses{};
  residency::AuthorityResult acquired{};
  residency::EpochLease input_lease{};
  residency::EpochLease device_output_lease{};
  VirtualOutputReservation output_reservation{};
  residency::EpochLease output_lease{};
  std::uint64_t execution_token{};
  std::array<PipelineFrameDownload, PipelineLeafCapacity> download_ranges{};
  std::array<residency::CacheKey, PipelineLeafCapacity> dirty{};
  std::size_t dirty_count{};
};

struct VirtualEpochPhaseResult final {
  Status status{Status::success()};
  std::uint64_t failed_page{ResidencyStats::no_failed_page};
  bool poison_pipeline{};
  bool invalidate_all{};
};

[[nodiscard]] VirtualEpochResult
admit_virtual_epoch(VirtualEpochContext &) noexcept;

[[nodiscard]] bool cancel_virtual_prefetch(residency::Pool *,
                                           std::array<bool, 2u> &) noexcept;

[[nodiscard]] VirtualEpochResult
abort_virtual_epoch(VirtualEpochContext &, Status, bool poison = false,
                    bool invalidate_all = false) noexcept;

[[nodiscard]] VirtualEpochPhaseResult
supply_virtual_epoch(VirtualEpochContext &) noexcept;

[[nodiscard]] Status
execute_virtual_epoch_pipeline(VirtualEpochContext &) noexcept;

[[nodiscard]] VirtualEpochPhaseResult
finish_virtual_epoch_output(VirtualEpochContext &) noexcept;

[[nodiscard]] VirtualEpochPhaseResult
drain_virtual_epoch_output(VirtualEpochContext &) noexcept;

} // namespace rund::compute::detail
