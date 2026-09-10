#pragma once

#include "projection.hpp"

#include "../../../hash/fnv.hpp"

#include <array>

namespace rund::compute::detail {

struct VirtualReduction;
struct VirtualScan;
struct VirtualRunTransaction;

struct VirtualEpochResult final {
  Status status{Status::success()};
  std::uint64_t failed_page{ResidencyStats::no_failed_page};
  bool poison_pipeline{};
};

[[nodiscard]] VirtualEpochResult
execute_virtual_epoch(VirtualPipelineState &state, VirtualBacking &input,
                      VirtualBacking &output, const VirtualRunProjection &run,
                      std::uint64_t epoch,
                      std::array<bool, 2u> &prefetch_pending, Stats &stats,
                      ::rund::node::hash_detail::Fnv &output_hash,
                      VirtualReduction *reduction, VirtualScan *scan,
                      VirtualRunTransaction *transaction = nullptr) noexcept;

} // namespace rund::compute::detail
