#pragma once

#include "projection.hpp"

#include "../../../hash/fnv.hpp"

namespace rund::compute::detail {

struct VirtualWaveResult final {
  Status status{Status::success()};
  std::uint64_t failed_page{ResidencyStats::no_failed_page};
  bool poison_pipeline{};
};

[[nodiscard]] VirtualWaveResult
execute_virtual_wave(VirtualPipelineState &state, VirtualBacking &input,
                     VirtualBacking &output, const VirtualRunProjection &run,
                     std::uint64_t wave, Stats &stats,
                     ::rund::node::hash_detail::Fnv &output_hash) noexcept;

} // namespace rund::compute::detail
