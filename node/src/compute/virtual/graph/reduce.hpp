#pragma once

#include "../run/projection.hpp"
#include "../run/reduce.hpp"

#include <span>

namespace rund::compute {
class VirtualBacking;
}

namespace rund::compute::detail {

struct VirtualGraphResult final {
  Status status{Status::success()};
  std::uint64_t failed_page{ResidencyStats::no_failed_page};
  std::uint64_t output_hash{};
  bool poison_pipeline{};
};

[[nodiscard]] VirtualGraphResult
execute_virtual_graph_reduction(VirtualPipelineState &state,
                                std::span<VirtualBacking *const> inputs,
                                const VirtualRunProjection &run, Stats &stats,
                                VirtualReduction &reduction) noexcept;

[[nodiscard]] VirtualGraphResult execute_virtual_graph_pointwise(
    VirtualPipelineState &state, std::span<VirtualBacking *const> inputs,
    VirtualBacking &output, const VirtualRunProjection &run,
    Stats &stats) noexcept;

} // namespace rund::compute::detail
