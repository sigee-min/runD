#pragma once

#include "../model.hpp"

#include <array>

namespace rund_node_flow_contract::expression_detail {

using FixedLane32 = rund::compute::Fixed<1, 31>;

[[nodiscard]] int RunCompositeLane32(rund::compute::Backend backend,
                                     const std::array<FixedLane32, 3u> &input,
                                     rund::compute::Stats &stats);

[[nodiscard]] int RunHashLane32(rund::compute::Backend backend,
                                const std::array<FixedLane32, 3u> &input,
                                rund::compute::Stats &stats);

[[nodiscard]] int RunExtendedLane32(rund::compute::Backend backend,
                                    const std::array<FixedLane32, 3u> &input,
                                    rund::compute::Stats &stats);

[[nodiscard]] int RunFunctionalLane32(rund::compute::Backend backend,
                                      const std::array<FixedLane32, 3u> &input,
                                      rund::compute::Stats &stats);

} // namespace rund_node_flow_contract::expression_detail
