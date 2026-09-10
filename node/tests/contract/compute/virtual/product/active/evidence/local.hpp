#pragma once

#include "../local.hpp"

namespace rund_node_test_virtual::product::active::evidence_detail {

[[nodiscard]] bool ExactActiveStats(const rund::compute::Stats &,
                                    rund::compute::Backend,
                                    std::size_t) noexcept;
void ReportActiveFirstFalse(const char *, std::uint64_t,
                            std::uint64_t) noexcept;
void ReportActiveStatsFirstFalse(const rund::compute::Stats &,
                                 rund::compute::Backend, std::size_t) noexcept;

} // namespace rund_node_test_virtual::product::active::evidence_detail
