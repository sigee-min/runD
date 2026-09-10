#pragma once

#include "../internal.hpp"

namespace rund_node_test_virtual::product::graph_resident::fixture_detail {

[[nodiscard]] bool run_once(Case &, Observation &, rund::compute::Device &,
                            rund::compute::Backend);
[[nodiscard]] bool run_u32_once(U32Case &, Observation &,
                                rund::compute::Device &,
                                rund::compute::Backend);
[[nodiscard]] bool run_staged_probe(rund::compute::Backend, Variant);

} // namespace rund_node_test_virtual::product::graph_resident::fixture_detail
