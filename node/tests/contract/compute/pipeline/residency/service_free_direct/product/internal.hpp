#pragma once

#include "../product.hpp"

namespace rund_node_test_pipeline_residency::service_free_direct_test {

[[nodiscard]] bool product_success_evidence(
    const rund::compute::detail::PipelineState &, std::uint64_t) noexcept;

} // namespace rund_node_test_pipeline_residency::service_free_direct_test
