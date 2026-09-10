#pragma once

#include "src/accel/kernel/residency/service_free_direct/proof.hpp"
#include "src/compute/pipeline/state.hpp"

#include <rund/compute/pipeline.hpp>

#include <memory>

namespace rund_node_test_pipeline_residency::service_free_direct_test {

[[nodiscard]] bool product_known_rejection_retry(
    rund::compute::Pipeline &,
    const std::shared_ptr<rund::compute::detail::PipelineState> &,
    const std::shared_ptr<
        const rund::node::accel::detail::ServiceFreeDirectProof> &) noexcept;

[[nodiscard]] bool product_unknown_quarantine(
    rund::compute::Pipeline &,
    const std::shared_ptr<rund::compute::detail::PipelineState> &) noexcept;

} // namespace rund_node_test_pipeline_residency::service_free_direct_test
