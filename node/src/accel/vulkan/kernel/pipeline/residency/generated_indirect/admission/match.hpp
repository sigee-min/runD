#pragma once

#include "../internal.hpp"

#include "../../../../../map/api.hpp"
#include "../../../../../map/local.hpp"
#include "../../../../ops/table.hpp"
#include "../../../prepare/record.hpp"
#include "../../../../../../kernel/bindings/step.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace rund::node::accel::detail::vulkan_generated_indirect_detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

namespace impl {

[[nodiscard]] const BackendRun *
pipeline_run(const VulkanPipelineRecordEntry &entry) noexcept;

[[nodiscard]] const BoundStep *
entry_step(const VulkanPipelineRecordEntry &entry,
           const BackendRun *run) noexcept;

[[nodiscard]] bool
copy_binding(const rund::kernel::ResidentBufferRef *ref,
             const std::shared_ptr<void> *handle, std::size_t index,
             VulkanResidencyGraphStageGeneratedProof &proof) noexcept;

[[nodiscard]] bool
same_resident_storage(const rund::kernel::ResidentBufferRef &left,
                      const std::shared_ptr<void> *left_handle,
                      const rund::kernel::ResidentBufferRef &right,
                      const std::shared_ptr<void> *right_handle) noexcept;

[[nodiscard]] bool graph_alias_matches(
    const KernelExecution &execution, const RunBinds &source,
    std::uint64_t graph_binding,
    const rund::kernel::ResidentBufferRef &resident,
    const std::shared_ptr<void> *resident_handle) noexcept;

[[nodiscard]] bool
same_proof_storage(const VulkanResidencyGraphStageGeneratedProof &left,
                   std::size_t left_index,
                   const VulkanResidencyGraphStageGeneratedProof &right,
                   std::size_t right_index) noexcept;

} // namespace impl

#endif

} // namespace rund::node::accel::detail::vulkan_generated_indirect_detail
