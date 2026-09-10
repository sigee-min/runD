#pragma once

#include "../graph_direct.hpp"

#include "../../../../../kernel/backend/run.hpp"

#include <cstdint>
#include <memory>

namespace rund::node::accel::detail::graph_direct_detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

[[nodiscard]] bool
GraphDirectResidentRef(const rund::kernel::ResidentBufferRef *ref,
                       const std::shared_ptr<void> *handle,
                       std::uint32_t usage) noexcept;

[[nodiscard]] bool GraphDirectCanonicalResidentRef(
    const RunBinds &binds, const rund::kernel::ResidentBufferRef *ref,
    const std::shared_ptr<void> *handle, std::uint64_t index,
    std::uint32_t usage) noexcept;

[[nodiscard]] bool GraphDirectMapBindings(const KernelExecution &execution,
                                          const BoundStep &step,
                                          const RunBinds &binds) noexcept;

[[nodiscard]] bool GraphDirectReduceBindings(const KernelExecution &execution,
                                             const BoundStep &step,
                                             const RunBinds &binds) noexcept;

[[nodiscard]] bool GraphDirectExecution(const BackendRun &run,
                                        const BoundStep &bound,
                                        const RunBinds &binds) noexcept;

#endif

} // namespace rund::node::accel::detail::graph_direct_detail
