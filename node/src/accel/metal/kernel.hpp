#pragma once

#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../kernel/backend/run.hpp"
#include "../kernel/callback.hpp"
#include "../kernel/memory.hpp"
#include "../kernel/preparation.hpp"

#include <cstddef>
#include <memory>

namespace rund::node::accel::detail {

struct MetalAdapter;

[[nodiscard]] rund::AccelCheck
PrepareMetalResources(const rund::AccelDevice &pick, const BoundStep *steps,
                      std::size_t step_count, std::uint64_t dispatch_count,
                      KernelPreparationMode mode, const BoundResets *resets,
                      const KernelViewLayout *views, const RunBinds *view_binds,
                      const KernelScratchLayout *scratch,
                      std::uint32_t *failed_node,
                      std::shared_ptr<void> &prepared, PreparedMemory &memory);
[[nodiscard]] std::uint64_t
MetalKernelTraffic(const std::shared_ptr<void> &prepared) noexcept;
[[nodiscard]] rund::AccelCheck
RunMetalResources(const rund::AccelDevice &pick,
                  const std::shared_ptr<void> &prepared);
[[nodiscard]] rund::AccelCheck
SubmitMetalResources(const rund::AccelDevice &pick,
                     const std::shared_ptr<void> &prepared,
                     KernelCompletion completion, void *user) noexcept;

[[nodiscard]] rund::AccelCheck
SeedPreparedMetalPipelineGeneration(const std::shared_ptr<void> &prepared,
                                    std::uint32_t generation) noexcept;

} // namespace rund::node::accel::detail
