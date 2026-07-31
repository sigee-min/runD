#include "../../../kernel/backend/execute.hpp"

namespace rund::node::accel::detail {

#if !defined(__APPLE__) || !defined(RUND_NODE_HAVE_METAL_SDK)

rund::AccelCheck PrepareMetalPipeline(const std::span<const BackendBatchEntry>,
                                      const std::span<const BackendBatchEntry>,
                                      const std::span<const std::uint8_t>,
                                      const std::span<const BackendPublish>,
                                      PreparedPipelineStatusLayout &,
                                      const bool,
                                      std::shared_ptr<void> &prepared,
                                      PreparedPipelineMemory &memory) {
  prepared.reset();
  memory = {};
  return rund::AccelCheck{false, "accel_metal_unavailable"};
}

rund::AccelCheck
SeedPreparedMetalPipelineGeneration(const std::shared_ptr<void> &,
                                    const std::uint32_t) noexcept {
  return rund::AccelCheck{false, "accel_metal_unavailable"};
}

rund::AccelCheck SubmitPreparedMetalPipeline(const std::shared_ptr<void> &,
                                             KernelCompletion,
                                             void *) noexcept {
  return rund::AccelCheck{false, "accel_metal_unavailable"};
}

#endif

} // namespace rund::node::accel::detail
