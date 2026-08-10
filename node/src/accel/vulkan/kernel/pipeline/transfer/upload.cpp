#include "../transfer.hpp"

#include "../../../runtime/counter.hpp"

#include <rund/counter.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

BackendUpload
UploadPreparedVulkanPipeline(const std::shared_ptr<void> &prepared,
                             const void *const data,
                             const std::uint64_t bytes) noexcept {
  auto *const pipeline = static_cast<VulkanPipeline *>(prepared.get());
  if (!ValidVulkanPipeline(pipeline) || data == nullptr ||
      !pipeline->transfer.ready || bytes != pipeline->transfer.input_bytes ||
      pipeline->transfer.staging.mapped == nullptr) {
    return {};
  }
  std::scoped_lock lock{pipeline->submission.mutex, pipeline->adapter->mutex};
  if (pipeline->submission.active()) {
    return BackendUpload{.check = {false, "compute_pipeline_busy"}};
  }
  std::memcpy(pipeline->transfer.staging.mapped, data,
              static_cast<std::size_t>(bytes));
  pipeline->transfer.input_staged = true;
  pipeline->transfer.output_ready = false;
  ::rund::detail::counter::Accumulate(pipeline->adapter->buffer_reuse_hit_count,
                                      1u);
  return BackendUpload{.check = {true, "ok"},
                       .staging_bytes = bytes,
                       .staging_peak_bytes = bytes,
                       .staging_reused_bytes = bytes,
                       .buffer_reuses = 1u};
}

#endif

} // namespace rund::node::accel::detail
