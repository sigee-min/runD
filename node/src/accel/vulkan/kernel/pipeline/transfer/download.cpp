#include "../../../../backend/result.hpp"

#include "../transfer.hpp"

#include "../../../../../hash/fnv.hpp"
#include "../../../../clock.hpp"
#include "../../../runtime/counter.hpp"

#include <rund/counter.hpp>

#include <cstddef>
#include <cstdint>
#include <mutex>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

BackendDownload
DownloadPreparedVulkanPipeline(const std::shared_ptr<void> &prepared,
                               void *const data, const std::uint64_t bytes,
                               std::uint64_t *const payload_hash) noexcept {
  auto *const pipeline = static_cast<VulkanPipeline *>(prepared.get());
  if (!ValidVulkanPipeline(pipeline) || data == nullptr ||
      payload_hash == nullptr || !pipeline->transfer.ready ||
      bytes != pipeline->transfer.output_bytes ||
      pipeline->transfer.staging.mapped == nullptr) {
    return {};
  }
  std::scoped_lock lock{pipeline->submission.mutex, pipeline->adapter->mutex};
  if (pipeline->submission.active()) {
    return BackendDownload{.check = {false, "compute_pipeline_busy"}};
  }
  if (!pipeline->transfer.output_ready) {
    return BackendDownload{.check = {false, "accel_vulkan_transfer_invalid"}};
  }
  const std::uint64_t readback_begin = MonotonicNanoseconds();
  const auto *const source =
      static_cast<const std::uint8_t *>(pipeline->transfer.staging.mapped);
  auto *const target = static_cast<std::uint8_t *>(data);
  std::uint64_t hash = ::rund::node::hash_detail::kFnvOffset;
  for (std::size_t index = 0u; index < static_cast<std::size_t>(bytes);
       ++index) {
    const std::uint8_t value = source[index];
    target[index] = value;
    hash = (hash ^ value) * ::rund::node::hash_detail::kFnvPrime;
  }
  *payload_hash = hash;
  const std::uint64_t readback_ns = MonotonicNanoseconds() - readback_begin;
  ::rund::detail::counter::Accumulate(pipeline->adapter->readback_ns,
                                      readback_ns);
  ::rund::detail::counter::Accumulate(pipeline->adapter->buffer_reuse_hit_count,
                                      1u);
  pipeline->transfer.input_staged = false;
  pipeline->transfer.output_ready = false;
  return BackendDownload{.check = {true, "ok"},
                         .payload_hash = hash,
                         .staging_bytes = bytes,
                         .staging_peak_bytes = bytes,
                         .staging_reused_bytes = bytes,
                         .buffer_reuses = 1u,
                         .readback_ns = readback_ns,
                         .staging_reused = true,
                         .payload_hash_valid = true};
}

#endif

} // namespace rund::node::accel::detail
