#include "../../../../../src/accel/backend/result.hpp"

#include "local.hpp"

#include "src/accel/backend/resource.hpp"
#include "src/accel/backend/token.hpp"
#include "src/accel/metal/state.hpp"
#include "src/accel/vulkan/adapter/access.hpp"
#include <node/accel/buffer.hpp>
#include <node/accel/pick.hpp>

#include <accel/runtime.hpp>

#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace node_accel_contract::backend_runtime {
namespace {

namespace detail = rund::node::accel::detail;

template <class Adapter>
[[nodiscard]] bool WaitForActiveHostReadback(Adapter *const adapter) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds{5};
  while (std::chrono::steady_clock::now() < deadline) {
    {
      std::lock_guard lock{adapter->mutex};
      if (adapter->active_host_readbacks != 0u) {
        return true;
      }
    }
    std::this_thread::yield();
  }
  return false;
}

template <class Adapter>
[[nodiscard]] bool HostReadbackEpochContract(rund::AccelDevice pick,
                                             Adapter *const adapter,
                                             const bool destroy_owner) {
  constexpr std::size_t kBytes = 16u * 1024u * 1024u;
  if (adapter == nullptr) {
    return false;
  }
  std::shared_ptr<detail::PickToken> token = detail::AdmitPick(pick);
  if (token == nullptr) {
    return false;
  }
  rund::Buffer buffer = detail::CreateBackendBuffer(
      token, rund::BufferDesc{.bytes = kBytes,
                              .usage = rund::BufferUsage::ReadWrite,
                              .alignment = 16u});
  std::vector<std::uint8_t> input(kBytes, 0x5au);
  std::vector<std::uint8_t> output(kBytes);
  if (!buffer.check.ok || !detail::UploadBackendBuffer(
                               token, buffer, input.data(), input.size(), 0u)
                               .ok) {
    return false;
  }
  rund::node::accel::ResetRuntimeStats(pick);

  detail::BackendDownload downloaded{};
  std::thread reader{[token, owned_buffer = buffer, &downloaded, &output] {
    downloaded = detail::DownloadBackendBuffer(
        token, owned_buffer, output.data(), output.size(), 0u, true);
  }};
  if (!WaitForActiveHostReadback(adapter)) {
    reader.join();
    return false;
  }

  if (destroy_owner) {
    buffer = {};
    token.reset();
    pick = {};
  } else {
    rund::node::accel::ResetRuntimeStats(pick);
  }
  reader.join();
  if (!downloaded.check.ok || !downloaded.payload_hash_valid ||
      output != input) {
    return false;
  }
  if (destroy_owner) {
    return true;
  }
  const rund::RuntimeStats after = rund::node::accel::ReadRuntimeStats(pick);
  return after.outcome.ok && after.run.transfer.device_to_host_bytes == 0u &&
         after.run.time.readback_ns == 0u;
}

} // namespace

bool CheckMetalHostReadback() {
  rund::AccelDevice epoch_pick = Pick(rund::AccelApi::Metal);
  if (!epoch_pick.check.ok) {
    return node_accel_contract::MetalFailsClosed(epoch_pick);
  }
  const std::shared_ptr<detail::PickToken> epoch_token =
      detail::AdmitPick(epoch_pick);
  const rund::AccelDevice *const epoch_raw =
      epoch_token == nullptr ? nullptr : &epoch_token->raw;
  auto *const epoch_adapter =
      epoch_raw == nullptr
          ? nullptr
          : static_cast<detail::MetalAdapter *>(epoch_raw->backend.context);
  if (!HostReadbackEpochContract(epoch_pick, epoch_adapter, false)) {
    return false;
  }

  rund::AccelDevice destroy_pick = Pick(rund::AccelApi::Metal);
  std::shared_ptr<detail::PickToken> destroy_token =
      detail::AdmitPick(destroy_pick);
  const rund::AccelDevice *const destroy_raw =
      destroy_token == nullptr ? nullptr : &destroy_token->raw;
  auto *const destroy_adapter =
      destroy_raw == nullptr
          ? nullptr
          : static_cast<detail::MetalAdapter *>(destroy_raw->backend.context);
  destroy_token.reset();
  return HostReadbackEpochContract(std::move(destroy_pick), destroy_adapter,
                                   true);
}

bool CheckVulkanHostReadback() {
  rund::AccelDevice epoch_pick = Pick(rund::AccelApi::Vulkan);
  if (!epoch_pick.check.ok) {
    return node_accel_contract::vulkan::FailureReasonIsPrecise(epoch_pick);
  }
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  const std::shared_ptr<detail::PickToken> epoch_token =
      detail::AdmitPick(epoch_pick);
  const rund::AccelDevice *const epoch_raw =
      epoch_token == nullptr ? nullptr : &epoch_token->raw;
  detail::VulkanAdapter *const epoch_adapter =
      epoch_raw == nullptr ? nullptr : detail::CheckedVulkanAdapter(*epoch_raw);
  if (!HostReadbackEpochContract(epoch_pick, epoch_adapter, false)) {
    return false;
  }

  rund::AccelDevice destroy_pick = Pick(rund::AccelApi::Vulkan);
  std::shared_ptr<detail::PickToken> destroy_token =
      detail::AdmitPick(destroy_pick);
  const rund::AccelDevice *const destroy_raw =
      destroy_token == nullptr ? nullptr : &destroy_token->raw;
  detail::VulkanAdapter *const destroy_adapter =
      destroy_raw == nullptr ? nullptr
                             : detail::CheckedVulkanAdapter(*destroy_raw);
  destroy_token.reset();
  return HostReadbackEpochContract(std::move(destroy_pick), destroy_adapter,
                                   true);
#else
  return false;
#endif
}

} // namespace node_accel_contract::backend_runtime
