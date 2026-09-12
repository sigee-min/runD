#include "local.hpp"

#include "src/compute/device/residency/prefetch.hpp"
#include "src/compute/device/residency/registry.hpp"

#include <array>
#include <cstddef>
#include <span>

namespace {

class PrefetchBacking final : public rund::compute::VirtualBacking {
public:
  [[nodiscard]] std::uint64_t size_bytes() const noexcept override {
    return 4096u;
  }

  [[nodiscard]] rund::compute::Status
  read(std::uint64_t, std::span<std::byte>) noexcept override {
    return rund::compute::Status::success();
  }

  [[nodiscard]] rund::compute::Status
  write(std::uint64_t, std::span<const std::byte>) noexcept override {
    return rund::compute::Status::success();
  }
};

} // namespace

namespace rund_node_test_pipeline_residency {

int CheckAuthorityPrefetch() {
  using namespace rund::compute::detail::residency;
  // A Pool may release its Authority regions only after every worker has
  // returned the associated lease token. Idle and unconfigured workers are
  // quiescent; Pending/Ready work is not silently discarded by destruction.
  Prefetcher prefetcher;
  PrefetchBacking backing;
  std::array<std::byte, 4096u> frame{};
  const std::array request{PrefetchRequest{
      .key = {.backing = 50u, .version = 1u, .page = 0u},
      .offset = 0u,
      .bytes = frame.size(),
      .target_offset = 0u,
      .frame = frame.data(),
      .physical_frame = 0u,
      .fetch = false,
  }};
  if (!prefetcher.quiescent() || !prefetcher.configure(4096u, 1u) ||
      !prefetcher.quiescent() ||
      !prefetcher.submit(backing, request, 43u, true) ||
      prefetcher.quiescent()) {
    return 43;
  }
  const PrefetchReceipt receipt = prefetcher.wait();
  if (!receipt.status || receipt.token != 43u || receipt.pages.size() != 1u ||
      receipt.pages.front().fetched || receipt.io_ns != 0u ||
      !prefetcher.quiescent()) {
    return 44;
  }
  const int completion = CheckPrefetchCompletion();
  return completion == 0 ? 0 : 44 + completion;
}

} // namespace rund_node_test_pipeline_residency
