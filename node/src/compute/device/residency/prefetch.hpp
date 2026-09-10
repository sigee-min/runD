#pragma once

#include "registry/credentials/epoch.hpp"

#include <rund/compute/virtual.hpp>

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <span>
#include <thread>
#include <vector>

namespace rund::compute::detail::residency {

struct PrefetchRequest final {
  CacheKey key{};
  std::uint64_t offset{};
  std::size_t bytes{};
  std::size_t target_offset{};
  std::uint64_t read_offset{};
  std::size_t read_bytes{};
  std::size_t read_target_offset{};
  std::size_t reuse_source_offset{};
  std::size_t reuse_target_offset{};
  std::size_t reuse_bytes{};
  std::byte *frame{};
  std::uint32_t physical_frame{};
  // A cross-epoch canonical overlap is copied only through an Authority
  // AliasLease retained by the Prefetcher until worker completion. The raw
  // source pointer is meaningful only with this exact nonce/frame pair.
  const std::byte *alias_frame{};
  std::uint32_t alias_source_frame{};
  std::uint64_t alias_nonce{};
  CacheKey alias_source_key{};
  CacheKey alias_target_key{};
  FrameRegion alias_source_region{};
  FrameRegion alias_target_region{};
  std::uint64_t alias_frame_bytes{};
  std::uint64_t alias_owner_token{};
  std::uint64_t alias_generation{};
  bool alias_reuse{};
  bool fetch{};
};

struct PrefetchedPage final {
  CacheKey key{};
  std::size_t bytes{};
  std::size_t backing_bytes{};
  std::size_t target_offset{};
  std::byte *frame{};
  std::uint32_t physical_frame{};
  bool fetched{};
};

struct PrefetchReceipt final {
  Status status{Status::success()};
  std::span<const PrefetchedPage> pages;
  std::uint64_t token{};
  std::uint64_t io_ns{};
  std::uint64_t wait_ns{};
  bool speculative{};
  // The token is the complete execution transform and every page pointer is
  // an authenticated writable view into its exact Device Input binding.
  // False retains the ordinary separate Host-supply lease contract.
  bool coherent_input{};
  // Host-tier service was selected only because the target Device bank was
  // still live.  After its same-bank predecessor terminal, the consumer may
  // promote these exact pages through an authenticated HostVisible view.
  bool coherent_deferred{};
  std::span<AliasLease> aliases;
};

// Each of the two cold-created workers in a Device-global Pool retains request
// metadata only. Payload always lives in Authority-registered Host frames, and
// submit/wait performs no allocation.
class Prefetcher final {
public:
  Prefetcher() = default;
  ~Prefetcher();
  Prefetcher(const Prefetcher &) = delete;
  Prefetcher &operator=(const Prefetcher &) = delete;

  [[nodiscard]] bool configure(std::uint64_t page_bytes,
                               std::uint32_t capacity) noexcept;
  [[nodiscard]] bool submit(VirtualBacking &backing,
                            std::span<const PrefetchRequest> requests,
                            std::uint64_t token, bool speculative,
                            bool coherent_input = false,
                            bool coherent_deferred = false,
                            std::span<AliasLease> aliases = {}) noexcept;
  // A ready observation never consumes the receipt or releases its Authority
  // token. Graph wavefront admission uses this bounded poll to choose the
  // callback that completed first; wait() remains the exact receipt consumer.
  [[nodiscard]] bool ready() const noexcept;
  [[nodiscard]] PrefetchReceipt wait() noexcept;
  // Authenticated cancellation owns the complete terminal sequence: worker
  // quiescence, alias release, then Authority completion/invalidation.  The
  // token remains in this owner until the final Authority step succeeds.
  [[nodiscard]] bool cancel(Authority &, bool source_known = true,
                            bool invalidate = true) noexcept;
  [[nodiscard]] bool cancel(Authority &, PrefetchReceipt &,
                            bool source_known = true,
                            bool invalidate = true) noexcept;
  [[nodiscard]] bool release_aliases(Authority &, bool source_known) noexcept;
  [[nodiscard]] bool quiescent() const noexcept;
  // Exact runD-owned heap extent of the three cold-sized metadata arrays.
  // Native std::thread/runtime bookkeeping is opaque allocator state and is
  // not inferred here.
  [[nodiscard]] std::uint64_t retained_host_bytes() const noexcept;

private:
  enum class State : std::uint8_t { Empty, Idle, Pending, Ready, Stop };

  void work() noexcept;
  [[nodiscard]] bool release_aliases_locked(Authority &,
                                            bool source_known) noexcept;
  [[nodiscard]] bool cancel_locked(Authority &, std::uint64_t token,
                                   bool source_known, bool invalidate) noexcept;

  mutable std::mutex gate_;
  std::condition_variable ready_;
  std::condition_variable pending_;
  std::thread worker_;
  std::vector<PrefetchRequest> requests_;
  std::vector<PrefetchedPage> pages_;
  std::vector<AliasLease> aliases_;
  VirtualBacking *backing_{};
  std::uint64_t page_bytes_{};
  std::size_t count_{};
  std::size_t alias_count_{};
  std::uint64_t token_{};
  bool speculative_{};
  bool coherent_input_{};
  bool coherent_deferred_{};
  Status status_{Status::success()};
  std::uint64_t io_ns_{};
  State state_{State::Empty};
};

} // namespace rund::compute::detail::residency
