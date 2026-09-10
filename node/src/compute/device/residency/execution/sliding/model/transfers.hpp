#pragma once

#include "ticket.hpp"

#include <array>

namespace rund::compute::detail::residency::execution {

class SlidingFetch final {
public:
  SlidingFetch() = default;
  SlidingFetch(const SlidingFetch &) = delete;
  SlidingFetch &operator=(const SlidingFetch &) = delete;
  SlidingFetch(SlidingFetch &&other) noexcept
      : ticket_(other.ticket_), source_(other.source_), reuse_(other.reuse_),
        frame_(other.frame_), reuse_frame_(other.reuse_frame_),
        backing_bytes_(other.backing_bytes_), nonce_(other.nonce_),
        backing_(other.backing_), reuses_frame_(other.reuses_frame_) {
    other.ticket_ = {};
    other.source_ = {};
    other.reuse_ = {};
    other.frame_ = 0u;
    other.reuse_frame_ = 0u;
    other.backing_bytes_ = 0u;
    other.nonce_ = 0u;
    other.backing_ = false;
    other.reuses_frame_ = false;
  }
  SlidingFetch &operator=(SlidingFetch &&) = delete;

  [[nodiscard]] explicit operator bool() const noexcept { return nonce_ != 0u; }
  [[nodiscard]] bool requires_backing() const noexcept { return backing_; }
  [[nodiscard]] std::uint32_t frame() const noexcept { return frame_; }
  [[nodiscard]] SlidingCoordinate coordinate() const noexcept {
    return ticket_.coordinate;
  }
  [[nodiscard]] std::uint32_t use() const noexcept { return ticket_.use; }
  [[nodiscard]] FetchSource source() const noexcept { return source_; }
  [[nodiscard]] bool reuses_frame() const noexcept { return reuses_frame_; }
  [[nodiscard]] std::uint32_t reuse_frame() const noexcept {
    return reuse_frame_;
  }
  [[nodiscard]] FetchReuseSource reuse() const noexcept { return reuse_; }
  [[nodiscard]] std::uint64_t backing_bytes() const noexcept {
    return backing_bytes_;
  }

private:
  friend class ::rund::compute::detail::residency::Authority;
  friend class ::rund::compute::detail::residency::SlidingOwner;
  SlidingTicket ticket_{};
  FetchSource source_{};
  FetchReuseSource reuse_{};
  std::uint32_t frame_{};
  std::uint32_t reuse_frame_{};
  std::uint64_t backing_bytes_{};
  std::uint64_t nonce_{};
  bool backing_{};
  bool reuses_frame_{};
};

class SlidingPromote final {
public:
  SlidingPromote() = default;
  SlidingPromote(const SlidingPromote &) = delete;
  SlidingPromote &operator=(const SlidingPromote &) = delete;
  SlidingPromote(SlidingPromote &&other) noexcept
      : ticket_(other.ticket_), host_frames_(other.host_frames_),
        device_input_frames_(other.device_input_frames_),
        device_output_frames_(other.device_output_frames_),
        host_output_frames_(other.host_output_frames_),
        targets_(other.targets_), slices_(other.slices_), nonce_(other.nonce_),
        transfer_mask_(other.transfer_mask_),
        source_count_(other.source_count_), input_count_(other.input_count_),
        output_count_(other.output_count_), slice_count_(other.slice_count_) {
    other.ticket_ = {};
    other.host_frames_ = {};
    other.device_input_frames_ = {};
    other.device_output_frames_ = {};
    other.host_output_frames_ = {};
    other.targets_ = {};
    other.slices_ = {};
    other.nonce_ = 0u;
    other.transfer_mask_ = 0u;
    other.source_count_ = 0u;
    other.input_count_ = 0u;
    other.output_count_ = 0u;
    other.slice_count_ = 0u;
  }
  SlidingPromote &operator=(SlidingPromote &&) = delete;

  [[nodiscard]] explicit operator bool() const noexcept { return nonce_ != 0u; }
  [[nodiscard]] SlidingCoordinate coordinate() const noexcept {
    return ticket_.coordinate;
  }
  [[nodiscard]] std::uint64_t expected_bytes() const noexcept {
    return ticket_.expected_bytes;
  }
  [[nodiscard]] std::uint32_t transfer_mask() const noexcept {
    return transfer_mask_;
  }
  [[nodiscard]] bool assembles_window() const noexcept {
    return slice_count_ != 0u;
  }
  [[nodiscard]] std::span<const std::uint32_t> host_frames() const noexcept {
    return {host_frames_.data(), source_count_};
  }
  [[nodiscard]] std::span<const std::uint32_t>
  device_input_frames() const noexcept {
    return {device_input_frames_.data(), input_count_};
  }
  [[nodiscard]] std::span<const std::uint32_t>
  device_output_frames() const noexcept {
    return {device_output_frames_.data(), output_count_};
  }
  [[nodiscard]] std::span<const std::uint32_t>
  host_output_frames() const noexcept {
    return {host_output_frames_.data(), output_count_};
  }
  [[nodiscard]] std::span<const FetchSource> targets() const noexcept {
    return {targets_.data(), input_count_};
  }
  [[nodiscard]] std::span<const WindowFootprintSlice> slices() const noexcept {
    return {slices_.data(), slice_count_};
  }

private:
  friend class ::rund::compute::detail::residency::Authority;
  friend class ::rund::compute::detail::residency::SlidingOwner;
  SlidingTicket ticket_{};
  std::array<std::uint32_t, WindowFootprintSourceCapacity> host_frames_{};
  std::array<std::uint32_t, UseCapacity> device_input_frames_{};
  std::array<std::uint32_t, UseCapacity> device_output_frames_{};
  std::array<std::uint32_t, UseCapacity> host_output_frames_{};
  std::array<FetchSource, UseCapacity> targets_{};
  std::array<WindowFootprintSlice, WindowFootprintSliceCapacity> slices_{};
  std::uint64_t nonce_{};
  std::uint32_t transfer_mask_{};
  std::uint32_t source_count_{};
  std::uint32_t input_count_{};
  std::uint32_t output_count_{};
  std::uint32_t slice_count_{};
};

class SlidingNative final {
public:
  SlidingNative() = default;
  SlidingNative(const SlidingNative &) = delete;
  SlidingNative &operator=(const SlidingNative &) = delete;
  SlidingNative(SlidingNative &&other) noexcept
      : ticket_(other.ticket_), input_frames_(other.input_frames_),
        output_frames_(other.output_frames_), nonce_(other.nonce_),
        active_mask_(other.active_mask_), input_count_(other.input_count_),
        output_count_(other.output_count_) {
    other.ticket_ = {};
    other.input_frames_ = {};
    other.output_frames_ = {};
    other.nonce_ = 0u;
    other.active_mask_ = 0u;
    other.input_count_ = 0u;
    other.output_count_ = 0u;
  }
  SlidingNative &operator=(SlidingNative &&) = delete;

  [[nodiscard]] explicit operator bool() const noexcept { return nonce_ != 0u; }
  [[nodiscard]] SlidingCoordinate coordinate() const noexcept {
    return ticket_.coordinate;
  }
  [[nodiscard]] std::uint32_t active_mask() const noexcept {
    return active_mask_;
  }
  [[nodiscard]] std::span<const std::uint32_t> input_frames() const noexcept {
    return {input_frames_.data(), input_count_};
  }
  [[nodiscard]] std::span<const std::uint32_t> output_frames() const noexcept {
    return {output_frames_.data(), output_count_};
  }

private:
  friend class ::rund::compute::detail::residency::Authority;
  friend class ::rund::compute::detail::residency::SlidingOwner;
  SlidingTicket ticket_{};
  std::array<std::uint32_t, UseCapacity> input_frames_{};
  std::array<std::uint32_t, UseCapacity> output_frames_{};
  std::uint64_t nonce_{};
  std::uint32_t active_mask_{};
  std::uint32_t input_count_{};
  std::uint32_t output_count_{};
};

class SlidingDrain final {
public:
  SlidingDrain() = default;
  SlidingDrain(const SlidingDrain &) = delete;
  SlidingDrain &operator=(const SlidingDrain &) = delete;
  SlidingDrain(SlidingDrain &&other) noexcept
      : ticket_(other.ticket_), device_frame_(other.device_frame_),
        host_frame_(other.host_frame_), nonce_(other.nonce_) {
    other.ticket_ = {};
    other.device_frame_ = 0u;
    other.host_frame_ = 0u;
    other.nonce_ = 0u;
  }
  SlidingDrain &operator=(SlidingDrain &&) = delete;

  [[nodiscard]] explicit operator bool() const noexcept { return nonce_ != 0u; }
  [[nodiscard]] SlidingCoordinate coordinate() const noexcept {
    return ticket_.coordinate;
  }
  [[nodiscard]] std::uint32_t use() const noexcept { return ticket_.use; }
  [[nodiscard]] std::uint64_t expected_bytes() const noexcept {
    return ticket_.expected_bytes;
  }
  [[nodiscard]] std::uint32_t device_frame() const noexcept {
    return device_frame_;
  }
  [[nodiscard]] std::uint32_t host_frame() const noexcept {
    return host_frame_;
  }

private:
  friend class ::rund::compute::detail::residency::Authority;
  friend class ::rund::compute::detail::residency::SlidingOwner;
  SlidingTicket ticket_{};
  std::uint32_t device_frame_{};
  std::uint32_t host_frame_{};
  std::uint64_t nonce_{};
};

class SlidingPersist final {
public:
  SlidingPersist() = default;
  SlidingPersist(const SlidingPersist &) = delete;
  SlidingPersist &operator=(const SlidingPersist &) = delete;
  SlidingPersist(SlidingPersist &&other) noexcept
      : ticket_(other.ticket_), key_(other.key_), extent_(other.extent_),
        frame_(other.frame_), nonce_(other.nonce_) {
    other.ticket_ = {};
    other.key_ = {};
    other.extent_ = {};
    other.frame_ = 0u;
    other.nonce_ = 0u;
  }
  SlidingPersist &operator=(SlidingPersist &&) = delete;

  [[nodiscard]] explicit operator bool() const noexcept { return nonce_ != 0u; }
  [[nodiscard]] SlidingCoordinate coordinate() const noexcept {
    return ticket_.coordinate;
  }
  [[nodiscard]] std::uint32_t use() const noexcept { return ticket_.use; }
  [[nodiscard]] std::uint64_t expected_bytes() const noexcept {
    return ticket_.expected_bytes;
  }
  [[nodiscard]] std::uint32_t frame() const noexcept { return frame_; }
  [[nodiscard]] residency::CacheKey key() const noexcept { return key_; }
  [[nodiscard]] residency::DirtyExtent extent() const noexcept {
    return extent_;
  }

private:
  friend class ::rund::compute::detail::residency::Authority;
  friend class ::rund::compute::detail::residency::SlidingOwner;
  SlidingTicket ticket_{};
  residency::CacheKey key_{};
  residency::DirtyExtent extent_{};
  std::uint32_t frame_{};
  std::uint64_t nonce_{};
};

} // namespace rund::compute::detail::residency::execution
