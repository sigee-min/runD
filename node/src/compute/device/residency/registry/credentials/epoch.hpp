#pragma once

#include "../graph.hpp"
#include "../transfer.hpp"
#include "result.hpp"

#include <cstdint>
#include <span>

namespace rund::compute::detail::residency {

class Authority;

struct EpochLease final {
  std::span<const CacheBinding> bindings;
  std::span<const CacheTransition> transitions;
  std::span<const GraphLeasePort> ports;
  std::span<const GraphRelocation> relocations;
  std::span<const GraphPageRemap> remaps;
  std::uint64_t token{};
  std::uint64_t generation{};
};

class AliasLease final {
public:
  AliasLease() = default;
  AliasLease(const AliasLease &) = delete;
  AliasLease &operator=(const AliasLease &) = delete;
  AliasLease(AliasLease &&other) noexcept
      : source_key_(other.source_key_), target_key_(other.target_key_),
        source_region_(other.source_region_),
        target_region_(other.target_region_),
        source_frame_(other.source_frame_), target_frame_(other.target_frame_),
        source_offset_(other.source_offset_),
        target_offset_(other.target_offset_), bytes_(other.bytes_),
        frame_bytes_(other.frame_bytes_), owner_token_(other.owner_token_),
        generation_(other.generation_), nonce_(other.nonce_) {
    other.clear();
  }
  AliasLease &operator=(AliasLease &&other) noexcept {
    if (this != &other) {
      source_key_ = other.source_key_;
      target_key_ = other.target_key_;
      source_region_ = other.source_region_;
      target_region_ = other.target_region_;
      source_frame_ = other.source_frame_;
      target_frame_ = other.target_frame_;
      source_offset_ = other.source_offset_;
      target_offset_ = other.target_offset_;
      bytes_ = other.bytes_;
      frame_bytes_ = other.frame_bytes_;
      owner_token_ = other.owner_token_;
      generation_ = other.generation_;
      nonce_ = other.nonce_;
      other.clear();
    }
    return *this;
  }

  [[nodiscard]] explicit operator bool() const noexcept { return nonce_ != 0u; }
  [[nodiscard]] CacheKey source_key() const noexcept { return source_key_; }
  [[nodiscard]] CacheKey target_key() const noexcept { return target_key_; }
  [[nodiscard]] FrameRegion source_region() const noexcept {
    return source_region_;
  }
  [[nodiscard]] FrameRegion target_region() const noexcept {
    return target_region_;
  }
  [[nodiscard]] std::uint32_t source_frame() const noexcept {
    return source_frame_;
  }
  [[nodiscard]] std::uint32_t target_frame() const noexcept {
    return target_frame_;
  }
  [[nodiscard]] std::uint64_t source_offset() const noexcept {
    return source_offset_;
  }
  [[nodiscard]] std::uint64_t target_offset() const noexcept {
    return target_offset_;
  }
  [[nodiscard]] std::uint64_t bytes() const noexcept { return bytes_; }
  [[nodiscard]] std::uint64_t frame_bytes() const noexcept {
    return frame_bytes_;
  }
  [[nodiscard]] std::uint64_t owner_token() const noexcept {
    return owner_token_;
  }
  [[nodiscard]] std::uint64_t generation() const noexcept {
    return generation_;
  }
  [[nodiscard]] std::uint64_t nonce() const noexcept { return nonce_; }

private:
  friend class Authority;

  void clear() noexcept {
    source_key_ = {};
    target_key_ = {};
    source_region_ = {};
    target_region_ = {};
    source_frame_ = 0u;
    target_frame_ = 0u;
    source_offset_ = 0u;
    target_offset_ = 0u;
    bytes_ = 0u;
    frame_bytes_ = 0u;
    owner_token_ = 0u;
    generation_ = 0u;
    nonce_ = 0u;
  }

  CacheKey source_key_{};
  CacheKey target_key_{};
  FrameRegion source_region_{};
  FrameRegion target_region_{};
  std::uint32_t source_frame_{};
  std::uint32_t target_frame_{};
  std::uint64_t source_offset_{};
  std::uint64_t target_offset_{};
  std::uint64_t bytes_{};
  std::uint64_t frame_bytes_{};
  std::uint64_t owner_token_{};
  std::uint64_t generation_{};
  std::uint64_t nonce_{};
};

struct AuthorityResult final {
  AuthorityFailure failure{AuthorityFailure::Invalid};
  EpochLease lease{};

  [[nodiscard]] explicit operator bool() const noexcept {
    return failure == AuthorityFailure::None;
  }
};

} // namespace rund::compute::detail::residency
