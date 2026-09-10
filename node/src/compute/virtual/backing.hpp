#pragma once

#include <rund/compute/virtual.hpp>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>

namespace rund::compute::detail {

// The backing, not a VirtualBuffer wrapper, owns mutation serialization and
// partial-write validity. Multiple typed views over one logical dataset
// therefore cannot publish competing poison or callback order.
struct VirtualBackingState final {
  std::mutex gate;
  // Non-null only for runD-minted resident backings. Custom VirtualBacking
  // implementations cannot manufacture this physical authority.
  std::shared_ptr<BufferState> resident;
  std::uint64_t id{};
  std::uint64_t version{1u};
  // Zero is clean. A nonzero value is the output prefix that a successful
  // retry must overwrite completely before this backing can become readable.
  std::uint64_t recovery_bytes{};
  VirtualBackingTransaction *transaction_provider{};
  VirtualBackingTransactionToken *transaction_token{};
};

struct VirtualBackingAccess final {
  [[nodiscard]] static std::mutex &gate(VirtualBacking &backing) noexcept {
    return backing.state_->gate;
  }

  [[nodiscard]] static std::uint64_t
  recovery_bytes(const VirtualBacking &backing) noexcept {
    return backing.state_->recovery_bytes;
  }

  [[nodiscard]] static const std::shared_ptr<BufferState> &
  resident(const VirtualBacking &backing) noexcept {
    return backing.state_->resident;
  }

  static void bind_resident(VirtualBacking &backing,
                            std::shared_ptr<BufferState> resident) noexcept {
    backing.state_->resident = std::move(resident);
  }

  static void require_recovery(VirtualBacking &backing,
                               const std::uint64_t bytes) noexcept {
    backing.state_->recovery_bytes =
        std::max(backing.state_->recovery_bytes, bytes);
  }

  static void clear_recovery(VirtualBacking &backing) noexcept {
    backing.state_->recovery_bytes = 0u;
  }

  [[nodiscard]] static std::uint64_t
  id(const VirtualBacking &backing) noexcept {
    return backing.state_->id;
  }

  [[nodiscard]] static std::uint64_t
  version(const VirtualBacking &backing) noexcept {
    return backing.state_->version;
  }

  [[nodiscard]] static std::uint32_t
  write_lanes(const VirtualBacking &backing) noexcept {
    const auto *const capability =
        dynamic_cast<const VirtualWriteLanes *>(&backing);
    if (capability == nullptr || capability->write_lanes() < 2u) {
      return 1u;
    }
    return 2u;
  }

  static void publish_write(VirtualBacking &backing) noexcept {
    ++backing.state_->version;
    if (backing.state_->version == 0u) {
      backing.state_->version = 1u;
    }
  }

  static void bind_transaction(VirtualBacking &backing,
                               VirtualBackingTransaction *provider,
                               VirtualBackingTransactionToken *token) noexcept {
    backing.state_->transaction_provider = provider;
    backing.state_->transaction_token = token;
  }

  static void clear_transaction(VirtualBacking &backing) noexcept {
    backing.state_->transaction_provider = nullptr;
    backing.state_->transaction_token = nullptr;
  }

  [[nodiscard]] static VirtualBackingTransaction *
  transaction_provider(VirtualBacking &backing) noexcept {
    return backing.state_->transaction_provider;
  }

  [[nodiscard]] static VirtualBackingTransactionToken *
  transaction_token(VirtualBacking &backing) noexcept {
    return backing.state_->transaction_token;
  }
};

} // namespace rund::compute::detail
