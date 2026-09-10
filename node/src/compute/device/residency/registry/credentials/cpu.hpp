#pragma once

#include "../../../../pipeline/residency/model.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace rund::compute::detail::graph_reduce {
class CpuReceiptBook;
} // namespace rund::compute::detail::graph_reduce

namespace rund::compute::detail::residency {

class Authority;
class CpuGraphOwner;
namespace registry_model {
class CpuQuarantineOwner;
} // namespace registry_model

struct GraphPersistIdentity final {
  Identity plan{};
  std::array<std::uint64_t, TiledGraphResourceCapacity> input_backings{};
  std::array<std::uint64_t, TiledGraphResourceCapacity> input_versions{};
  std::uint64_t output_backing{};
  std::uint64_t output_version{};
  std::uint64_t topology_hi{};
  std::uint64_t topology_lo{};
  std::uint64_t stage{};
  std::uint64_t resource{};
  std::uint64_t port{};
  std::uint64_t operation{};
  std::uint64_t frame_capacity{};
  std::uint64_t page_count{};
  std::uint64_t batch_count{};
  std::uint64_t input_page_bytes{};
  std::uint64_t output_page_bytes{};
  std::uint64_t input_payload_bytes{};
  std::uint64_t output_payload_bytes{};
  std::uint64_t input_frame_elements{};
  std::uint32_t input_count{};

  [[nodiscard]] bool valid() const noexcept {
    if (plan == Identity{} || input_count == 0u ||
        input_count > TiledGraphResourceCapacity || output_backing == 0u ||
        output_version == 0u || (topology_hi == 0u && topology_lo == 0u) ||
        frame_capacity == 0u || page_count == 0u || batch_count == 0u ||
        output_page_bytes == 0u || output_payload_bytes == 0u) {
      return false;
    }
    for (std::size_t index = 0u; index < input_count; ++index) {
      if (input_backings[index] == 0u || input_versions[index] == 0u) {
        return false;
      }
    }
    return true;
  }

  bool operator==(const GraphPersistIdentity &) const noexcept = default;
};

class CpuReservationKey final {
public:
  constexpr CpuReservationKey() noexcept = default;
  constexpr CpuReservationKey(const CpuReservationKey &) noexcept = default;
  constexpr CpuReservationKey &
  operator=(const CpuReservationKey &) noexcept = default;

  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return domain_ != 0u && domain_ != Max && slot_ != 0u && slot_ != Max &&
           nonce_ != 0u && nonce_ != Max && owner_ != 0u && owner_ != Max;
  }
  [[nodiscard]] constexpr bool
  operator==(const CpuReservationKey &) const noexcept = default;

private:
  friend class Authority;
  friend class CpuGraphOwner;
  friend class graph_reduce::CpuReceiptBook;
  friend class registry_model::CpuQuarantineOwner;
  static constexpr std::uint64_t Max =
      std::numeric_limits<std::uint64_t>::max();

  [[nodiscard]] static constexpr CpuReservationKey
  make(const std::uint64_t domain, const std::uint64_t slot,
       const std::uint64_t nonce, const std::uint64_t owner) noexcept {
    CpuReservationKey key;
    key.domain_ = domain;
    key.slot_ = slot;
    key.nonce_ = nonce;
    key.owner_ = owner;
    return key;
  }

  [[nodiscard]] constexpr std::uint64_t domain() const noexcept {
    return domain_;
  }
  [[nodiscard]] constexpr std::uint64_t slot() const noexcept { return slot_; }
  [[nodiscard]] constexpr std::uint64_t nonce() const noexcept {
    return nonce_;
  }
  [[nodiscard]] constexpr std::uint64_t owner() const noexcept {
    return owner_;
  }

  std::uint64_t domain_{};
  std::uint64_t slot_{};
  std::uint64_t nonce_{};
  std::uint64_t owner_{};
};

[[nodiscard]] std::uint64_t next_cpu_book_domain() noexcept;
[[nodiscard]] std::uint64_t next_cpu_authority_owner() noexcept;

} // namespace rund::compute::detail::residency
