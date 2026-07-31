#pragma once

#include <rund/reason.hpp>

#include <cstdint>
#include <string_view>
#include <vector>

namespace rund {

enum class EvidenceTruth : std::uint8_t {
  Unknown,
  Hint,
  Verified,
};

struct Topology final {
  EvidenceTruth numa = EvidenceTruth::Unknown;
  EvidenceTruth affinity = EvidenceTruth::Unknown;
  EvidenceTruth worker_capacity = EvidenceTruth::Unknown;
  std::uint32_t numa_domains = 0u;
  bool spans_numa_domains = false;
};

struct Resources final {
  ReasonCode code = ReasonCode::RuntimeResourcesInvalid;
  std::uint32_t workers = 1u;
  Topology topology{};
  std::vector<std::uint32_t> worker_capacity_milli{};

  [[nodiscard]] constexpr bool ok() const noexcept {
    return code == ReasonCode::Ok;
  }
  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return ok();
  }
  [[nodiscard]] std::string_view error() const noexcept {
    return ok() ? std::string_view{} : std::string_view{ReasonString(code)};
  }
  [[nodiscard]] constexpr int exit_code() const noexcept {
    return ok() ? 0 : 1;
  }
};

} // namespace rund
