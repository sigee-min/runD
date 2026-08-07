#pragma once

#include <cstdint>
#include <type_traits>

namespace rund::detail::task {

struct StopSourceIdentity final {
  std::uint64_t source_id = 0u;
  std::uint64_t generation = 0u;
  std::uint64_t epoch = 0u;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return source_id != 0u && generation != 0u && epoch != 0u;
  }
  [[nodiscard]] constexpr bool empty() const noexcept {
    return source_id == 0u && generation == 0u && epoch == 0u;
  }
  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return valid();
  }

  friend constexpr bool operator==(const StopSourceIdentity &,
                                   const StopSourceIdentity &) noexcept =
      default;
};

struct StopIdentity final {
  std::uint64_t scheduler_id = 0u;
  StopSourceIdentity source_identity{};

  [[nodiscard]] constexpr bool valid() const noexcept {
    return scheduler_id != 0u && source_identity.valid();
  }
  [[nodiscard]] constexpr bool empty() const noexcept {
    return scheduler_id == 0u && source_identity.empty();
  }
  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return valid();
  }
  [[nodiscard]] constexpr StopSourceIdentity source() const noexcept {
    return source_identity;
  }

  friend constexpr bool operator==(const StopIdentity &,
                                   const StopIdentity &) noexcept = default;
};

static_assert(sizeof(StopSourceIdentity) == 24u);
static_assert(sizeof(StopIdentity) == 32u);
static_assert(std::is_standard_layout_v<StopSourceIdentity>);
static_assert(std::is_standard_layout_v<StopIdentity>);
static_assert(std::is_trivially_copyable_v<StopSourceIdentity>);
static_assert(std::is_trivially_copyable_v<StopIdentity>);

} // namespace rund::detail::task
