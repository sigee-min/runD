#pragma once

namespace rund::compute_dsl {

struct Axis {
  struct XTag {
    explicit constexpr XTag() noexcept = default;
  };
  struct YTag {
    explicit constexpr YTag() noexcept = default;
  };
  struct ZTag {
    explicit constexpr ZTag() noexcept = default;
  };

  inline static constexpr XTag X{};
  inline static constexpr YTag Y{};
  inline static constexpr ZTag Z{};
};

} // namespace rund::compute_dsl
