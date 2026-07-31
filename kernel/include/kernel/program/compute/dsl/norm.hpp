#pragma once

namespace rund::compute_dsl {

struct Norm {
  struct L1Tag {
    explicit constexpr L1Tag() noexcept = default;
  };
  struct LInfTag {
    explicit constexpr LInfTag() noexcept = default;
  };

  inline static constexpr L1Tag L1{};
  inline static constexpr LInfTag LInf{};
};

} // namespace rund::compute_dsl
