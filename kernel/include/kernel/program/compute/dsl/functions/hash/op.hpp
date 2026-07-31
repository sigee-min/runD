#pragma once

namespace rund::compute_dsl {

struct HashOpUnit final {};

struct HashOp final {
  inline static constexpr HashOpUnit Unit{};
};

} // namespace rund::compute_dsl
