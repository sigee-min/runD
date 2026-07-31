#pragma once

namespace rund::compute_dsl {

struct ActivationOpRelu final {};
struct ActivationOpLeakyRelu final {};
struct ActivationOpHardSigmoid final {};
struct ActivationOpHardSwish final {};
struct ActivationOpHardTanh final {};

struct ActivationOp final {
  inline static constexpr ActivationOpRelu Relu{};
  inline static constexpr ActivationOpLeakyRelu LeakyRelu{};
  inline static constexpr ActivationOpHardSigmoid HardSigmoid{};
  inline static constexpr ActivationOpHardSwish HardSwish{};
  inline static constexpr ActivationOpHardTanh HardTanh{};
};

} // namespace rund::compute_dsl
