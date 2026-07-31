#pragma once

namespace rund::compute_dsl {

struct WindowOpParabolic final {};
struct WindowOpTriangular final {};
struct WindowOpHann final {};
struct WindowOpHamming final {};
struct WindowOpBlackman final {};
struct WindowOpLanczos final {};

struct WindowOp final {
  inline static constexpr WindowOpParabolic Parabolic{};
  inline static constexpr WindowOpTriangular Triangular{};
  inline static constexpr WindowOpHann Hann{};
  inline static constexpr WindowOpHamming Hamming{};
  inline static constexpr WindowOpBlackman Blackman{};
  inline static constexpr WindowOpLanczos Lanczos{};
};

} // namespace rund::compute_dsl
