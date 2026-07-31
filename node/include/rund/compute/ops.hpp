#pragma once

#include <cstddef>

namespace rund::compute {

enum class Scan : unsigned char {
  InclusiveSum,
  ExclusiveSum,
};

enum class Reduce : unsigned char {
  Sum,
  Min,
  Max,
};

enum class Window : unsigned char {
  Sum,
  Min,
  Max,
};

enum class WindowEdge : unsigned char {
  Clamp,
  Clip,
};

enum class Direction : unsigned char {
  Forward,
  Inverse,
};

enum class FactorOp : unsigned char {
  Lu,
  Qr,
  Cholesky,
};

enum class SpectrumOp : unsigned char {
  Svd,
  Eigen,
};

enum class SpectrumVectors : unsigned char {
  Values,
  Thin,
  Full,
};

struct Compact final {
  std::size_t capacity{};
};

struct Histogram final {
  std::size_t bins{};
};

struct Scatter final {
  std::size_t count{};
};

struct MaxItems final {
  std::size_t value{};
};

struct MaxMatches final {
  std::size_t value{};
};

struct WindowSpec final {
  Window op{Window::Sum};
  std::size_t radius{1};
  WindowEdge edge{WindowEdge::Clamp};
};

struct MatrixShape final {
  std::size_t rows{};
  std::size_t cols{};
  std::size_t batches{1};
};

struct Transform final {
  Direction direction{Direction::Forward};
  bool normalize{false};
};

namespace detail {

enum class MatrixMode : unsigned char {
  Mul,
  Transpose,
  BatchMul,
};

} // namespace detail

} // namespace rund::compute
