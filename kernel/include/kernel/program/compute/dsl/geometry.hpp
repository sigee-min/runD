#pragma once

namespace rund::compute_dsl {

struct GeometryOpParameter final {};
struct GeometryOpProjection final {};
struct GeometryOpDistance final {};

struct LineOp final {
  inline static constexpr GeometryOpParameter Parameter{};
  inline static constexpr GeometryOpProjection Projection{};
  inline static constexpr GeometryOpDistance Distance{};
};

struct PlaneOp final {
  inline static constexpr GeometryOpParameter Parameter{};
  inline static constexpr GeometryOpProjection Projection{};
  inline static constexpr GeometryOpDistance Distance{};
};

} // namespace rund::compute_dsl
