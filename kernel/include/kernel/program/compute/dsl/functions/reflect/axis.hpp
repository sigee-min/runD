#pragma once

#include <kernel/program/compute/dsl/functions/geometry/projection/component/axis.hpp>
#include <kernel/program/compute/dsl/functions/reflect/component.hpp>

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue
reflect(Axis::XTag axis, const ComputeValue ax, const ComputeValue ay,
        const ComputeValue bx, const ComputeValue by) noexcept {
  return detail::ReflectComponent(ax, proj(axis, ax, ay, bx, by));
}

[[nodiscard]] inline ComputeValue
reflect(Axis::YTag axis, const ComputeValue ax, const ComputeValue ay,
        const ComputeValue bx, const ComputeValue by) noexcept {
  return detail::ReflectComponent(ay, proj(axis, ax, ay, bx, by));
}

[[nodiscard]] inline ComputeValue
reflect(Axis::XTag axis, const ComputeValue ax, const ComputeValue ay,
        const ComputeValue az, const ComputeValue bx, const ComputeValue by,
        const ComputeValue bz) noexcept {
  return detail::ReflectComponent(ax, proj(axis, ax, ay, az, bx, by, bz));
}

[[nodiscard]] inline ComputeValue
reflect(Axis::YTag axis, const ComputeValue ax, const ComputeValue ay,
        const ComputeValue az, const ComputeValue bx, const ComputeValue by,
        const ComputeValue bz) noexcept {
  return detail::ReflectComponent(ay, proj(axis, ax, ay, az, bx, by, bz));
}

[[nodiscard]] inline ComputeValue
reflect(Axis::ZTag axis, const ComputeValue ax, const ComputeValue ay,
        const ComputeValue az, const ComputeValue bx, const ComputeValue by,
        const ComputeValue bz) noexcept {
  return detail::ReflectComponent(az, proj(axis, ax, ay, az, bx, by, bz));
}

} // namespace rund::compute_dsl
