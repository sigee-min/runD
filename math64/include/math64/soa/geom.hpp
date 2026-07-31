#pragma once

#include <math64/geom/vector.hpp>
#include <math64/geom/view.hpp>
#include <math64/soa/range.hpp>
#include <math64/soa/validate.hpp>

#include <array>

namespace rund::math64::soa {
namespace detail {

inline Status FromViewStatus(const geom::ViewStatus status) noexcept {
  if (status.ok()) {
    return Status{.reason = StatusReason::Ok, .processed = SatProcessed(status.size)};
  }
  const StatusReason reason = status.reason == geom::ViewStatusReason::SizeMismatch
                                  ? StatusReason::SizeMismatch
                                  : StatusReason::ComponentOverlap;
  return Status{.reason = reason, .processed = 0u};
}

inline bool ComponentsOverlap(const geom::Vec3View input,
                              const geom::Vec3MutView out) noexcept {
  return Overlaps(input.x, out.x) || Overlaps(input.x, out.y) ||
         Overlaps(input.x, out.z) || Overlaps(input.y, out.x) ||
         Overlaps(input.y, out.y) || Overlaps(input.y, out.z) ||
         Overlaps(input.z, out.x) || Overlaps(input.z, out.y) ||
         Overlaps(input.z, out.z);
}

inline Status ValidateBinary(const geom::Vec3View lhs, const geom::Vec3View rhs, const geom::Vec3MutView out) noexcept {
  const Status lhs_status = FromViewStatus(geom::Validate(lhs));
  if (!lhs_status.ok()) return lhs_status;
  const Status rhs_status = FromViewStatus(geom::Validate(rhs));
  if (!rhs_status.ok()) return rhs_status;
  const Status out_status = FromViewStatus(geom::Validate(out));
  if (!out_status.ok()) return out_status;
  if (lhs.x.size() != rhs.x.size() || lhs.x.size() != out.x.size()) {
    return Status{.reason = StatusReason::SizeMismatch, .processed = 0u};
  }
  if (ComponentsOverlap(lhs, out) || ComponentsOverlap(rhs, out)) {
    return Status{.reason = StatusReason::InputOutputOverlap, .processed = 0u};
  }
  return Status{.reason = StatusReason::Ok, .processed = SatProcessed(lhs.x.size())};
}

inline Status ValidateTernary(const geom::Vec3View value,
                              const geom::Vec3View lower,
                              const geom::Vec3View upper,
                              const geom::Vec3MutView out) noexcept {
  const Status value_status = ValidateBinary(value, lower, out);
  if (!value_status.ok()) return value_status;
  if (value.x.size() != upper.x.size()) {
    return Status{.reason = StatusReason::SizeMismatch, .processed = 0u};
  }
  if (ComponentsOverlap(upper, out)) {
    return Status{.reason = StatusReason::InputOutputOverlap, .processed = 0u};
  }
  return value_status;
}

inline geom::Vec3x LoadTail(const geom::Vec3View view, const u64 base) noexcept {
  alignas(16) std::array<i64, simd::LaneCount> x{};
  alignas(16) std::array<i64, simd::LaneCount> y{};
  alignas(16) std::array<i64, simd::LaneCount> z{};
  if (base < view.x.size()) {
    x[0] = view.x[base];
    y[0] = view.y[base];
    z[0] = view.z[base];
  }
  return geom::Vec3x{.x = simd::LoadI64(x.data()), .y = simd::LoadI64(y.data()), .z = simd::LoadI64(z.data())};
}

inline void StoreTail(const geom::Vec3MutView out, const u64 base, const geom::Vec3x value) noexcept {
  alignas(16) std::array<i64, simd::LaneCount> x{};
  alignas(16) std::array<i64, simd::LaneCount> y{};
  alignas(16) std::array<i64, simd::LaneCount> z{};
  simd::Store(x.data(), value.x);
  simd::Store(y.data(), value.y);
  simd::Store(z.data(), value.z);
  if (base < out.x.size()) {
    out.x[base] = x[0];
    out.y[base] = y[0];
    out.z[base] = z[0];
  }
}

template <typename Op>
inline Status ApplyBinary(const geom::Vec3View lhs, const geom::Vec3View rhs, const geom::Vec3MutView out, Op op) noexcept {
  const Status status = ValidateBinary(lhs, rhs, out);
  if (!status.ok()) return status;
  u64 base = 0u;
  for (; base + simd::LaneCount <= lhs.x.size(); base += simd::LaneCount) {
    geom::Store(out, base, op(geom::Load(lhs, base), geom::Load(rhs, base)));
  }
  if (base < lhs.x.size()) {
    StoreTail(out, base, op(LoadTail(lhs, base), LoadTail(rhs, base)));
  }
  return status;
}

}  // namespace detail

[[nodiscard]] inline Status Add(const geom::Vec3View lhs, const geom::Vec3View rhs, const geom::Vec3MutView out) noexcept {
  return detail::ApplyBinary(lhs, rhs, out, [](const geom::Vec3x a, const geom::Vec3x b) noexcept { return geom::Add(a, b); });
}

[[nodiscard]] inline Status Sub(const geom::Vec3View lhs, const geom::Vec3View rhs, const geom::Vec3MutView out) noexcept {
  return detail::ApplyBinary(lhs, rhs, out, [](const geom::Vec3x a, const geom::Vec3x b) noexcept { return geom::Sub(a, b); });
}

[[nodiscard]] inline Status Clamp(const geom::Vec3View value,
                                  const geom::Vec3View lower,
                                  const geom::Vec3View upper,
                                  const geom::Vec3MutView out) noexcept {
  const Status status = detail::ValidateTernary(value, lower, upper, out);
  if (!status.ok()) return status;
  u64 base = 0u;
  for (; base + simd::LaneCount <= value.x.size(); base += simd::LaneCount) {
    geom::Store(out, base, geom::Clamp(geom::Load(value, base), geom::Load(lower, base), geom::Load(upper, base)));
  }
  if (base < value.x.size()) {
    detail::StoreTail(out, base, geom::Clamp(detail::LoadTail(value, base), detail::LoadTail(lower, base), detail::LoadTail(upper, base)));
  }
  return status;
}

[[nodiscard]] inline Status AddInPlace(const geom::Vec3MutView lhs, const geom::Vec3View rhs) noexcept {
  const Status lhs_status = detail::FromViewStatus(geom::Validate(lhs));
  if (!lhs_status.ok()) return lhs_status;
  const Status rhs_status = detail::FromViewStatus(geom::Validate(rhs));
  if (!rhs_status.ok()) return rhs_status;
  if (lhs.x.size() != rhs.x.size()) {
    return Status{.reason = StatusReason::SizeMismatch, .processed = 0u};
  }
  if (detail::ComponentsOverlap(rhs, lhs)) {
    return Status{.reason = StatusReason::InputOutputOverlap, .processed = 0u};
  }
  geom::Vec3View lhs_view{.x = lhs.x, .y = lhs.y, .z = lhs.z};
  u64 base = 0u;
  for (; base + simd::LaneCount <= lhs.x.size(); base += simd::LaneCount) {
    geom::Store(lhs, base, geom::Add(geom::Load(lhs_view, base), geom::Load(rhs, base)));
  }
  if (base < lhs.x.size()) {
    detail::StoreTail(lhs, base, geom::Add(detail::LoadTail(lhs_view, base), detail::LoadTail(rhs, base)));
  }
  return Status{.reason = StatusReason::Ok, .processed = detail::SatProcessed(lhs.x.size())};
}

}  // namespace rund::math64::soa
