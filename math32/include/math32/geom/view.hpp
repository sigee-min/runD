#pragma once

#include <math32/core/model.hpp>
#include <math32/geom/vec3x.hpp>
#include <math32/simd/memory.hpp>
#include <math32/soa/range.hpp>

#include <span>

namespace rund::math32::geom {

struct Vec2View {
  std::span<const i32> x;
  std::span<const i32> y;
};

struct Vec2MutView {
  std::span<i32> x;
  std::span<i32> y;
};

struct Vec3View {
  std::span<const i32> x;
  std::span<const i32> y;
  std::span<const i32> z;
};

struct Vec3MutView {
  std::span<i32> x;
  std::span<i32> y;
  std::span<i32> z;
};

enum class ViewStatusReason : u8 {
  Ok = 0u,
  SizeMismatch = 1u,
  ComponentOverlap = 2u,
  LaneOutOfRange = 3u,
  NotEvaluated = 4u,
};

struct ViewStatus {
  ViewStatusReason reason = ViewStatusReason::NotEvaluated;
  u64 size = 0u;
  [[nodiscard]] constexpr bool ok() const noexcept {
    return reason == ViewStatusReason::Ok;
  }
  constexpr explicit operator bool() const noexcept { return ok(); }
};

inline bool SizeMatch(const Vec2View view) noexcept {
  return view.x.size() == view.y.size();
}

inline bool SizeMatch(const Vec2MutView view) noexcept {
  return view.x.size() == view.y.size();
}

inline bool SizeMatch(const Vec3View view) noexcept {
  return view.x.size() == view.y.size() && view.x.size() == view.z.size();
}

inline bool SizeMatch(const Vec3MutView view) noexcept {
  return view.x.size() == view.y.size() && view.x.size() == view.z.size();
}

inline ViewStatus Validate(const Vec2View view) noexcept {
  if (!SizeMatch(view)) {
    return ViewStatus{.reason = ViewStatusReason::SizeMismatch, .size = 0u};
  }
  if (soa::detail::Overlaps(view.x, view.y)) {
    return ViewStatus{.reason = ViewStatusReason::ComponentOverlap, .size = view.x.size()};
  }
  return ViewStatus{.reason = ViewStatusReason::Ok, .size = view.x.size()};
}

inline ViewStatus Validate(const Vec2MutView view) noexcept {
  if (!SizeMatch(view)) {
    return ViewStatus{.reason = ViewStatusReason::SizeMismatch, .size = 0u};
  }
  if (soa::detail::Overlaps(view.x, view.y)) {
    return ViewStatus{.reason = ViewStatusReason::ComponentOverlap, .size = view.x.size()};
  }
  return ViewStatus{.reason = ViewStatusReason::Ok, .size = view.x.size()};
}

inline ViewStatus Validate(const Vec3View view) noexcept {
  if (!SizeMatch(view)) {
    return ViewStatus{.reason = ViewStatusReason::SizeMismatch, .size = 0u};
  }
  if (soa::detail::Overlaps(view.x, view.y) ||
      soa::detail::Overlaps(view.x, view.z) ||
      soa::detail::Overlaps(view.y, view.z)) {
    return ViewStatus{.reason = ViewStatusReason::ComponentOverlap, .size = view.x.size()};
  }
  return ViewStatus{.reason = ViewStatusReason::Ok, .size = view.x.size()};
}

inline ViewStatus Validate(const Vec3MutView view) noexcept {
  if (!SizeMatch(view)) {
    return ViewStatus{.reason = ViewStatusReason::SizeMismatch, .size = 0u};
  }
  if (soa::detail::Overlaps(view.x, view.y) ||
      soa::detail::Overlaps(view.x, view.z) ||
      soa::detail::Overlaps(view.y, view.z)) {
    return ViewStatus{.reason = ViewStatusReason::ComponentOverlap, .size = view.x.size()};
  }
  return ViewStatus{.reason = ViewStatusReason::Ok, .size = view.x.size()};
}

inline ViewStatus CanLoad(const Vec2View view, const u64 lane_base) noexcept {
  const ViewStatus status = Validate(view);
  if (!status) {
    return status;
  }
  if (lane_base > status.size || status.size - lane_base < simd::LaneCount) {
    return ViewStatus{.reason = ViewStatusReason::LaneOutOfRange, .size = status.size};
  }
  return status;
}

inline ViewStatus CanLoad(const Vec3View view, const u64 lane_base) noexcept {
  const ViewStatus status = Validate(view);
  if (!status) {
    return status;
  }
  if (lane_base > status.size || status.size - lane_base < simd::LaneCount) {
    return ViewStatus{.reason = ViewStatusReason::LaneOutOfRange, .size = status.size};
  }
  return status;
}

inline ViewStatus CanStore(const Vec2MutView view, const u64 lane_base) noexcept {
  const ViewStatus status = Validate(view);
  if (!status) {
    return status;
  }
  if (lane_base > status.size || status.size - lane_base < simd::LaneCount) {
    return ViewStatus{.reason = ViewStatusReason::LaneOutOfRange, .size = status.size};
  }
  return status;
}

inline ViewStatus CanStore(const Vec3MutView view, const u64 lane_base) noexcept {
  const ViewStatus status = Validate(view);
  if (!status) {
    return status;
  }
  if (lane_base > status.size || status.size - lane_base < simd::LaneCount) {
    return ViewStatus{.reason = ViewStatusReason::LaneOutOfRange, .size = status.size};
  }
  return status;
}

inline Vec2x Load(const Vec2View view, const u64 lane_base) noexcept {
  return Vec2x{.x = simd::LoadI32(view.x.data() + lane_base), .y = simd::LoadI32(view.y.data() + lane_base)};
}

inline Vec3x Load(const Vec3View view, const u64 lane_base) noexcept {
  return Vec3x{.x = simd::LoadI32(view.x.data() + lane_base), .y = simd::LoadI32(view.y.data() + lane_base), .z = simd::LoadI32(view.z.data() + lane_base)};
}

inline void Store(const Vec2MutView view, const u64 lane_base, const Vec2x value) noexcept {
  simd::Store(view.x.data() + lane_base, value.x);
  simd::Store(view.y.data() + lane_base, value.y);
}

inline void Store(const Vec3MutView view, const u64 lane_base, const Vec3x value) noexcept {
  simd::Store(view.x.data() + lane_base, value.x);
  simd::Store(view.y.data() + lane_base, value.y);
  simd::Store(view.z.data() + lane_base, value.z);
}

}  // namespace rund::math32::geom
