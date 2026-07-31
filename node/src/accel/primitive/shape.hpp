#pragma once

#include <kernel/core/checked.hpp>
#include <kernel/program/compute/binding/model.hpp>

namespace rund::node::accel::detail {

[[nodiscard]] inline bool
PrimitiveResidentShapeOk(const rund::kernel::ResidentBufferRef &ref,
                         const rund::kernel::u64 element_bytes,
                         const rund::kernel::u64 required_count,
                         const rund::kernel::u32 usage,
                         const bool exact_count) noexcept {
  if (ref.element_bytes != element_bytes || ref.stride_bytes != element_bytes ||
      ref.usage != usage ||
      (exact_count ? ref.count != required_count
                   : ref.count < required_count) ||
      !rund::kernel::checked::mul(ref.count, element_bytes)) {
    return false;
  }
  return ref.bytes >= ref.count * element_bytes;
}

[[nodiscard]] inline bool
PrimitiveResidentExactShapeOk(const rund::kernel::ResidentBufferRef &ref,
                              const rund::kernel::u64 element_bytes,
                              const rund::kernel::u64 required_count,
                              const rund::kernel::u32 usage) noexcept {
  return PrimitiveResidentShapeOk(ref, element_bytes, required_count, usage,
                                  true);
}

[[nodiscard]] inline bool
PrimitiveResidentExactShapeOk(const rund::kernel::ResidentBufferRef *const ref,
                              const rund::kernel::u64 element_bytes,
                              const rund::kernel::u64 required_count,
                              const rund::kernel::u32 usage) noexcept {
  return ref != nullptr && PrimitiveResidentExactShapeOk(*ref, element_bytes,
                                                         required_count, usage);
}

[[nodiscard]] inline bool
PrimitiveResidentAtLeastShapeOk(const rund::kernel::ResidentBufferRef &ref,
                                const rund::kernel::u64 element_bytes,
                                const rund::kernel::u64 required_count,
                                const rund::kernel::u32 usage) noexcept {
  return PrimitiveResidentShapeOk(ref, element_bytes, required_count, usage,
                                  false);
}

[[nodiscard]] inline bool PrimitiveResidentAtLeastShapeOk(
    const rund::kernel::ResidentBufferRef *const ref,
    const rund::kernel::u64 element_bytes,
    const rund::kernel::u64 required_count,
    const rund::kernel::u32 usage) noexcept {
  return ref != nullptr && PrimitiveResidentAtLeastShapeOk(
                               *ref, element_bytes, required_count, usage);
}

[[nodiscard]] inline bool
ResidentSpan(const rund::kernel::ResidentBufferRef &ref, std::uint64_t &begin,
             std::uint64_t &end) noexcept {
  if (ref.id == 0u || ref.element_bytes == 0u || ref.count == 0u ||
      ref.stride_bytes < ref.element_bytes ||
      !rund::kernel::checked::mul(ref.count - 1u, ref.stride_bytes)) {
    return false;
  }
  const std::uint64_t tail = (ref.count - 1u) * ref.stride_bytes;
  if (!rund::kernel::checked::add(ref.offset_bytes, tail) ||
      !rund::kernel::checked::add(ref.offset_bytes + tail, ref.element_bytes)) {
    return false;
  }
  begin = ref.offset_bytes;
  end = ref.offset_bytes + tail + ref.element_bytes;
  return end <= ref.bytes;
}

[[nodiscard]] inline bool
ResidentOverlap(const rund::kernel::ResidentBufferRef &left,
                const rund::kernel::ResidentBufferRef &right) noexcept {
  if (left.id != right.id) {
    return false;
  }
  std::uint64_t left_begin = 0u;
  std::uint64_t left_end = 0u;
  std::uint64_t right_begin = 0u;
  std::uint64_t right_end = 0u;
  if (!ResidentSpan(left, left_begin, left_end) ||
      !ResidentSpan(right, right_begin, right_end)) {
    return true;
  }
  return left_begin < right_end && right_begin < left_end;
}

} // namespace rund::node::accel::detail
