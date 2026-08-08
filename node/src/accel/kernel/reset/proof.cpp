#include "proof.hpp"

namespace rund::node::accel::detail::reset {
Spec Project(const rund::kernel::ResidentBufferRef &source,
             const Replacement *const replacement) noexcept {
  if (replacement != nullptr) {
    return Spec{
        .count = replacement->count,
        .stride = replacement->element,
        .element = replacement->element,
    };
  }
  return Spec{
      .offset = source.offset_bytes,
      .count = source.count,
      .stride = source.stride_bytes,
      .element = source.element_bytes,
  };
}

Range Prove(const Spec spec, const std::uint64_t target_bytes) noexcept {
  if (spec.count == 0u ||
      (spec.element != sizeof(std::uint32_t) &&
       spec.element != sizeof(std::uint64_t)) ||
      spec.offset % kWordBytes != 0u || spec.stride % kWordBytes != 0u ||
      spec.stride < spec.element || spec.offset > target_bytes) {
    return {};
  }

  const std::uint64_t remaining = target_bytes - spec.offset;
  if (spec.element > remaining ||
      spec.count - 1u > (remaining - spec.element) / spec.stride) {
    return {};
  }

  // The preceding quotient proof bounds both operations by target_bytes.
  const std::uint64_t end =
      spec.offset + (spec.count - 1u) * spec.stride + spec.element;
  return Range{spec, end};
}

} // namespace rund::node::accel::detail::reset
