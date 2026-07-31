#pragma once

template <class S>
[[nodiscard]] inline rund::kernel::StencilResult ExecuteSignedStencilReference(
    const rund::kernel::StencilOp op, const S *const input, S *const output,
    const rund::kernel::u64 element_count, const rund::kernel::u64 radius) {
  static_assert(std::is_signed_v<S>);
  if (input == nullptr || output == nullptr) {
    return rund::kernel::StencilResult{.reason =
                                           "compute_stencil_buffer_invalid"};
  }
  using U = std::make_unsigned_t<S>;
  for (rund::kernel::u64 index = 0; index < element_count; ++index) {
    S value = input[index];
    U sum = std::bit_cast<U>(value);
    for (rund::kernel::u64 distance = 1; distance <= radius; ++distance) {
      const rund::kernel::u64 left = index < distance ? 0 : index - distance;
      const rund::kernel::u64 right =
          std::min(index + distance, element_count - 1u);
      if (op == rund::kernel::StencilOp::Min) {
        value = std::min(value, std::min(input[left], input[right]));
      } else if (op == rund::kernel::StencilOp::Max) {
        value = std::max(value, std::max(input[left], input[right]));
      } else {
        sum += std::bit_cast<U>(input[left]);
        sum += std::bit_cast<U>(input[right]);
      }
    }
    output[index] =
        op == rund::kernel::StencilOp::Sum ? std::bit_cast<S>(sum) : value;
  }
  return rund::kernel::StencilResult{
      .element_count = element_count, .ok = true, .reason = "ok"};
}
