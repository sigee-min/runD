#pragma once

#include <kernel/program/compute/dsl.hpp>

#include <string_view>
#include <vector>

namespace program_compute_contract::rejection_support {

using Mode = rund::compute_dsl::detail::ScalarMode;

inline constexpr rund::kernel::ComputeFixedFormat kFixed16x16{
    .integer_bits = 16u,
    .fraction_bits = 16u,
    .rounding = rund::kernel::ComputeRounding::NearestEven,
    .overflow = rund::kernel::ComputeOverflow::Saturate,
    .approximation = rund::kernel::ComputeApproximation::Exact,
};

inline constexpr rund::kernel::ComputeFixedFormat kFixed20x44{
    .integer_bits = 20u,
    .fraction_bits = 44u,
    .rounding = rund::kernel::ComputeRounding::NearestEven,
    .overflow = rund::kernel::ComputeOverflow::Saturate,
    .approximation = rund::kernel::ComputeApproximation::Exact,
};

template <Mode Header> struct Body final {
  [[nodiscard]] static constexpr Mode scalar_mode() noexcept { return Header; }

  [[nodiscard]] const std::vector<rund::compute_dsl::detail::BindingRuntime> &
  bindings() const noexcept {
    return values;
  }

  [[nodiscard]] constexpr rund::kernel::u64 tile_count() const noexcept {
    return 1u;
  }

  [[nodiscard]] static constexpr rund::kernel::ComputeFixedFormat
  fixed_format() noexcept {
    if constexpr (Header == Mode::FixedLane32) {
      return kFixed16x16;
    }
    if constexpr (Header == Mode::FixedLane64) {
      return kFixed20x44;
    }
    return {};
  }

  [[nodiscard]] constexpr bool ok() const noexcept { return true; }
  [[nodiscard]] constexpr const char *reason() const noexcept { return "ok"; }

  std::vector<rund::compute_dsl::detail::BindingRuntime> values;
};

[[nodiscard]] inline std::vector<rund::compute_dsl::detail::BindingRuntime>
IntegerBindings(const Mode mode) {
  using namespace rund::compute_dsl::detail;
  return {BindingRuntime{
      .kind = BindingKind::Read,
      .numeric_mode = mode,
      .name = "input",
      .element_bytes = WideMode(mode) ? 8u : 4u,
  }};
}

[[nodiscard]] bool Accepts(const rund::kernel::ComputeIR &ir);
[[nodiscard]] bool Rejects(const rund::kernel::ComputeIR &ir,
                           std::string_view reason);

} // namespace program_compute_contract::rejection_support
