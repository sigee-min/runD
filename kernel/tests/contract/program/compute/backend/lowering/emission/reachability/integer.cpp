#include "contract/program/compute/backend/lowering/local.hpp"
#include "local.hpp"
#include "test/assert.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace program_compute_contract {
namespace {

using namespace backend_lowering_support;

template <typename T>
[[nodiscard]] auto SignedIntegerSaturationBody(T (&lhs)[1], T (&rhs)[1],
                                               T (&add)[1], T (&sub)[1]) {
  if constexpr (sizeof(T) == sizeof(i64)) {
    return rund::compute_dsl::bind(1u)
        .i64()
        .template read<"lhs">(lhs)
        .template read<"rhs">(rhs)
        .template write<"add">(add)
        .template write<"sub">(sub);
  } else {
    return rund::compute_dsl::bind(1u)
        .i32()
        .template read<"lhs">(lhs)
        .template read<"rhs">(rhs)
        .template write<"add">(add)
        .template write<"sub">(sub);
  }
}

template <typename T> [[nodiscard]] auto BuildSignedIntegerSaturationOp() {
  T lhs[1]{};
  T rhs[1]{};
  T add[1]{};
  T sub[1]{};
  return rund::compute_dsl::def(sizeof(T) == sizeof(i64)
                                    ? "helper-emission-i64-saturation"
                                    : "helper-emission-i32-saturation")
      .on(SignedIntegerSaturationBody(lhs, rhs, add, sub))
      .map([](auto i, auto b) {
        const auto lhs = b.template read<"lhs">();
        const auto rhs = b.template read<"rhs">();
        const auto add = b.template write<"add">();
        const auto sub = b.template write<"sub">();
        add[i] = rund::compute_dsl::add_sat(lhs[i], rhs[i]);
        sub[i] = rund::compute_dsl::sub_sat(lhs[i], rhs[i]);
      });
}

template <typename T>
[[nodiscard]] auto UnsignedIntegerSaturationBody(T (&lhs)[1], T (&rhs)[1],
                                                 T (&add)[1]) {
  if constexpr (sizeof(T) == sizeof(rund::kernel::u64)) {
    return rund::compute_dsl::bind(1u)
        .u64()
        .template read<"lhs">(lhs)
        .template read<"rhs">(rhs)
        .template write<"add">(add);
  } else {
    return rund::compute_dsl::bind(1u)
        .u32()
        .template read<"lhs">(lhs)
        .template read<"rhs">(rhs)
        .template write<"add">(add);
  }
}

template <typename T> [[nodiscard]] auto BuildUnsignedIntegerSaturationOp() {
  T lhs[1]{};
  T rhs[1]{};
  T add[1]{};
  return rund::compute_dsl::def(sizeof(T) == sizeof(rund::kernel::u64)
                                    ? "helper-emission-u64-saturation"
                                    : "helper-emission-u32-saturation")
      .on(UnsignedIntegerSaturationBody(lhs, rhs, add))
      .map([](auto i, auto b) {
        const auto lhs = b.template read<"lhs">();
        const auto rhs = b.template read<"rhs">();
        const auto add = b.template write<"add">();
        add[i] = rund::compute_dsl::add_sat_unsigned(lhs[i], rhs[i]);
      });
}

template <rund::compute_dsl::detail::ScalarMode HeaderMode>
struct MixedIntegerBody final {
  [[nodiscard]] static constexpr rund::compute_dsl::detail::ScalarMode
  scalar_mode() noexcept {
    return HeaderMode;
  }
  [[nodiscard]] const std::vector<rund::compute_dsl::detail::BindingRuntime> &
  bindings() const noexcept {
    return values;
  }
  [[nodiscard]] constexpr rund::kernel::u64 tile_count() const noexcept {
    return 1u;
  }
  [[nodiscard]] constexpr rund::kernel::ComputeFixedFormat
  fixed_format() const noexcept {
    return {};
  }
  [[nodiscard]] constexpr bool ok() const noexcept { return true; }
  [[nodiscard]] constexpr const char *reason() const noexcept { return "ok"; }

  std::vector<rund::compute_dsl::detail::BindingRuntime> values;
};

template <rund::compute_dsl::detail::ScalarMode HeaderMode,
          rund::compute_dsl::detail::ScalarMode SecondaryMode, bool Divide>
[[nodiscard]] rund::kernel::ComputeIR BuildMixedIntegerIr() {
  using namespace rund::compute_dsl::detail;
  constexpr rund::kernel::u32 bytes = WideMode(HeaderMode) ? 8u : 4u;
  static_assert(WideMode(HeaderMode) == WideMode(SecondaryMode));
  constexpr bool header_signed =
      HeaderMode == ScalarMode::I32 || HeaderMode == ScalarMode::I64;
  constexpr bool secondary_signed =
      SecondaryMode == ScalarMode::I32 || SecondaryMode == ScalarMode::I64;
  static_assert(header_signed != secondary_signed);
  MixedIntegerBody<HeaderMode> body{.values = {
                                        BindingRuntime{
                                            .kind = BindingKind::Read,
                                            .numeric_mode = HeaderMode,
                                            .name = "header",
                                            .element_bytes = bytes,
                                        },
                                        BindingRuntime{
                                            .kind = BindingKind::Read,
                                            .numeric_mode = SecondaryMode,
                                            .name = "secondary",
                                            .element_bytes = bytes,
                                        },
                                        BindingRuntime{
                                            .kind = BindingKind::Write,
                                            .numeric_mode = HeaderMode,
                                            .name = "header_output",
                                            .element_bytes = bytes,
                                        },
                                        BindingRuntime{
                                            .kind = BindingKind::Write,
                                            .numeric_mode = SecondaryMode,
                                            .name = "secondary_output",
                                            .element_bytes = bytes,
                                        },
                                    }};
  BuildContext context{body.bindings(), HeaderMode};
  const auto header = DynamicRead(context, 0u);
  const auto secondary = DynamicRead(context, 1u);
  if constexpr (Divide) {
    DynamicWrite(context, 2u, header / 2u);
    DynamicWrite(context, 3u, secondary / 2u);
  } else {
    DynamicWrite(context, 2u, header);
    DynamicWrite(context, 3u, secondary);
  }
  return BuildIr("", body, context);
}

[[nodiscard]] std::string_view
IntegerDivideDefinition(const rund::kernel::ComputeApi api, const bool wide,
                        const bool signed_divide) {
  if (api == rund::kernel::ComputeApi::Metal) {
    if (wide) {
      return signed_divide ? "inline long RundDivSigned64("
                           : "inline long RundDivUnsigned64(";
    }
    return signed_divide ? "inline int RundDivSigned32("
                         : "inline int RundDivUnsigned32(";
  }
  if (api == rund::kernel::ComputeApi::Vulkan) {
    if (wide) {
      return signed_divide ? "uint64_t RundDivSigned64("
                           : "uint64_t RundDivUnsigned64(";
    }
    return signed_divide ? "uint RundDivSigned32(" : "uint RundDivUnsigned32(";
  }
  if (wide) {
    return signed_divide ? "fn RundDivSigned64(" : "fn RundDivUnsigned64(";
  }
  return signed_divide ? "fn RundDivSigned32(" : "fn RundDivUnsigned32(";
}

int test_integer_division_helpers_match_reachable_ops() {
  using Mode = rund::compute_dsl::detail::ScalarMode;
  const std::array divide_irs{
      BuildMixedIntegerIr<Mode::I32, Mode::U32, true>(),
      BuildMixedIntegerIr<Mode::U32, Mode::I32, true>(),
      BuildMixedIntegerIr<Mode::I64, Mode::U64, true>(),
      BuildMixedIntegerIr<Mode::U64, Mode::I64, true>(),
  };
  for (const auto api :
       {rund::kernel::ComputeApi::Metal, rund::kernel::ComputeApi::Vulkan}) {
    for (std::size_t index = 0u; index < divide_irs.size(); ++index) {
      const bool wide = index >= 2u;
      const auto artifact =
          rund::kernel::LowerComputeIR(divide_irs[index], api);
      TEST_ASSERT(artifact.ok);
      TEST_ASSERT(artifact.key.domain ==
                  (index == 0u   ? rund::kernel::ComputeDomain::I32
                   : index == 1u ? rund::kernel::ComputeDomain::U32
                   : index == 2u ? rund::kernel::ComputeDomain::I64
                                 : rund::kernel::ComputeDomain::U64));
      TEST_ASSERT(artifact.source_text.find("].op=div_signed") !=
                  std::string::npos);
      TEST_ASSERT(artifact.source_text.find("].op=div_unsigned") !=
                  std::string::npos);
      TEST_ASSERT(artifact.source_text.find(IntegerDivideDefinition(
                      api, wide, true)) != std::string::npos);
      TEST_ASSERT(artifact.source_text.find(IntegerDivideDefinition(
                      api, wide, false)) != std::string::npos);
      TEST_ASSERT(CountOccurrences(artifact.source_text,
                                   wide ? "RundDivSigned64("
                                        : "RundDivSigned32(") >= 2u);
      TEST_ASSERT(CountOccurrences(artifact.source_text,
                                   wide ? "RundDivUnsigned64("
                                        : "RundDivUnsigned32(") >= 2u);
    }
  }
  return 0;
}

int test_integer_saturation_sources_define_every_called_helper() {
  const auto signed32 = BuildSignedIntegerSaturationOp<i32>();
  const auto unsigned32 = BuildUnsignedIntegerSaturationOp<rund::kernel::u32>();
  const auto signed64 = BuildSignedIntegerSaturationOp<i64>();
  const auto unsigned64 = BuildUnsignedIntegerSaturationOp<rund::kernel::u64>();
  const auto check_signed = [](const auto &op, const std::string_view add,
                               const std::string_view sub) {
    for (const auto api :
         {rund::kernel::ComputeApi::Metal, rund::kernel::ComputeApi::Vulkan}) {
      const auto artifact = rund::kernel::LowerComputeIR(op.ir(), api);
      TEST_ASSERT(artifact.ok);
      TEST_ASSERT(CountOccurrences(artifact.source_text, add) >= 2u);
      TEST_ASSERT(CountOccurrences(artifact.source_text, sub) >= 2u);
    }
  };
  const auto check_unsigned = [](const auto &op, const std::string_view add) {
    for (const auto api :
         {rund::kernel::ComputeApi::Metal, rund::kernel::ComputeApi::Vulkan}) {
      const auto artifact = rund::kernel::LowerComputeIR(op.ir(), api);
      TEST_ASSERT(artifact.ok);
      TEST_ASSERT(CountOccurrences(artifact.source_text, add) >= 2u);
    }
  };
  check_signed(signed32, "RundAddSat32", "RundSubSat32");
  check_unsigned(unsigned32, "RundAddSatUnsigned32");
  check_signed(signed64, "RundAddSat64", "RundSubSat64");
  check_unsigned(unsigned64, "RundAddSatUnsigned64");
  return 0;
}

} // namespace

namespace helper_emission {

int CheckIntegerHelpers() {
  if (test_integer_division_helpers_match_reachable_ops() != 0) {
    return 1;
  }
  return test_integer_saturation_sources_define_every_called_helper();
}

} // namespace helper_emission
} // namespace program_compute_contract
