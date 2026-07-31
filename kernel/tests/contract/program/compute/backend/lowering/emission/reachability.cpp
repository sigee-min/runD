#include "contract/program/compute/backend/lowering/local.hpp"
#include "test/assert.hpp"

#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace program_compute_contract {
namespace {

using namespace backend_lowering_support;

template <typename T>
[[nodiscard]] auto FixedUnaryBody(T (&input)[4], T (&output)[4]) {
  if constexpr (sizeof(T) == sizeof(i64)) {
    return rund::compute_dsl::bind(4u)
        .fixed<1, 63>()
        .template read<"input">(input)
        .template write<"output">(output);
  } else {
    return rund::compute_dsl::bind(4u)
        .fixed<1, 31>()
        .template read<"input">(input)
        .template write<"output">(output);
  }
}

template <typename T> [[nodiscard]] auto BuildFixedSinOp() {
  T input[4]{};
  T output[4]{};
  return rund::compute_dsl::def("helper-emission-sin")
      .on(FixedUnaryBody(input, output))
      .map([](auto i, auto b) {
        const auto input = b.template read<"input">();
        const auto output = b.template write<"output">();
        output[i] = rund::compute_dsl::sin(input[i]);
      });
}

template <typename T> [[nodiscard]] auto BuildFixedTanOp() {
  T input[4]{};
  T output[4]{};
  return rund::compute_dsl::def("helper-emission-tan")
      .on(FixedUnaryBody(input, output))
      .map([](auto i, auto b) {
        const auto input = b.template read<"input">();
        const auto output = b.template write<"output">();
        output[i] = rund::compute_dsl::tan(input[i]);
      });
}

template <typename T> [[nodiscard]] auto BuildFixedAtan2Op() {
  T real[4]{};
  T imag[4]{};
  T output[4]{};
  const auto body = [&]() {
    if constexpr (sizeof(T) == sizeof(i64)) {
      return rund::compute_dsl::bind(4u)
          .fixed<1, 63>()
          .template read<"real">(real)
          .template read<"imag">(imag)
          .template write<"output">(output);
    } else {
      return rund::compute_dsl::bind(4u)
          .fixed<1, 31>()
          .template read<"real">(real)
          .template read<"imag">(imag)
          .template write<"output">(output);
    }
  }();
  return rund::compute_dsl::def("helper-emission-atan2")
      .on(body)
      .map([](auto i, auto b) {
        const auto real = b.template read<"real">();
        const auto imag = b.template read<"imag">();
        const auto output = b.template write<"output">();
        output[i] = rund::compute_dsl::complex(
            rund::compute_dsl::ComplexOp::Phase, real[i], imag[i]);
      });
}

template <typename T> [[nodiscard]] auto BuildFixedAllCanonicalOps() {
  T lhs[4]{};
  T rhs[4]{};
  T output[4]{};
  const auto body = [&]() {
    if constexpr (sizeof(T) == sizeof(i64)) {
      return rund::compute_dsl::bind(4u)
          .fixed<1, 63>()
          .template read<"lhs">(lhs)
          .template read<"rhs">(rhs)
          .template write<"output">(output);
    } else {
      return rund::compute_dsl::bind(4u)
          .fixed<1, 31>()
          .template read<"lhs">(lhs)
          .template read<"rhs">(rhs)
          .template write<"output">(output);
    }
  }();
  return rund::compute_dsl::def("helper-emission-all-canonical")
      .on(body)
      .map([](auto i, auto b) {
        const auto lhs = b.template read<"lhs">();
        const auto rhs = b.template read<"rhs">();
        const auto output = b.template write<"output">();
        output[i] =
            rund::compute_dsl::sin(lhs[i]) + rund::compute_dsl::cos(lhs[i]) +
            rund::compute_dsl::tan(lhs[i]) + rund::compute_dsl::exp(lhs[i]) +
            rund::compute_dsl::log(lhs[i]) +
            rund::compute_dsl::complex(rund::compute_dsl::ComplexOp::Phase,
                                       lhs[i], rhs[i]);
      });
}

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

using NativeU128 = rund::kernel::u128;

struct OracleU128 {
  std::uint64_t hi;
  std::uint64_t lo;
};

[[nodiscard]] constexpr OracleU128 PairOf(const NativeU128 value) noexcept {
  return OracleU128{.hi = static_cast<std::uint64_t>(value >> 64u),
                    .lo = static_cast<std::uint64_t>(value)};
}

[[nodiscard]] constexpr bool OracleGe(const OracleU128 lhs,
                                      const OracleU128 rhs) noexcept {
  return lhs.hi > rhs.hi || (lhs.hi == rhs.hi && lhs.lo >= rhs.lo);
}

[[nodiscard]] constexpr OracleU128 OracleSub(const OracleU128 lhs,
                                             const OracleU128 rhs) noexcept {
  const std::uint64_t borrow = lhs.lo < rhs.lo ? 1u : 0u;
  return OracleU128{.hi = lhs.hi - rhs.hi - borrow, .lo = lhs.lo - rhs.lo};
}

[[nodiscard]] constexpr OracleU128 OracleShl1Or(const OracleU128 value,
                                                const std::uint64_t bit) {
  return OracleU128{.hi = (value.hi << 1u) | (value.lo >> 63u),
                    .lo = (value.lo << 1u) | bit};
}

[[nodiscard]] constexpr std::uint64_t
RestoringDivU128ByU64(const OracleU128 numerator,
                      const std::uint64_t denominator) noexcept {
  constexpr std::uint64_t max = std::numeric_limits<std::uint64_t>::max();
  if (denominator == 0u) {
    return max;
  }
  const OracleU128 denominator_wide{.hi = 0u, .lo = denominator};
  OracleU128 remainder{};
  std::uint64_t quotient = 0u;
  bool overflow = false;
  for (int bit = 127; bit >= 0; --bit) {
    const std::uint64_t input_bit =
        bit >= 64 ? ((numerator.hi >> static_cast<unsigned>(bit - 64)) & 1u)
                  : ((numerator.lo >> static_cast<unsigned>(bit)) & 1u);
    remainder = OracleShl1Or(remainder, input_bit);
    if (OracleGe(remainder, denominator_wide)) {
      remainder = OracleSub(remainder, denominator_wide);
      if (bit >= 64) {
        overflow = true;
      } else {
        quotient |= std::uint64_t{1u} << static_cast<unsigned>(bit);
      }
    }
  }
  return overflow ? max : quotient;
}

[[nodiscard]] constexpr std::uint64_t
NativeDivU128ByU64(const OracleU128 numerator,
                   const std::uint64_t denominator) noexcept {
  constexpr std::uint64_t max = std::numeric_limits<std::uint64_t>::max();
  if (denominator == 0u) {
    return max;
  }
  const NativeU128 value =
      (static_cast<NativeU128>(numerator.hi) << 64u) | numerator.lo;
  const NativeU128 quotient = value / denominator;
  return quotient > max ? max : static_cast<std::uint64_t>(quotient);
}

int test_generic_u128_division_preserves_top_bit_and_carry() {
  constexpr std::uint64_t max = std::numeric_limits<std::uint64_t>::max();
  constexpr std::uint64_t high_denominator = max;
  constexpr std::uint64_t high_quotient = 0x8000000000000001ull;
  constexpr NativeU128 high_product =
      static_cast<NativeU128>(high_denominator) * high_quotient +
      high_denominator / 2u;
  struct DivisionCase {
    OracleU128 numerator;
    std::uint64_t denominator;
  };
  constexpr std::array<DivisionCase, 6u> cases{
      DivisionCase{.numerator = PairOf(0u), .denominator = 0u},
      DivisionCase{.numerator = PairOf(max), .denominator = 1u},
      DivisionCase{.numerator = PairOf(NativeU128{1u} << 127u),
                   .denominator = max},
      DivisionCase{.numerator = PairOf(high_product),
                   .denominator = high_denominator},
      DivisionCase{.numerator = PairOf(~NativeU128{0u}), .denominator = max},
      DivisionCase{.numerator = PairOf(~NativeU128{0u}), .denominator = 1u},
  };
  TEST_ASSERT((cases[3].numerator.hi & (std::uint64_t{1u} << 63u)) != 0u);
  for (const DivisionCase &value : cases) {
    TEST_ASSERT(RestoringDivU128ByU64(value.numerator, value.denominator) ==
                NativeDivU128ByU64(value.numerator, value.denominator));
  }
  TEST_ASSERT(RestoringDivU128ByU64(cases[3].numerator, cases[3].denominator) ==
              high_quotient);

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

int test_fixed_helpers_follow_actual_ops_and_selected_lane() {
  const auto fixed_arithmetic_32 = BuildFixedLane32ArithmeticOps();
  const auto fixed_arithmetic_64 = BuildFixedLane64ArithmeticOps();
  const auto sin32 = BuildFixedSinOp<i32>();
  const auto tan32 = BuildFixedTanOp<i32>();
  const auto atan64 = BuildFixedAtan2Op<i64>();
  for (const auto api :
       {rund::kernel::ComputeApi::Metal, rund::kernel::ComputeApi::Vulkan}) {
    const auto stored32 = rund::kernel::LowerComputeIR(fixed_arithmetic_32.ir(), api);
    const auto stored64 = rund::kernel::LowerComputeIR(fixed_arithmetic_64.ir(), api);
    const auto selected_sin = rund::kernel::LowerComputeIR(sin32.ir(), api);
    const auto selected_tan = rund::kernel::LowerComputeIR(tan32.ir(), api);
    const auto selected_atan = rund::kernel::LowerComputeIR(atan64.ir(), api);
    TEST_ASSERT(stored32.ok);
    TEST_ASSERT(stored64.ok);
    TEST_ASSERT(selected_sin.ok);
    TEST_ASSERT(selected_tan.ok);
    TEST_ASSERT(selected_atan.ok);

    TEST_ASSERT(stored32.source_text.find("RundAddSat32") != std::string::npos);
    TEST_ASSERT(stored32.source_text.find("RundAddSatUnsigned32") !=
                std::string::npos);
    TEST_ASSERT(stored32.source_text.find("RundSubSat32") != std::string::npos);
    TEST_ASSERT(stored32.source_text.find("RundNegPositiveFixedLane32") !=
                std::string::npos);

    TEST_ASSERT(stored64.source_text.find("RundAddSat64") != std::string::npos);

    TEST_ASSERT(selected_sin.source_text.find("RundSin32") !=
                std::string::npos);
    TEST_ASSERT(selected_sin.source_text.find("RundMulFixedLane32") !=
                std::string::npos);

    TEST_ASSERT(selected_tan.source_text.find("RundSin32") !=
                std::string::npos);
    TEST_ASSERT(selected_tan.source_text.find("RundCos32") !=
                std::string::npos);

    TEST_ASSERT(selected_atan.source_text.find("RundAtan264") !=
                std::string::npos);
  }
  return 0;
}

int test_fixed_helper_source_size_tracks_required_library() {
  const auto minimal32 = BuildFixedLane32Op(7);
  const auto minimal64 = BuildFixedLane64Op(7);
  const auto stored32 = BuildFixedLane32ArithmeticOps();
  const auto stored64 = BuildFixedLane64ArithmeticOps();
  const auto full32 = BuildFixedAllCanonicalOps<i32>();
  const auto full64 = BuildFixedAllCanonicalOps<i64>();
  for (const auto api :
       {rund::kernel::ComputeApi::Metal, rund::kernel::ComputeApi::Vulkan}) {
    const auto minimal_artifact32 =
        rund::kernel::LowerComputeIR(minimal32.ir(), api);
    const auto minimal_artifact64 =
        rund::kernel::LowerComputeIR(minimal64.ir(), api);
    const auto stored_artifact32 =
        rund::kernel::LowerComputeIR(stored32.ir(), api);
    const auto stored_artifact64 =
        rund::kernel::LowerComputeIR(stored64.ir(), api);
    const auto full_artifact32 = rund::kernel::LowerComputeIR(full32.ir(), api);
    const auto full_artifact64 = rund::kernel::LowerComputeIR(full64.ir(), api);
    TEST_ASSERT(minimal_artifact32.ok);
    TEST_ASSERT(minimal_artifact64.ok);
    TEST_ASSERT(stored_artifact32.ok);
    TEST_ASSERT(stored_artifact64.ok);
    TEST_ASSERT(full_artifact32.ok);
    TEST_ASSERT(full_artifact64.ok);
    TEST_ASSERT(minimal_artifact32.source_text.size() <
                stored_artifact32.source_text.size());
    TEST_ASSERT(stored_artifact32.source_text.size() <
                full_artifact32.source_text.size());
    TEST_ASSERT(minimal_artifact64.source_text.size() <
                stored_artifact64.source_text.size());
    TEST_ASSERT(stored_artifact64.source_text.size() <
                full_artifact64.source_text.size());
  }
  return 0;
}

} // namespace

int RunComputeBackendHelperEmissionContract() {
  if (test_integer_division_helpers_match_reachable_ops() != 0) {
    return 1;
  }
  if (test_integer_saturation_sources_define_every_called_helper() != 0) {
    return 1;
  }
  if (test_fixed_helpers_follow_actual_ops_and_selected_lane() != 0) {
    return 1;
  }
  if (test_generic_u128_division_preserves_top_bit_and_carry() != 0) {
    return 1;
  }
  return test_fixed_helper_source_size_tracks_required_library();
}

} // namespace program_compute_contract
