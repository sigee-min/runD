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

int test_fixed_helpers_follow_actual_ops_and_selected_lane() {
  const auto fixed_arithmetic_32 = BuildFixedLane32ArithmeticOps();
  const auto fixed_arithmetic_64 = BuildFixedLane64ArithmeticOps();
  const auto sin32 = BuildFixedSinOp<i32>();
  const auto tan32 = BuildFixedTanOp<i32>();
  const auto atan64 = BuildFixedAtan2Op<i64>();
  for (const auto api :
       {rund::kernel::ComputeApi::Metal, rund::kernel::ComputeApi::Vulkan}) {
    const auto stored32 =
        rund::kernel::LowerComputeIR(fixed_arithmetic_32.ir(), api);
    const auto stored64 =
        rund::kernel::LowerComputeIR(fixed_arithmetic_64.ir(), api);
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

namespace helper_emission {

int CheckFixedHelperSelection() {
  return test_fixed_helpers_follow_actual_ops_and_selected_lane();
}

int CheckFixedHelperSourceSize() {
  return test_fixed_helper_source_size_tracks_required_library();
}

} // namespace helper_emission
} // namespace program_compute_contract
