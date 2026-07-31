#include "contract/program/compute/dsl/ops/local.hpp"

#include <string>

namespace program_compute_contract {
namespace {

using namespace dsl_support;

template <typename T> [[nodiscard]] auto BuildProjOp() {
  T ax[4]{};
  T ay[4]{};
  T az[4]{};
  T bx[4]{};
  T by[4]{};
  T bz[4]{};
  T out[4]{};
  const auto body = [&]() {
    if constexpr (sizeof(T) == sizeof(i64)) {
      return rund::compute_dsl::bind(4u)
          .fixed<1, 63>()
          .template read<"ax">(ax)
          .template read<"ay">(ay)
          .template read<"az">(az)
          .template read<"bx">(bx)
          .template read<"by">(by)
          .template read<"bz">(bz)
          .template write<"out">(out);
    } else {
      return rund::compute_dsl::bind(4u)
          .fixed<1, 31>()
          .template read<"ax">(ax)
          .template read<"ay">(ay)
          .template read<"az">(az)
          .template read<"bx">(bx)
          .template read<"by">(by)
          .template read<"bz">(bz)
          .template write<"out">(out);
    }
  }();
  return rund::compute_dsl::def("dsl-proj").on(body).map([](auto i, auto b) {
    auto ax = b.template read<"ax">();
    auto ay = b.template read<"ay">();
    auto az = b.template read<"az">();
    auto bx = b.template read<"bx">();
    auto by = b.template read<"by">();
    auto bz = b.template read<"bz">();
    auto out = b.template write<"out">();

    out[i] = rund::compute_dsl::unit(rund::compute_dsl::Axis::X, ax[i], ay[i]) +
             rund::compute_dsl::unit(rund::compute_dsl::Axis::Y, ax[i], ay[i]) +
             rund::compute_dsl::unit(rund::compute_dsl::Axis::X, ax[i], ay[i], az[i]) +
             rund::compute_dsl::unit(rund::compute_dsl::Axis::Y, ax[i], ay[i], az[i]) +
             rund::compute_dsl::unit(rund::compute_dsl::Axis::Z, ax[i], ay[i], az[i]) +
             rund::compute_dsl::proj(rund::compute_dsl::Axis::X, ax[i], ay[i], bx[i],
                               by[i]) +
             rund::compute_dsl::proj(rund::compute_dsl::Axis::Y, ax[i], ay[i], bx[i],
                               by[i]) +
             rund::compute_dsl::proj(rund::compute_dsl::Axis::X, ax[i], ay[i], az[i],
                               bx[i], by[i], bz[i]) +
             rund::compute_dsl::proj(rund::compute_dsl::Axis::Y, ax[i], ay[i], az[i],
                               bx[i], by[i], bz[i]) +
             rund::compute_dsl::proj(rund::compute_dsl::Axis::Z, ax[i], ay[i], az[i],
                               bx[i], by[i], bz[i]);
  });
}

[[nodiscard]] bool HasProjOps(const std::string &source) {
  return source.find("].op=sqrt") != std::string_view::npos &&
         source.find("].op=div_fixed") != std::string_view::npos &&
         source.find("].op=eq") != std::string_view::npos &&
         source.find("].op=select") != std::string_view::npos &&
         source.find("].op=mul_fixed") != std::string_view::npos &&
         source.find("].op=add_sat") != std::string_view::npos;
}

int test_compute_proj_helpers_build_deterministic_lowerable_ir() {
  const auto first32 = BuildProjOp<i32>();
  const auto second32 = BuildProjOp<i32>();
  const auto fixed_lane64 = BuildProjOp<i64>();

  TEST_ASSERT(first32.ok());
  TEST_ASSERT(second32.ok());
  TEST_ASSERT(fixed_lane64.ok());
  TEST_ASSERT(first32.ir().canonical_bytes == second32.ir().canonical_bytes);
  TEST_ASSERT(first32.ir().op_hash_hi == second32.ir().op_hash_hi);
  TEST_ASSERT(first32.ir().op_hash_lo == second32.ir().op_hash_lo);

  const rund::kernel::LoweringArtifact metal32 =
      rund::kernel::LowerComputeIR(first32.ir(),
                                   rund::kernel::ComputeApi::Metal);
  const rund::kernel::LoweringArtifact vulkan64 =
      rund::kernel::LowerComputeIR(fixed_lane64.ir(),
                                   rund::kernel::ComputeApi::Vulkan);
  TEST_ASSERT(metal32.ok);
  TEST_ASSERT(vulkan64.ok);
  TEST_ASSERT(HasProjOps(metal32.source_text));
  TEST_ASSERT(HasProjOps(vulkan64.source_text));
  return 0;
}

} // namespace

int RunComputeDslOpsProjContract() {
  return test_compute_proj_helpers_build_deterministic_lowerable_ir();
}

} // namespace program_compute_contract
