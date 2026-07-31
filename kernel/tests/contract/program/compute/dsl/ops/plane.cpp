#include "contract/program/compute/dsl/ops/local.hpp"

#include <string>

namespace program_compute_contract {
namespace {

using namespace dsl_support;

template <typename T> [[nodiscard]] auto BuildPlaneOp() {
  T px[4]{};
  T py[4]{};
  T pz[4]{};
  T ax[4]{};
  T ay[4]{};
  T az[4]{};
  T nx[4]{};
  T ny[4]{};
  T nz[4]{};
  T out[4]{};
  const auto body = [&]() {
    if constexpr (sizeof(T) == sizeof(i64)) {
      return rund::compute_dsl::bind(4u)
          .fixed<1, 63>()
          .template read<"px">(px)
          .template read<"py">(py)
          .template read<"pz">(pz)
          .template read<"ax">(ax)
          .template read<"ay">(ay)
          .template read<"az">(az)
          .template read<"nx">(nx)
          .template read<"ny">(ny)
          .template read<"nz">(nz)
          .template write<"out">(out);
    } else {
      return rund::compute_dsl::bind(4u)
          .fixed<1, 31>()
          .template read<"px">(px)
          .template read<"py">(py)
          .template read<"pz">(pz)
          .template read<"ax">(ax)
          .template read<"ay">(ay)
          .template read<"az">(az)
          .template read<"nx">(nx)
          .template read<"ny">(ny)
          .template read<"nz">(nz)
          .template write<"out">(out);
    }
  }();
  return rund::compute_dsl::def("dsl-plane").on(body).map([](auto i, auto b) {
    auto px = b.template read<"px">();
    auto py = b.template read<"py">();
    auto pz = b.template read<"pz">();
    auto ax = b.template read<"ax">();
    auto ay = b.template read<"ay">();
    auto az = b.template read<"az">();
    auto nx = b.template read<"nx">();
    auto ny = b.template read<"ny">();
    auto nz = b.template read<"nz">();
    auto out = b.template write<"out">();
    using Axis = rund::compute_dsl::Axis;
    using MetricOp = rund::compute_dsl::MetricOp;
    using PlaneOp = rund::compute_dsl::PlaneOp;

    out[i] = rund::compute_dsl::plane(PlaneOp::Parameter, px[i], py[i], pz[i],
                                ax[i], ay[i], az[i], nx[i], ny[i], nz[i]) +
             rund::compute_dsl::plane(PlaneOp::Projection, Axis::X, px[i], py[i],
                                pz[i], ax[i], ay[i], az[i], nx[i], ny[i],
                                nz[i]) +
             rund::compute_dsl::plane(PlaneOp::Projection, Axis::Y, px[i], py[i],
                                pz[i], ax[i], ay[i], az[i], nx[i], ny[i],
                                nz[i]) +
             rund::compute_dsl::plane(PlaneOp::Projection, Axis::Z, px[i], py[i],
                                pz[i], ax[i], ay[i], az[i], nx[i], ny[i],
                                nz[i]) +
             rund::compute_dsl::plane(PlaneOp::Distance, MetricOp::Squared, px[i],
                                py[i], pz[i], ax[i], ay[i], az[i], nx[i],
                                ny[i], nz[i]) +
             rund::compute_dsl::plane(PlaneOp::Distance, px[i], py[i], pz[i], ax[i],
                                ay[i], az[i], nx[i], ny[i], nz[i]);
  });
}

[[nodiscard]] bool HasPlaneOps(const std::string &source) {
  return source.find("].op=add_sat") != std::string_view::npos &&
         source.find("].op=mul_fixed") != std::string_view::npos &&
         source.find("].op=sub_sat") != std::string_view::npos &&
         source.find("].op=div_fixed") != std::string_view::npos &&
         source.find("].op=eq") != std::string_view::npos &&
         source.find("].op=select") != std::string_view::npos &&
         source.find("].op=sqrt") != std::string_view::npos;
}

int test_compute_plane_helpers_build_deterministic_lowerable_ir() {
  const auto first32 = BuildPlaneOp<i32>();
  const auto second32 = BuildPlaneOp<i32>();
  const auto fixed_lane64 = BuildPlaneOp<i64>();

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
  TEST_ASSERT(HasPlaneOps(metal32.source_text));
  TEST_ASSERT(HasPlaneOps(vulkan64.source_text));
  return 0;
}

} // namespace

int RunComputeDslOpsPlaneContract() {
  return test_compute_plane_helpers_build_deterministic_lowerable_ir();
}

} // namespace program_compute_contract
