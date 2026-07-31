#include "contract/program/compute/dsl/ops/local.hpp"

#include <string>

namespace program_compute_contract {
namespace {

using namespace dsl_support;

template <typename T> [[nodiscard]] auto BuildLineOp() {
  T px[4]{};
  T py[4]{};
  T ax[4]{};
  T ay[4]{};
  T bx[4]{};
  T by[4]{};
  T out[4]{};
  const auto body = [&]() {
    if constexpr (sizeof(T) == sizeof(i64)) {
      return rund::compute_dsl::bind(4u)
          .fixed<1, 63>()
          .template read<"px">(px)
          .template read<"py">(py)
          .template read<"ax">(ax)
          .template read<"ay">(ay)
          .template read<"bx">(bx)
          .template read<"by">(by)
          .template write<"out">(out);
    } else {
      return rund::compute_dsl::bind(4u)
          .fixed<1, 31>()
          .template read<"px">(px)
          .template read<"py">(py)
          .template read<"ax">(ax)
          .template read<"ay">(ay)
          .template read<"bx">(bx)
          .template read<"by">(by)
          .template write<"out">(out);
    }
  }();
  return rund::compute_dsl::def("dsl-line").on(body).map([](auto i, auto b) {
    auto px = b.template read<"px">();
    auto py = b.template read<"py">();
    auto ax = b.template read<"ax">();
    auto ay = b.template read<"ay">();
    auto bx = b.template read<"bx">();
    auto by = b.template read<"by">();
    auto out = b.template write<"out">();
    using Axis = rund::compute_dsl::Axis;
    using LineOp = rund::compute_dsl::LineOp;
    using MetricOp = rund::compute_dsl::MetricOp;

    out[i] = rund::compute_dsl::line(LineOp::Parameter, px[i], py[i], ax[i], ay[i],
                               bx[i], by[i]) +
             rund::compute_dsl::line(LineOp::Projection, Axis::X, px[i], py[i],
                               ax[i], ay[i], bx[i], by[i]) +
             rund::compute_dsl::line(LineOp::Projection, Axis::Y, px[i], py[i],
                               ax[i], ay[i], bx[i], by[i]) +
             rund::compute_dsl::line(LineOp::Distance, MetricOp::Squared, px[i],
                               py[i], ax[i], ay[i], bx[i], by[i]) +
             rund::compute_dsl::line(LineOp::Distance, px[i], py[i], ax[i], ay[i],
                               bx[i], by[i]);
  });
}

[[nodiscard]] bool HasLineOps(const std::string &source) {
  return source.find("].op=add_sat") != std::string_view::npos &&
         source.find("].op=mul_fixed") != std::string_view::npos &&
         source.find("].op=sub_sat") != std::string_view::npos &&
         source.find("].op=div_fixed") != std::string_view::npos &&
         source.find("].op=eq") != std::string_view::npos &&
         source.find("].op=select") != std::string_view::npos &&
         source.find("].op=sqrt") != std::string_view::npos;
}

int test_compute_line_helpers_build_deterministic_lowerable_ir() {
  const auto first32 = BuildLineOp<i32>();
  const auto second32 = BuildLineOp<i32>();
  const auto fixed_lane64 = BuildLineOp<i64>();

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
  TEST_ASSERT(HasLineOps(metal32.source_text));
  TEST_ASSERT(HasLineOps(vulkan64.source_text));
  return 0;
}

} // namespace

int RunComputeDslOpsLineContract() {
  return test_compute_line_helpers_build_deterministic_lowerable_ir();
}

} // namespace program_compute_contract
