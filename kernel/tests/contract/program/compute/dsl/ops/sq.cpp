#include "contract/program/compute/dsl/ops/local.hpp"

#include <kernel/program/compute/lowering/parse.hpp>

#include <cstddef>
#include <string>
#include <string_view>

namespace program_compute_contract {
namespace {

using namespace dsl_support;

template <typename T> [[nodiscard]] auto BuildSqOp() {
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
  return rund::compute_dsl::def("dsl-sq").on(body).map([](auto i, auto b) {
    auto ax = b.template read<"ax">();
    auto ay = b.template read<"ay">();
    auto az = b.template read<"az">();
    auto bx = b.template read<"bx">();
    auto by = b.template read<"by">();
    auto bz = b.template read<"bz">();
    auto out = b.template write<"out">();
    using MetricOp = rund::compute_dsl::MetricOp;

    out[i] = rund::compute_dsl::len(MetricOp::Squared, ax[i], ay[i]) +
             rund::compute_dsl::len(MetricOp::Squared, ax[i], ay[i], az[i]) +
             rund::compute_dsl::dist(MetricOp::Squared, ax[i], ay[i], bx[i],
                               by[i]) +
             rund::compute_dsl::dist(MetricOp::Squared, ax[i], ay[i], az[i], bx[i],
                               by[i], bz[i]);
  });
}

[[nodiscard]] std::size_t CountOp(
    const rund::kernel::compute_lowering_detail::ParsedIR& parsed,
    const rund::kernel::IrOp op) {
  std::size_t count = 0u;
  for (const auto& node : parsed.nodes) {
    if (node.op == static_cast<rund::kernel::u8>(op)) {
      ++count;
    }
  }
  return count;
}

[[nodiscard]] bool HasSqOps(const std::string& source) {
  return source.find("].op=mul_fixed") != std::string_view::npos &&
         source.find("].op=add_sat") != std::string_view::npos &&
         source.find("].op=sub_sat") != std::string_view::npos;
}

int test_compute_sq_helpers_build_without_sqrt() {
  const auto first32 = BuildSqOp<i32>();
  const auto second32 = BuildSqOp<i32>();
  const auto fixed_lane64 = BuildSqOp<i64>();

  TEST_ASSERT(first32.ok());
  TEST_ASSERT(second32.ok());
  TEST_ASSERT(fixed_lane64.ok());
  TEST_ASSERT(first32.ir().canonical_bytes == second32.ir().canonical_bytes);
  TEST_ASSERT(first32.ir().op_hash_hi == second32.ir().op_hash_hi);
  TEST_ASSERT(first32.ir().op_hash_lo == second32.ir().op_hash_lo);

  const auto parsed32 =
      rund::kernel::compute_lowering_detail::ParseComputeIR(first32.ir());
  const auto parsed64 =
      rund::kernel::compute_lowering_detail::ParseComputeIR(fixed_lane64.ir());
  TEST_ASSERT(parsed32.ok);
  TEST_ASSERT(parsed64.ok);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::Sqrt) == 0u);
  TEST_ASSERT(CountOp(parsed64, rund::kernel::IrOp::Sqrt) == 0u);

  const rund::kernel::LoweringArtifact metal32 =
      rund::kernel::LowerComputeIR(first32.ir(),
                                   rund::kernel::ComputeApi::Metal);
  const rund::kernel::LoweringArtifact vulkan64 =
      rund::kernel::LowerComputeIR(fixed_lane64.ir(),
                                   rund::kernel::ComputeApi::Vulkan);
  TEST_ASSERT(metal32.ok);
  TEST_ASSERT(vulkan64.ok);
  TEST_ASSERT(HasSqOps(metal32.source_text));
  TEST_ASSERT(HasSqOps(vulkan64.source_text));
  return 0;
}

} // namespace

int RunComputeDslOpsSqContract() {
  return test_compute_sq_helpers_build_without_sqrt();
}

} // namespace program_compute_contract
