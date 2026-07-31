#include "contract/program/compute/dsl/ops/local.hpp"

#include <kernel/program/compute/lowering/parse.hpp>

#include <cstddef>
#include <string>
#include <string_view>

namespace program_compute_contract {
namespace {

using namespace dsl_support;

template <typename T> [[nodiscard]] auto BuildLinearOp() {
  T x0[4]{};
  T x1[4]{};
  T x2[4]{};
  T x3[4]{};
  T x4[4]{};
  T x5[4]{};
  T x6[4]{};
  T x7[4]{};
  T x8[4]{};
  T k0[4]{};
  T k1[4]{};
  T k2[4]{};
  T k3[4]{};
  T k4[4]{};
  T k5[4]{};
  T k6[4]{};
  T k7[4]{};
  T k8[4]{};
  T out[4]{};
  const auto body = [&]() {
    if constexpr (sizeof(T) == sizeof(i64)) {
      return rund::compute_dsl::bind(4u)
          .fixed<1, 63>()
          .template read<"x0">(x0)
          .template read<"x1">(x1)
          .template read<"x2">(x2)
          .template read<"x3">(x3)
          .template read<"x4">(x4)
          .template read<"x5">(x5)
          .template read<"x6">(x6)
          .template read<"x7">(x7)
          .template read<"x8">(x8)
          .template read<"k0">(k0)
          .template read<"k1">(k1)
          .template read<"k2">(k2)
          .template read<"k3">(k3)
          .template read<"k4">(k4)
          .template read<"k5">(k5)
          .template read<"k6">(k6)
          .template read<"k7">(k7)
          .template read<"k8">(k8)
          .template write<"out">(out);
    } else {
      return rund::compute_dsl::bind(4u)
          .fixed<1, 31>()
          .template read<"x0">(x0)
          .template read<"x1">(x1)
          .template read<"x2">(x2)
          .template read<"x3">(x3)
          .template read<"x4">(x4)
          .template read<"x5">(x5)
          .template read<"x6">(x6)
          .template read<"x7">(x7)
          .template read<"x8">(x8)
          .template read<"k0">(k0)
          .template read<"k1">(k1)
          .template read<"k2">(k2)
          .template read<"k3">(k3)
          .template read<"k4">(k4)
          .template read<"k5">(k5)
          .template read<"k6">(k6)
          .template read<"k7">(k7)
          .template read<"k8">(k8)
          .template write<"out">(out);
    }
  }();
  return rund::compute_dsl::def("dsl-linear").on(body).map([](auto i, auto bind) {
    auto x0 = bind.template read<"x0">();
    auto x1 = bind.template read<"x1">();
    auto x2 = bind.template read<"x2">();
    auto x3 = bind.template read<"x3">();
    auto x4 = bind.template read<"x4">();
    auto x5 = bind.template read<"x5">();
    auto x6 = bind.template read<"x6">();
    auto x7 = bind.template read<"x7">();
    auto x8 = bind.template read<"x8">();
    auto k0 = bind.template read<"k0">();
    auto k1 = bind.template read<"k1">();
    auto k2 = bind.template read<"k2">();
    auto k3 = bind.template read<"k3">();
    auto k4 = bind.template read<"k4">();
    auto k5 = bind.template read<"k5">();
    auto k6 = bind.template read<"k6">();
    auto k7 = bind.template read<"k7">();
    auto k8 = bind.template read<"k8">();
    auto out = bind.template write<"out">();
    out[i] = rund::compute_dsl::dot(x0[i], x1[i], x2[i], x3[i], k0[i], k1[i],
                               k2[i], k3[i]) +
             rund::compute_dsl::dot(x0[i], x1[i], x2[i], x3[i], x4[i], k0[i],
                               k1[i], k2[i], k3[i], k4[i]) +
             rund::compute_dsl::dot(x0[i], x1[i], x2[i], x3[i], x4[i], x5[i],
                               k0[i], k1[i], k2[i], k3[i], k4[i], k5[i]) +
             rund::compute_dsl::dot(x0[i], x1[i], x2[i], x3[i], x4[i], x5[i],
                               x6[i], x7[i], k0[i], k1[i], k2[i], k3[i],
                               k4[i], k5[i], k6[i], k7[i]) +
             rund::compute_dsl::conv(x0[i], x1[i], x2[i], k0[i], k1[i], k2[i]) +
             rund::compute_dsl::conv(x0[i], x1[i], x2[i], x3[i], x4[i], k0[i],
                                k1[i], k2[i], k3[i], k4[i]) +
             rund::compute_dsl::conv(x0[i], x1[i], x2[i], x3[i], x4[i], x5[i],
                                x6[i], k0[i], k1[i], k2[i], k3[i], k4[i],
                                k5[i], k6[i]) +
             rund::compute_dsl::conv(x0[i], x1[i], x2[i], x3[i], x4[i], x5[i],
                                x6[i], x7[i], x8[i], k0[i], k1[i], k2[i],
                                k3[i], k4[i], k5[i], k6[i], k7[i], k8[i]);
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

[[nodiscard]] bool HasLinearOps(const std::string& source) {
  return source.find("].op=mul_fixed") != std::string_view::npos &&
         source.find("].op=add_sat") != std::string_view::npos;
}

int test_compute_linear_helpers_lower_through_existing_ops() {
  const auto first32 = BuildLinearOp<i32>();
  const auto second32 = BuildLinearOp<i32>();
  const auto fixed_lane64 = BuildLinearOp<i64>();

  TEST_ASSERT(first32.ok());
  TEST_ASSERT(second32.ok());
  TEST_ASSERT(fixed_lane64.ok());
  TEST_ASSERT(first32.ir().canonical_bytes == second32.ir().canonical_bytes);
  TEST_ASSERT(first32.ir().op_hash_hi == second32.ir().op_hash_hi);
  TEST_ASSERT(first32.ir().op_hash_lo == second32.ir().op_hash_lo);

  const auto parsed32 =
      rund::kernel::compute_lowering_detail::ParseComputeIR(first32.ir());
  TEST_ASSERT(parsed32.ok);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::MulFixed) > 0u);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::AddSat) > 0u);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::DivFixed) == 0u);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::Sqrt) == 0u);

  const rund::kernel::LoweringArtifact metal32 =
      rund::kernel::LowerComputeIR(first32.ir(),
                                   rund::kernel::ComputeApi::Metal);
  const rund::kernel::LoweringArtifact vulkan64 =
      rund::kernel::LowerComputeIR(fixed_lane64.ir(),
                                   rund::kernel::ComputeApi::Vulkan);
  TEST_ASSERT(metal32.ok);
  TEST_ASSERT(vulkan64.ok);
  TEST_ASSERT(HasLinearOps(metal32.source_text));
  TEST_ASSERT(HasLinearOps(vulkan64.source_text));
  return 0;
}

} // namespace

int RunComputeDslOpsLinearContract() {
  return test_compute_linear_helpers_lower_through_existing_ops();
}

} // namespace program_compute_contract
