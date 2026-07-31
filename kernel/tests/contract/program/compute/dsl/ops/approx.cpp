#include "contract/program/compute/dsl/ops/local.hpp"

#include <kernel/program/compute/lowering/parse.hpp>

#include <cstddef>
#include <string>
#include <string_view>

namespace program_compute_contract {
namespace {

using namespace dsl_support;

template <typename T> [[nodiscard]] auto BuildApproxOp() {
  T value[4]{};
  T amount[4]{};
  T out[4]{};
  const auto body = [&]() {
    if constexpr (sizeof(T) == sizeof(i64)) {
      return rund::compute_dsl::bind(4u)
          .fixed<1, 63>()
          .template read<"value">(value)
          .template read<"amount">(amount)
          .template write<"out">(out);
    } else {
      return rund::compute_dsl::bind(4u)
          .fixed<1, 31>()
          .template read<"value">(value)
          .template read<"amount">(amount)
          .template write<"out">(out);
    }
  }();
  return rund::compute_dsl::def("dsl-approx").on(body).map([](auto i, auto b) {
    auto value = b.template read<"value">();
    auto amount = b.template read<"amount">();
    auto out = b.template write<"out">();
    out[i] =
        rund::compute_dsl::softsign(value[i]) +
        rund::compute_dsl::saturate(rund::compute_dsl::mean(
            rund::compute_dsl::softsign(value[i], amount[i]),
            rund::compute_dsl::fixed_max(value[i]))) +
        rund::compute_dsl::saturate(
            rund::compute_dsl::mean(rund::compute_dsl::softsign(value[i]),
                                    rund::compute_dsl::fixed_max(value[i]))) +
        rund::compute_dsl::activation(rund::compute_dsl::ActivationOp::Relu,
                                      value[i]) +
        rund::compute_dsl::activation(rund::compute_dsl::ActivationOp::Relu,
                                      value[i], amount[i]) +
        rund::compute_dsl::activation(
            rund::compute_dsl::ActivationOp::LeakyRelu, value[i], amount[i]) +
        rund::compute_dsl::activation(
            rund::compute_dsl::ActivationOp::HardSigmoid, value[i]) +
        rund::compute_dsl::activation(
            rund::compute_dsl::ActivationOp::HardSwish, value[i]) +
        rund::compute_dsl::activation(rund::compute_dsl::ActivationOp::HardTanh,
                                      value[i]) +
        rund::compute_dsl::window(rund::compute_dsl::WindowOp::Parabolic,
                                  amount[i]) +
        rund::compute_dsl::window(rund::compute_dsl::WindowOp::Triangular,
                                  amount[i]);
  });
}

[[nodiscard]] std::size_t
CountOp(const rund::kernel::compute_lowering_detail::ParsedIR &parsed,
        const rund::kernel::IrOp op) {
  std::size_t count = 0u;
  for (const auto &node : parsed.nodes) {
    if (node.op == static_cast<rund::kernel::u8>(op)) {
      ++count;
    }
  }
  return count;
}

[[nodiscard]] bool HasApproxOps(const std::string &source) {
  return source.find("].op=abs") != std::string_view::npos &&
         source.find("].op=div_fixed") != std::string_view::npos &&
         source.find("].op=eq") != std::string_view::npos &&
         source.find("].op=select") != std::string_view::npos &&
         source.find("].op=mul_fixed") != std::string_view::npos &&
         source.find("].op=sub_sat") != std::string_view::npos &&
         source.find("].op=clamp") != std::string_view::npos;
}

int test_compute_approx_helpers_lower_through_existing_ops() {
  const auto first32 = BuildApproxOp<i32>();
  const auto second32 = BuildApproxOp<i32>();
  const auto fixed_lane64 = BuildApproxOp<i64>();

  TEST_ASSERT(first32.ok());
  TEST_ASSERT(second32.ok());
  TEST_ASSERT(fixed_lane64.ok());
  TEST_ASSERT(first32.ir().canonical_bytes == second32.ir().canonical_bytes);
  TEST_ASSERT(first32.ir().op_hash_hi == second32.ir().op_hash_hi);
  TEST_ASSERT(first32.ir().op_hash_lo == second32.ir().op_hash_lo);

  const auto parsed32 =
      rund::kernel::compute_lowering_detail::ParseComputeIR(first32.ir());
  TEST_ASSERT(parsed32.ok);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::Abs) > 0u);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::DivFixed) > 0u);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::Select) > 0u);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::MulFixed) > 0u);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::Max) > 0u);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::AddSat) > 0u);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::SubSat) > 0u);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::Clamp) > 0u);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::Sqrt) == 0u);

  const rund::kernel::LoweringArtifact metal32 = rund::kernel::LowerComputeIR(
      first32.ir(), rund::kernel::ComputeApi::Metal);
  const rund::kernel::LoweringArtifact vulkan64 = rund::kernel::LowerComputeIR(
      fixed_lane64.ir(), rund::kernel::ComputeApi::Vulkan);
  TEST_ASSERT(metal32.ok);
  TEST_ASSERT(vulkan64.ok);
  TEST_ASSERT(HasApproxOps(metal32.source_text));
  TEST_ASSERT(HasApproxOps(vulkan64.source_text));
  return 0;
}

} // namespace

int RunComputeDslOpsApproxContract() {
  return test_compute_approx_helpers_lower_through_existing_ops();
}

} // namespace program_compute_contract
