#include "contract/program/compute/dsl/ops/local.hpp"

#include <kernel/program/compute/lowering/parse.hpp>

#include <cstddef>
#include <string>
#include <string_view>

namespace program_compute_contract {
namespace {

using namespace dsl_support;

template <typename T> [[nodiscard]] auto BuildTolOp() {
  T value[4]{};
  T target[4]{};
  T tol[4]{};
  T out[4]{};
  const auto body = [&]() {
    if constexpr (sizeof(T) == sizeof(i64)) {
      return rund::compute_dsl::bind(4u)
          .fixed<1, 63>()
          .template read<"value">(value)
          .template read<"target">(target)
          .template read<"tol">(tol)
          .template write<"out">(out);
    } else {
      return rund::compute_dsl::bind(4u)
          .fixed<1, 31>()
          .template read<"value">(value)
          .template read<"target">(target)
          .template read<"tol">(tol)
          .template write<"out">(out);
    }
  }();
  return rund::compute_dsl::def("dsl-tol").on(body).map([](auto i, auto b) {
    auto value = b.template read<"value">();
    auto target = b.template read<"target">();
    auto tol = b.template read<"tol">();
    auto out = b.template write<"out">();
    out[i] = rund::compute_dsl::near(value[i], target[i], tol[i]) +
             rund::compute_dsl::near(value[i], tol[i]) +
             rund::compute_dsl::deadzone(value[i], tol[i]) +
             rund::compute_dsl::snap(value[i], target[i], tol[i]);
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

[[nodiscard]] bool HasTolOps(const std::string& source) {
  return source.find("].op=abs") != std::string_view::npos &&
         source.find("].op=sub_sat") != std::string_view::npos &&
         source.find("].op=le") != std::string_view::npos &&
         source.find("].op=select") != std::string_view::npos;
}

int test_compute_tol_helpers_lower_through_existing_ops() {
  const auto first32 = BuildTolOp<i32>();
  const auto second32 = BuildTolOp<i32>();
  const auto fixed_lane64 = BuildTolOp<i64>();

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
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::SubSat) > 0u);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::Le) > 0u);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::Select) > 0u);
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
  TEST_ASSERT(HasTolOps(metal32.source_text));
  TEST_ASSERT(HasTolOps(vulkan64.source_text));
  return 0;
}

} // namespace

int RunComputeDslOpsTolContract() {
  return test_compute_tol_helpers_lower_through_existing_ops();
}

} // namespace program_compute_contract
