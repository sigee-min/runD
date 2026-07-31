#include "contract/program/compute/dsl/local.hpp"
#include "test/assert.hpp"

#include <kernel/program/compute/ir.hpp>
#include <kernel/program/compute/lowering/parse.hpp>

#include <cstddef>

namespace program_compute_contract {
namespace {

using namespace dsl_support;

template <typename T> [[nodiscard]] auto BuildRepeatedMulOp() {
  T out[4]{};
  const auto body = [&]() {
    if constexpr (sizeof(T) == sizeof(i64)) {
      return rund::compute_dsl::bind(4u)
          .fixed<1, 63>()
          .template param<"a">(static_cast<T>(3))
          .template param<"b">(static_cast<T>(5))
          .template write<"out">(out);
    } else {
      return rund::compute_dsl::bind(4u)
          .fixed<1, 31>()
          .template param<"a">(static_cast<T>(3))
          .template param<"b">(static_cast<T>(5))
          .template write<"out">(out);
    }
  }();
  return rund::compute_dsl::def("dsl-cse-mul").on(body).map([](auto i, auto b) {
    auto a = b.template param<"a">();
    auto value = b.template param<"b">();
    auto out = b.template write<"out">();

    out[i] = rund::compute_dsl::mul_fixed(a, value) +
             rund::compute_dsl::mul_fixed(a, value);
  });
}

template <typename T> [[nodiscard]] auto BuildRepeatedProjectionOp() {
  T ax[4]{};
  T ay[4]{};
  T bx[4]{};
  T by[4]{};
  T out[4]{};
  const auto body = [&]() {
    if constexpr (sizeof(T) == sizeof(i64)) {
      return rund::compute_dsl::bind(4u)
          .fixed<1, 63>()
          .template read<"ax">(ax)
          .template read<"ay">(ay)
          .template read<"bx">(bx)
          .template read<"by">(by)
          .template write<"out">(out);
    } else {
      return rund::compute_dsl::bind(4u)
          .fixed<1, 31>()
          .template read<"ax">(ax)
          .template read<"ay">(ay)
          .template read<"bx">(bx)
          .template read<"by">(by)
          .template write<"out">(out);
    }
  }();
  return rund::compute_dsl::def("dsl-cse-proj")
      .on(body)
      .map([](auto i, auto b) {
        auto ax = b.template read<"ax">();
        auto ay = b.template read<"ay">();
        auto bx = b.template read<"bx">();
        auto by = b.template read<"by">();
        auto out = b.template write<"out">();

        out[i] = rund::compute_dsl::proj(rund::compute_dsl::Axis::X, ax[i],
                                         ay[i], bx[i], by[i]) +
                 rund::compute_dsl::proj(rund::compute_dsl::Axis::Y, ax[i],
                                         ay[i], bx[i], by[i]);
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

int test_compute_dsl_cse_deduplicates_pure_fixed_nodes() {
  const auto fixed_lane32 = BuildRepeatedMulOp<i32>();
  const auto fixed_lane64 = BuildRepeatedMulOp<i64>();

  TEST_ASSERT(fixed_lane32.ok());
  TEST_ASSERT(fixed_lane64.ok());

  const auto parsed32 =
      rund::kernel::compute_lowering_detail::ParseComputeIR(fixed_lane32.ir());
  const auto parsed64 =
      rund::kernel::compute_lowering_detail::ParseComputeIR(fixed_lane64.ir());

  TEST_ASSERT(parsed32.ok);
  TEST_ASSERT(parsed64.ok);
  TEST_ASSERT(parsed32.nodes.size() == 6u);
  TEST_ASSERT(parsed64.nodes.size() == 6u);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::MulFixed) == 1u);
  TEST_ASSERT(CountOp(parsed64, rund::kernel::IrOp::MulFixed) == 1u);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::Quantize) == 1u);
  TEST_ASSERT(CountOp(parsed64, rund::kernel::IrOp::Quantize) == 1u);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::Write) == 1u);
  TEST_ASSERT(CountOp(parsed64, rund::kernel::IrOp::Write) == 1u);
  return 0;
}

int test_compute_dsl_cse_shares_projection_scale() {
  const auto first = BuildRepeatedProjectionOp<i32>();
  const auto second = BuildRepeatedProjectionOp<i32>();
  const auto fixed_lane64 = BuildRepeatedProjectionOp<i64>();

  TEST_ASSERT(first.ok());
  TEST_ASSERT(second.ok());
  TEST_ASSERT(fixed_lane64.ok());
  TEST_ASSERT(first.ir().canonical_bytes == second.ir().canonical_bytes);

  const auto parsed32 =
      rund::kernel::compute_lowering_detail::ParseComputeIR(first.ir());
  const auto parsed64 =
      rund::kernel::compute_lowering_detail::ParseComputeIR(fixed_lane64.ir());

  TEST_ASSERT(parsed32.ok);
  TEST_ASSERT(parsed64.ok);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::DivFixed) == 1u);
  TEST_ASSERT(CountOp(parsed64, rund::kernel::IrOp::DivFixed) == 1u);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::Select) == 1u);
  TEST_ASSERT(CountOp(parsed64, rund::kernel::IrOp::Select) == 1u);
  return 0;
}

} // namespace

int RunComputeDslCseContract() {
  if (test_compute_dsl_cse_deduplicates_pure_fixed_nodes() != 0) {
    return 1;
  }
  return test_compute_dsl_cse_shares_projection_scale();
}

} // namespace program_compute_contract
