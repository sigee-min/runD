#include "../model.hpp"

#include "../../../../../../src/compute/flow/state.hpp"
#include "../../../../../../src/compute/program/state.hpp"

#include <array>
#include <cstdio>
#include <vector>

namespace rund_node_flow_contract {

template <typename Header, typename Secondary, std::size_t Count>
int CheckHeterogeneousDivide(const rund::compute::Backend backend,
                             const char *const name,
                             const std::array<Header, Count> &header,
                             const std::array<Secondary, Count> &secondary,
                             const std::vector<Header> &expected_header,
                             const std::vector<Secondary> &expected_secondary) {
  using namespace rund::compute;
  auto program =
      flow_on(backend)
          .template input<Header>(header.size())
          .template zip_input<Secondary>(secondary.size())
          .map(name,
               [](auto header_value, auto secondary_value) {
                 return record(
                     field<OpField<114>>(header_value / static_cast<Header>(2)),
                     field<OpField<115>>(secondary_value /
                                         static_cast<Secondary>(2)));
               })
          .compile();
  if (!program) {
    std::fprintf(stderr,
                 "heterogeneous divide compile failed backend=%u name=%s "
                 "reason=%.*s\n",
                 static_cast<unsigned>(backend), name,
                 static_cast<int>(program.error().size()),
                 program.error().data());
    return 1;
  }
  auto job = program->resident(header, secondary);
  if (!job || !job->run()) {
    return 2;
  }
  const auto header_output = job->template read<0u>();
  const auto secondary_output = job->template read<1u>();
  const auto stats = job->stats();
  if (!header_output || *header_output != expected_header ||
      !secondary_output || *secondary_output != expected_secondary ||
      stats.graph_hash == 0u || stats.output_hash == 0u) {
    return 3;
  }
  return 0;
}

int CheckRecordFields(const rund::compute::Backend backend) {
  using namespace rund::compute;
  const std::array<std::int32_t, 4u> input{-2, 0, 3, 5};
  const std::array<std::uint32_t, 4u> weights{3u, 4u, 5u, 6u};
  auto planned =
      flow_on(backend)
          .input<std::int32_t>(input.size())
          .zip_input<std::uint32_t>(weights.size())
          .map("weighted",
               [](auto value, auto weight) { return value * weight; })
          .compile();
  if (!planned) {
    std::fprintf(
        stderr, "typed record zip compile failed backend=%u reason=%.*s\n",
        static_cast<unsigned>(backend),
        static_cast<int>(planned.error().size()), planned.error().data());
    return 1;
  }
  auto job = planned->resident(input, weights);
  if (!job || !job->run()) {
    return 2;
  }
  auto planned_output = job->read();
  if (!planned_output ||
      *planned_output != std::vector<std::uint32_t>{
                             static_cast<std::uint32_t>(-6), 0u, 15u, 30u}) {
    return 3;
  }
  const std::array<std::uint32_t, 4u> ordered{0u, 0x7fffffffu, 0x80000000u,
                                              0xffffffffu};
  auto record_program =
      flow_on(backend)
          .input<std::int32_t>(input.size())
          .zip_input<std::uint32_t>(ordered.size())
          .map("typed-input-record",
               [](auto value, auto weight) {
                 return record(field<ValueField>(value),
                               field<WeightField>(weight),
                               field<ScoreField>(mask(weight > 0x7fffffffu)));
               })
          .compile();
  if (!record_program) {
    return 4;
  }
  auto record_job = record_program->resident(input, ordered);
  if (!record_job || !record_job->run()) {
    return 5;
  }
  auto order_flags = record_job->template read<2u>();
  if (!order_flags ||
      *order_flags != std::vector<std::uint32_t>{0u, 0u, 1u, 1u}) {
    return 6;
  }

  const std::array<std::uint32_t, 4u> unsigned_header{0u, 0x7fffffffu,
                                                      0x80000000u, 0xffffffffu};
  const std::array<std::int32_t, 4u> signed_secondary{-2, -1, 0, 1};
  auto heterogeneous_order =
      flow_on(backend)
          .input<std::uint32_t>(unsigned_header.size())
          .zip_input<std::int32_t>(signed_secondary.size())
          .map("heterogeneous-order-domain",
               [](auto unsigned_value, auto signed_value) {
                 constexpr std::uint32_t pivot = 0x80000000u;
                 return record(
                     field<OpField<100>>(min(signed_value, 0)),
                     field<OpField<101>>(max(signed_value, 0)),
                     field<OpField<102>>(clamp(signed_value, -1, 0)),
                     field<OpField<103>>(mask(signed_value < 0)),
                     field<OpField<104>>(mask(signed_value <= 0)),
                     field<OpField<105>>(mask(signed_value > 0)),
                     field<OpField<106>>(mask(signed_value >= 0)),
                     field<OpField<107>>(min(unsigned_value, pivot)),
                     field<OpField<108>>(max(unsigned_value, pivot)),
                     field<OpField<109>>(
                         clamp(unsigned_value, 0x7fffffffu, pivot)),
                     field<OpField<110>>(mask(unsigned_value < pivot)),
                     field<OpField<111>>(mask(unsigned_value <= pivot)),
                     field<OpField<112>>(mask(unsigned_value > pivot)),
                     field<OpField<113>>(mask(unsigned_value >= pivot)));
               })
          .compile();
  if (!heterogeneous_order) {
    std::fprintf(stderr,
                 "heterogeneous order compile failed backend=%u reason=%.*s\n",
                 static_cast<unsigned>(backend),
                 static_cast<int>(heterogeneous_order.error().size()),
                 heterogeneous_order.error().data());
    return 34;
  }
  auto heterogeneous_order_job =
      heterogeneous_order->resident(unsigned_header, signed_secondary);
  if (!heterogeneous_order_job || !heterogeneous_order_job->run()) {
    return 35;
  }
  const auto signed_min = heterogeneous_order_job->template read<0u>();
  const auto signed_max = heterogeneous_order_job->template read<1u>();
  const auto signed_clamp = heterogeneous_order_job->template read<2u>();
  const auto signed_lt = heterogeneous_order_job->template read<3u>();
  const auto signed_le = heterogeneous_order_job->template read<4u>();
  const auto signed_gt = heterogeneous_order_job->template read<5u>();
  const auto signed_ge = heterogeneous_order_job->template read<6u>();
  const auto unsigned_min = heterogeneous_order_job->template read<7u>();
  const auto unsigned_max = heterogeneous_order_job->template read<8u>();
  const auto unsigned_clamp = heterogeneous_order_job->template read<9u>();
  const auto unsigned_lt = heterogeneous_order_job->template read<10u>();
  const auto unsigned_le = heterogeneous_order_job->template read<11u>();
  const auto unsigned_gt = heterogeneous_order_job->template read<12u>();
  const auto unsigned_ge = heterogeneous_order_job->template read<13u>();
  const std::vector<std::uint32_t> lower_flags{1u, 1u, 0u, 0u};
  const std::vector<std::uint32_t> lower_equal_flags{1u, 1u, 1u, 0u};
  const std::vector<std::uint32_t> greater_flags{0u, 0u, 0u, 1u};
  const std::vector<std::uint32_t> greater_equal_flags{0u, 0u, 1u, 1u};
  const auto heterogeneous_stats = heterogeneous_order_job->stats();
  if (!signed_min || *signed_min != std::vector<std::int32_t>{-2, -1, 0, 0} ||
      !signed_max || *signed_max != std::vector<std::int32_t>{0, 0, 0, 1} ||
      !signed_clamp ||
      *signed_clamp != std::vector<std::int32_t>{-1, -1, 0, 0} || !signed_lt ||
      *signed_lt != lower_flags || !signed_le ||
      *signed_le != lower_equal_flags || !signed_gt ||
      *signed_gt != greater_flags || !signed_ge ||
      *signed_ge != greater_equal_flags || !unsigned_min ||
      *unsigned_min != std::vector<std::uint32_t>{0u, 0x7fffffffu, 0x80000000u,
                                                  0x80000000u} ||
      !unsigned_max ||
      *unsigned_max != std::vector<std::uint32_t>{0x80000000u, 0x80000000u,
                                                  0x80000000u, 0xffffffffu} ||
      !unsigned_clamp ||
      *unsigned_clamp != std::vector<std::uint32_t>{0x7fffffffu, 0x7fffffffu,
                                                    0x80000000u, 0x80000000u} ||
      !unsigned_lt || *unsigned_lt != lower_flags || !unsigned_le ||
      *unsigned_le != lower_equal_flags || !unsigned_gt ||
      *unsigned_gt != greater_flags || !unsigned_ge ||
      *unsigned_ge != greater_equal_flags ||
      heterogeneous_stats.graph_hash == 0u ||
      heterogeneous_stats.output_hash == 0u) {
    return 36;
  }
  const std::array<std::uint32_t, 4u> unsigned_dividend32{0u, 10u, 0x80000000u,
                                                          0xffffffffu};
  const std::array<std::int32_t, 4u> signed_dividend32{-9, -4, 5, 12};
  const std::vector<std::uint32_t> unsigned_quotient32{0u, 5u, 0x40000000u,
                                                       0x7fffffffu};
  const std::vector<std::int32_t> signed_quotient32{-4, -2, 2, 6};
  if (const int divide = CheckHeterogeneousDivide(
          backend, "heterogeneous-divide-u32-header", unsigned_dividend32,
          signed_dividend32, unsigned_quotient32, signed_quotient32);
      divide != 0) {
    return 40 + divide;
  }
  if (const int divide = CheckHeterogeneousDivide(
          backend, "heterogeneous-divide-i32-header", signed_dividend32,
          unsigned_dividend32, signed_quotient32, unsigned_quotient32);
      divide != 0) {
    return 44 + divide;
  }

  const std::array<std::uint64_t, 4u> unsigned_dividend64{
      0u, 10u, 0x8000000000000000ull, 0xffffffffffffffffull};
  const std::array<std::int64_t, 4u> signed_dividend64{-9, -4, 5, 12};
  const std::vector<std::uint64_t> unsigned_quotient64{
      0u, 5u, 0x4000000000000000ull, 0x7fffffffffffffffull};
  const std::vector<std::int64_t> signed_quotient64{-4, -2, 2, 6};
  if (const int divide = CheckHeterogeneousDivide(
          backend, "heterogeneous-divide-u64-header", unsigned_dividend64,
          signed_dividend64, unsigned_quotient64, signed_quotient64);
      divide != 0) {
    return 48 + divide;
  }
  if (const int divide = CheckHeterogeneousDivide(
          backend, "heterogeneous-divide-i64-header", signed_dividend64,
          unsigned_dividend64, signed_quotient64, unsigned_quotient64);
      divide != 0) {
    return 52 + divide;
  }
  auto retyped_constant =
      flow_on(backend)
          .input<std::int32_t>(input.size())
          .map("typed-constant-output",
               [](auto value) { return select(value == value, 7u, 7u); })
          .compile();
  if (!retyped_constant) {
    return 37;
  }
  auto retyped_constant_job = retyped_constant->resident(input);
  if (!retyped_constant_job || !retyped_constant_job->run()) {
    return 38;
  }
  const auto retyped_values = retyped_constant_job->read();
  const auto retyped_stats = retyped_constant_job->stats();
  if (!retyped_values ||
      *retyped_values != std::vector<std::uint32_t>{7u, 7u, 7u, 7u} ||
      retyped_stats.graph_hash == 0u || retyped_stats.output_hash == 0u) {
    return 39;
  }
  return 0;
}

} // namespace rund_node_flow_contract
