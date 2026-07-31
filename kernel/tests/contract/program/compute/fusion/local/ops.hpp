#pragma once

#include "base.hpp"

namespace program_compute_contract::fusion_support {

[[nodiscard]] inline rund::compute_dsl::ComputeOp BuildAddFiveOp() {
  std::array<rund::kernel::i32, 4u> input{};
  std::array<rund::kernel::i32, 4u> output{};
  const auto body = rund::compute_dsl::bind(input.size())
                        .fixed<1, 31>()
                        .param<"addend">(5)
                        .read<"input">(input.data())
                        .write<"middle">(output.data());
  return rund::compute_dsl::def("fusion-add-five")
      .on(body)
      .map([](auto i, auto b) {
        auto addend = b.template param<"addend">();
        auto input = b.template read<"input">();
        auto middle = b.template write<"middle">();
        middle[i] = input[i] + addend;
      });
}

[[nodiscard]] inline rund::compute_dsl::ComputeOp BuildMulThreeOp() {
  std::array<rund::kernel::i32, 4u> input{};
  std::array<rund::kernel::i32, 4u> output{};
  const auto body = rund::compute_dsl::bind(input.size())
                        .fixed<1, 31>()
                        .param<"factor">(3)
                        .read<"middle">(input.data())
                        .write<"output">(output.data());
  return rund::compute_dsl::def("fusion-mul-three")
      .on(body)
      .map([](auto i, auto b) {
        auto factor = b.template param<"factor">();
        auto middle = b.template read<"middle">();
        auto output = b.template write<"output">();
        output[i] = middle[i] * factor;
      });
}

[[nodiscard]] inline rund::compute_dsl::ComputeOp BuildMultiWriteOp() {
  std::array<rund::kernel::i32, 4u> input{};
  std::array<rund::kernel::i32, 4u> first{};
  std::array<rund::kernel::i32, 4u> second{};
  std::array<rund::kernel::i32, 4u> third{};
  const auto body = rund::compute_dsl::bind(input.size())
                        .fixed<1, 31>()
                        .read<"middle">(input.data())
                        .write<"first">(first.data())
                        .write<"second">(second.data())
                        .write<"third">(third.data());
  return rund::compute_dsl::def("fusion-multi-write")
      .on(body)
      .map([](auto i, auto b) {
        auto middle = b.template read<"middle">();
        auto first = b.template write<"first">();
        auto second = b.template write<"second">();
        auto third = b.template write<"third">();
        first[i] = middle[i] + 1;
        second[i] = middle[i] + 2;
        third[i] = middle[i] + 3;
      });
}

[[nodiscard]] inline rund::compute_dsl::ComputeOp BuildBitShiftConsumerOp() {
  std::array<rund::kernel::i32, 4u> input{};
  std::array<rund::kernel::i32, 4u> output{};
  const auto body = rund::compute_dsl::bind(input.size())
                        .fixed<1, 31>()
                        .read<"middle">(input.data())
                        .write<"output">(output.data());
  return rund::compute_dsl::def("fusion-fixed-bit-shift-consumer")
      .on(body)
      .map([](auto i, auto b) {
        auto middle = b.template read<"middle">();
        auto output = b.template write<"output">();
        auto value = middle[i];
        output[i] = rund::compute_dsl::shl_const<3>(value) +
                    rund::compute_dsl::shr_logical_const<5>(value) +
                    rund::compute_dsl::shr_arithmetic_const<7>(value);
      });
}

[[nodiscard]] inline rund::kernel::u32 FirstProducerOutputNode(
    const rund::kernel::compute_lowering_detail::ParsedIR &parsed) noexcept {
  rund::kernel::u32 add = 0u;
  for (std::size_t index = 0u; index < parsed.nodes.size(); ++index) {
    const auto &node = parsed.nodes[index];
    if (node.op == static_cast<rund::kernel::u8>(rund::kernel::IrOp::Add) &&
        node.aux == 0u) {
      add = static_cast<rund::kernel::u32>(index + 1u);
      break;
    }
  }
  for (std::size_t index = 0u; index < parsed.nodes.size(); ++index) {
    const auto &node = parsed.nodes[index];
    if (node.op ==
            static_cast<rund::kernel::u8>(rund::kernel::IrOp::Quantize) &&
        node.lhs == add) {
      return static_cast<rund::kernel::u32>(index + 1u);
    }
  }
  return 0u;
}

} // namespace program_compute_contract::fusion_support
