#pragma once

#include <kernel/program/compute/dsl.hpp>
#include <array>
#include <vector>

namespace node_accel_contract::kernel_case {

[[nodiscard]] inline rund::compute_dsl::ComputeOp BuildFixedLane32Op() {
  std::array<rund::kernel::i32, 8u> input{};
  std::array<rund::kernel::i32, 8u> output{};
  const auto body = rund::compute_dsl::bind(input.size())
                        .fixed<1, 31>()
                        .param<"dt">(7)
                        .read<"input">(input.data())
                        .write<"output">(output.data());
  return rund::compute_dsl::def("node-context-kernel").on(body).map(
      [](auto i, auto b) {
        b.template write<"output">()[i] =
            b.template read<"input">()[i] + b.template param<"dt">();
      });
}

[[nodiscard]] inline rund::compute_dsl::ComputeOp BuildFixedLane32OpForTiles(
    const std::size_t tile_count) {
  std::vector<rund::kernel::i32> input(tile_count);
  std::vector<rund::kernel::i32> output(tile_count);
  const auto body = rund::compute_dsl::bind(tile_count)
                        .fixed<1, 31>()
                        .param<"dt">(7)
                        .read<"input">(input.data())
                        .write<"output">(output.data());
  return rund::compute_dsl::def("node-context-kernel-windowed").on(body).map(
      [](auto i, auto b) {
        b.template write<"output">()[i] =
            b.template read<"input">()[i] + b.template param<"dt">();
      });
}

[[nodiscard]] inline rund::compute_dsl::ComputeOp BuildTwoReadFixedLane32Op() {
  std::array<rund::kernel::i32, 8u> pos{};
  std::array<rund::kernel::i32, 8u> vel{};
  std::array<rund::kernel::i32, 8u> output{};
  const auto body = rund::compute_dsl::bind(pos.size())
                        .fixed<1, 31>()
                        .read<"pos">(pos.data())
                        .read<"vel">(vel.data())
                        .write<"output">(output.data());
  return rund::compute_dsl::def("node-context-kernel-two-read").on(body).map(
      [](auto i, auto b) {
        b.template write<"output">()[i] =
            b.template read<"pos">()[i] + b.template read<"vel">()[i];
      });
}

[[nodiscard]] inline rund::compute_dsl::ComputeOp BuildMultiWriteFixedLane32Op() {
  std::array<rund::kernel::i32, 8u> input{};
  std::array<rund::kernel::i32, 8u> plus{};
  std::array<rund::kernel::i32, 8u> times{};
  const auto body = rund::compute_dsl::bind(input.size())
                        .fixed<1, 31>()
                        .read<"input">(input.data())
                        .write<"plus">(plus.data())
                        .write<"times">(times.data());
  return rund::compute_dsl::def("node-context-kernel-multi-write")
      .on(body)
      .map([](auto i, auto b) {
        const auto input = b.template read<"input">();
        b.template write<"plus">()[i] = input[i] + 1;
        b.template write<"times">()[i] = input[i] + input[i] + input[i];
      });
}

}  // namespace node_accel_contract::kernel_case
