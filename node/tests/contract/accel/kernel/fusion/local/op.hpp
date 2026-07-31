#pragma once

#include "model.hpp"

namespace node_accel_contract::fusion {

[[nodiscard]] inline rund::compute_dsl::ComputeOp
BuildAddOp(const char *const name) {
  std::array<rund::kernel::i32, 8u> input{};
  std::array<rund::kernel::i32, 8u> output{};
  const auto body = rund::compute_dsl::bind(input.size())
                        .fixed<1, 31>()
                        .param<"dt">(7)
                        .read<"input">(input.data())
                        .write<"output">(output.data());
  return rund::compute_dsl::def(name).on(body).map([](auto i, auto b) {
    auto dt = b.template param<"dt">();
    auto input = b.template read<"input">();
    auto output = b.template write<"output">();
    output[i] = input[i] + dt;
  });
}

[[nodiscard]] inline rund::compute_dsl::ComputeOp BuildFixedLane32Op() {
  return BuildAddOp("node-context-kernel");
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
  return rund::compute_dsl::def("node-context-fusion-two-read")
      .on(body)
      .map([](auto i, auto b) {
        auto pos = b.template read<"pos">();
        auto vel = b.template read<"vel">();
        auto output = b.template write<"output">();
        output[i] = pos[i] + vel[i];
      });
}

[[nodiscard]] inline rund::compute_dsl::ComputeOp BuildTerminalWriteOp() {
  std::array<rund::kernel::i32, 8u> input{};
  std::array<rund::kernel::i32, 8u> first{};
  std::array<rund::kernel::i32, 8u> second{};
  std::array<rund::kernel::i32, 8u> third{};
  const auto body = rund::compute_dsl::bind(input.size())
                        .fixed<1, 31>()
                        .read<"input">(input.data())
                        .write<"first">(first.data())
                        .write<"second">(second.data())
                        .write<"third">(third.data());
  return rund::compute_dsl::def("node-context-fusion-terminal")
      .on(body)
      .map([](auto i, auto b) {
        const auto input = b.template read<"input">();
        b.template write<"first">()[i] = input[i] + 1;
        b.template write<"second">()[i] = input[i] + 2;
        b.template write<"third">()[i] = input[i] + 3;
      });
}

} // namespace node_accel_contract::fusion
