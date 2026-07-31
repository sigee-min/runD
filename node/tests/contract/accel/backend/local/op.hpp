#pragma once

#include <kernel/program/compute/dsl.hpp>

namespace node_accel_contract::backend {

struct ComputeOpFixture {
  rund::kernel::ComputeIR ir{};
  rund::kernel::ComputeMap map{};
};

[[nodiscard]] inline ComputeOpFixture ComputeOp(
    const rund::kernel::ComputeApi api) {
  rund::kernel::i32 dt = 7;
  rund::kernel::i32 input[4]{};
  rund::kernel::i32 output[4]{};
  const auto body = rund::compute_dsl::bind(4u)
                        .fixed<1, 31>()
                        .param<"dt">(dt)
                        .read<"input">(input)
                        .write<"output">(output);

  const rund::compute_dsl::ComputeOp op =
      rund::compute_dsl::def("node-fake-accel").on(body).map([](auto i, auto b) {
        auto dt_value = b.template param<"dt">();
        auto input = b.template read<"input">();
        auto output = b.template write<"output">();
        output[i] = input[i] + dt_value;
      });

  rund::kernel::ComputeMap map = op.map();
  map.api = api;
  return ComputeOpFixture{.ir = op.ir(), .map = map};
}

}  // namespace node_accel_contract::backend
