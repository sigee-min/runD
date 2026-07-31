#pragma once

#include "work.hpp"

#include <kernel/program/compute/dsl.hpp>

namespace node_accel_contract::vulkan {

[[nodiscard]] inline rund::compute_dsl::ComputeOp
MakeValidForgeTarget(ForgedArtifactWork& work) {
  const auto body = rund::compute_dsl::bind(kForgedArtifactTileCount)
                        .fixed<1, 31>()
                        .param<"scale">(work.scale)
                        .read<"lhs">(work.lhs.data())
                        .read<"rhs">(work.rhs.data())
                        .write<"output">(work.output.data());
  return rund::compute_dsl::def("node-vulkan-valid-forge-target")
      .on(body)
      .map([](auto i, auto b) {
        auto scale_value = b.template param<"scale">();
        auto lhs_values = b.template read<"lhs">();
        auto rhs_values = b.template read<"rhs">();
        auto out = b.template write<"output">();
        out[i] = lhs_values[i] + rhs_values[i] * scale_value;
      });
}

[[nodiscard]] inline rund::compute_dsl::ComputeOp
MakeForgedBody(ForgedArtifactWork& work) {
  const auto body = rund::compute_dsl::bind(kForgedArtifactTileCount)
                        .fixed<1, 31>()
                        .param<"scale">(work.scale)
                        .read<"lhs">(work.lhs.data())
                        .read<"rhs">(work.rhs.data())
                        .write<"output">(work.output.data());
  return rund::compute_dsl::def("node-vulkan-forged-body")
      .on(body)
      .map([](auto i, auto b) {
        auto scale_value = b.template param<"scale">();
        auto lhs_values = b.template read<"lhs">();
        auto rhs_values = b.template read<"rhs">();
        auto out = b.template write<"output">();
        out[i] = lhs_values[i] + rhs_values[i] + scale_value;
      });
}

}  // namespace node_accel_contract::vulkan
