#pragma once

#include <accel/buffer.hpp>
#include <accel/context/buffer.hpp>
#include <accel/context/value.hpp>

#include <node/accel/context.hpp>

#include "../../local.hpp"

namespace node_accel_contract::fusion::chain {

struct Resources {
  rund::AccelBuffer fused_input{};
  rund::AccelBuffer fused_mid{};
  rund::AccelBuffer fused_output{};
  rund::AccelBuffer ref_input{};
  rund::AccelBuffer ref_mid{};
  rund::AccelBuffer ref_output{};
  bool ok = false;
};

[[nodiscard]] Resources MakeResources(const rund::AccelContext &context,
                                      const Inputs &inputs) {
  Resources out{
      .fused_input = MakeBuffer(context, rund::BufferUsage::ReadOnly),
      .fused_mid = MakeBuffer(context, rund::BufferUsage::ReadWrite),
      .fused_output = MakeBuffer(context, rund::BufferUsage::WriteOnly),
      .ref_input = MakeBuffer(context, rund::BufferUsage::ReadOnly),
      .ref_mid = MakeBuffer(context, rund::BufferUsage::ReadWrite),
      .ref_output = MakeBuffer(context, rund::BufferUsage::WriteOnly),
  };
  out.ok = out.fused_input.check.ok && out.fused_mid.check.ok &&
           out.fused_output.check.ok && out.ref_input.check.ok &&
           out.ref_mid.check.ok && out.ref_output.check.ok &&
           rund::node::accel::UploadAccelBuffer(context, out.fused_input,
                                                inputs.host.data(),
                                                sizeof(inputs.host))
               .ok &&
           rund::node::accel::UploadAccelBuffer(
               context, out.ref_input, inputs.host.data(), sizeof(inputs.host))
               .ok;
  return out;
}

} // namespace node_accel_contract::fusion::chain
