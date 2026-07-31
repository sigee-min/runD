#pragma once

#include <accel/buffer.hpp>
#include <accel/context/buffer.hpp>
#include <accel/context/buffer/descriptor.hpp>
#include <accel/context/value.hpp>
#include <accel/device.hpp>
#include <accel/kernel/value.hpp>

#include <node/accel/context.hpp>

#include "work.hpp"

namespace node_accel_contract::cpu_context::scan {

struct Resources {
  rund::AccelContext context{};
  rund::AccelBuffer read{};
  rund::AccelBuffer mid{};
  rund::AccelBuffer write{};
  rund::AccelKernel kernel{};
};

[[nodiscard]] inline Resources
BuildBuffers(const rund::AccelDevice &pick,
             const rund::compute_dsl::ComputeOp &op, const Work &work) {
  Resources out{};
  out.context = rund::node::accel::OpenAccel(pick);
  out.read = rund::node::accel::CreateAccelBuffer(
      out.context, rund::AccelBufferDesc{
                       .scalar_width_bytes = sizeof(rund::kernel::u32),
                       .count = kCount,
                       .usage = rund::BufferUsage::ReadOnly,
                   });
  out.mid = rund::node::accel::CreateAccelBuffer(
      out.context, rund::AccelBufferDesc{
                       .scalar_width_bytes = sizeof(rund::kernel::u32),
                       .count = kCount,
                       .usage = rund::BufferUsage::ReadWrite,
                   });
  out.write = rund::node::accel::CreateAccelBuffer(
      out.context, rund::AccelBufferDesc{
                       .scalar_width_bytes = sizeof(rund::kernel::i32),
                       .count = kCount,
                       .usage = rund::BufferUsage::WriteOnly,
                   });
  if (!out.context.check.ok || !op.ok() || !out.read.check.ok ||
      !out.mid.check.ok || !out.write.check.ok ||
      !rund::node::accel::UploadAccelBuffer(
           out.context, out.read, work.input.data(),
           work.input.size() * sizeof(work.input[0]))
           .ok) {
    return out;
  }
  return out;
}

} // namespace node_accel_contract::cpu_context::scan
