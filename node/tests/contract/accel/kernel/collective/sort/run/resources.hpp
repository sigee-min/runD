#pragma once

#include <accel/graph/factory/sort/values.hpp>
#include <kernel/program/compute/sort/plan.hpp>

#include <accel/buffer.hpp>
#include <accel/context/buffer.hpp>
#include <accel/context/value.hpp>
#include <accel/device.hpp>
#include <accel/graph/buffer/ref.hpp>
#include <accel/graph/node.hpp>
#include <accel/graph/value.hpp>
#include <accel/kernel/value.hpp>

#include <node/accel/context.hpp>

#include "ref.hpp"
#include "test/compute/fixed.hpp"

namespace node_accel_contract::collective::sort_run {

template <typename Key, std::size_t Count> struct Resources {
  rund::AccelContext context{};
  rund::AccelBuffer read_keys{};
  rund::AccelBuffer read_values{};
  rund::AccelBuffer write_keys{};
  rund::AccelBuffer write_values{};
  rund::AccelKernel kernel{};
};

template <typename Key, std::size_t Count>
[[nodiscard]] Resources<Key, Count> BuildResources(
    const rund::AccelDevice &pick, const rund::kernel::ComputeScalar scalar,
    const rund::kernel::ComputeDomain domain,
    const std::array<Key, Count> &input_keys, const Reference<Key, Count> &ref,
    const rund::kernel::u32 key_bits) {
  Resources<Key, Count> out{};
  out.context = rund::node::accel::OpenAccel(pick);
  if (!out.context.check.ok) {
    return out;
  }
  out.read_keys = rund::node::accel::CreateAccelBuffer(
      out.context,
      BufferDesc(rund::BufferUsage::ReadOnly, sizeof(Key), input_keys.size()));
  out.read_values = rund::node::accel::CreateAccelBuffer(
      out.context,
      BufferDesc(rund::BufferUsage::ReadOnly, sizeof(rund::kernel::u32),
                 ref.input_values.size()));
  out.write_keys = rund::node::accel::CreateAccelBuffer(
      out.context,
      BufferDesc(rund::BufferUsage::WriteOnly, sizeof(Key), input_keys.size()));
  out.write_values = rund::node::accel::CreateAccelBuffer(
      out.context,
      BufferDesc(rund::BufferUsage::WriteOnly, sizeof(rund::kernel::u32),
                 ref.input_values.size()));
  if (!out.read_keys.check.ok || !out.read_values.check.ok ||
      !out.write_keys.check.ok || !out.write_values.check.ok ||
      !rund::node::accel::UploadAccelBuffer(out.context, out.read_keys,
                                            input_keys.data(),
                                            input_keys.size() * sizeof(Key))
           .ok ||
      !rund::node::accel::UploadAccelBuffer(
           out.context, out.read_values, ref.input_values.data(),
           ref.input_values.size() * sizeof(rund::kernel::u32))
           .ok) {
    return out;
  }

  std::array<rund::AccelGraphBufferRef, 4u> refs{
      rund::AccelGraphBufferRef{
          .buffer = &out.read_keys,
          .role = rund::kernel::BufferRole::Read,
      },
      rund::AccelGraphBufferRef{
          .buffer = &out.read_values,
          .role = rund::kernel::BufferRole::Read,
      },
      rund::AccelGraphBufferRef{
          .buffer = &out.write_keys,
          .role = rund::kernel::BufferRole::Write,
      },
      rund::AccelGraphBufferRef{
          .buffer = &out.write_values,
          .role = rund::kernel::BufferRole::Write,
      },
  };
  const rund::kernel::SortDesc desc{
      .key = sizeof(Key) == sizeof(rund::kernel::u64)
                 ? rund::kernel::SortKey::U64
                 : rund::kernel::SortKey::U32,
      .value = rund::kernel::SortValue::U32,
      .element_count = input_keys.size(),
      .radix_bits = 8u,
      .key_bits = key_bits,
      .stable = true,
  };
  const rund::kernel::SortPlan plan = rund::kernel::PlanSort(desc);
  if (!plan.ok) {
    return out;
  }
  const std::array<rund::AccelGraphNode, 1u> nodes{
      rund::AccelSort(refs.data(), refs.size(), desc)};
  out.kernel = rund::node::accel::CompileAccelKernel(
      out.context,
      rund::AccelGraph{
          .nodes = nodes.data(),
          .node_count = nodes.size(),
          .scalar = scalar,
          .domain = domain,
          .fixed_format = domain == rund::kernel::ComputeDomain::Fixed
                              ? test::FixedFormatForLane(scalar)
                              : rund::kernel::ComputeFixedFormat{},
      });
  return out;
}

} // namespace node_accel_contract::collective::sort_run
