#pragma once

#include <accel/graph/factory/sort/values.hpp>

#include <accel/buffer.hpp>
#include <accel/context/buffer.hpp>
#include <accel/context/value.hpp>
#include <accel/device.hpp>
#include <accel/graph/buffer/ref.hpp>
#include <accel/graph/value.hpp>
#include <accel/graph/node.hpp>
#include <accel/kernel/value.hpp>

#include <node/accel/context.hpp>

#include "../../local.hpp"
#include "test/compute/fixed.hpp"

#include <array>

namespace node_accel_contract::collective::sort_reject {

template <typename Key> struct Resources {
  rund::AccelContext context{};
  rund::AccelBuffer read_keys{};
  rund::AccelBuffer read_values{};
  rund::AccelBuffer write_keys{};
  rund::AccelBuffer write_values{};
  rund::AccelBuffer run_read_keys{};
  rund::AccelBuffer run_read_values{};
  rund::AccelBuffer run_write_keys{};
  rund::AccelBuffer run_write_values{};
  rund::AccelKernel kernel{};
};

template <typename Key>
[[nodiscard]] Resources<Key>
BuildResources(const rund::AccelDevice &pick,
               const rund::kernel::ComputeScalar scalar) {
  Resources<Key> out{};
  out.context = rund::node::accel::OpenAccel(pick);
  if (!out.context.check.ok) {
    return out;
  }
  out.read_keys = rund::node::accel::CreateAccelBuffer(
      out.context, BufferDesc(rund::BufferUsage::ReadOnly, sizeof(Key), 8u));
  out.read_values = rund::node::accel::CreateAccelBuffer(
      out.context,
      BufferDesc(rund::BufferUsage::ReadOnly, sizeof(rund::kernel::u32), 8u));
  out.write_keys = rund::node::accel::CreateAccelBuffer(
      out.context, BufferDesc(rund::BufferUsage::WriteOnly, sizeof(Key), 8u));
  out.write_values = rund::node::accel::CreateAccelBuffer(
      out.context,
      BufferDesc(rund::BufferUsage::WriteOnly, sizeof(rund::kernel::u32), 8u));
  out.run_read_keys = rund::node::accel::CreateAccelBuffer(
      out.context, BufferDesc(rund::BufferUsage::ReadOnly, sizeof(Key), 9u));
  out.run_read_values = rund::node::accel::CreateAccelBuffer(
      out.context,
      BufferDesc(rund::BufferUsage::ReadOnly, sizeof(rund::kernel::u32), 9u));
  out.run_write_keys = rund::node::accel::CreateAccelBuffer(
      out.context, BufferDesc(rund::BufferUsage::WriteOnly, sizeof(Key), 9u));
  out.run_write_values = rund::node::accel::CreateAccelBuffer(
      out.context,
      BufferDesc(rund::BufferUsage::WriteOnly, sizeof(rund::kernel::u32), 9u));
  if (!out.read_keys.check.ok || !out.read_values.check.ok ||
      !out.write_keys.check.ok || !out.write_values.check.ok ||
      !out.run_read_keys.check.ok || !out.run_read_values.check.ok ||
      !out.run_write_keys.check.ok || !out.run_write_values.check.ok) {
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
      .element_count = 8u,
      .radix_bits = 8u,
      .stable = true,
  };
  const std::array<rund::AccelGraphNode, 1u> nodes{
      rund::AccelSort(refs.data(), refs.size(), desc)};
  out.kernel = rund::node::accel::CompileAccelKernel(
      out.context, rund::AccelGraph{
                       .nodes = nodes.data(),
                       .node_count = nodes.size(),
                       .scalar = scalar,
                       .domain = rund::kernel::ComputeDomain::Fixed,
                       .fixed_format = test::FixedFormatForLane(scalar),
                   });
  return out;
}

} // namespace node_accel_contract::collective::sort_reject
