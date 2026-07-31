#pragma once

#include <accel/graph/factory/sort/values.hpp>
#include <accel/kernel/run.hpp>

#include <node/accel/buffer.hpp>
#include <node/accel/context.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace node_accel_contract::collective::bounded_sort {

inline constexpr std::size_t kCapacity = 16u;
inline constexpr rund::kernel::u32 kKeySentinel = 0xdeadbeefu;
inline constexpr rund::kernel::u32 kValueSentinel = 0xfefefefeu;
inline constexpr std::size_t kSparseCount = 7u;
inline constexpr std::array<rund::kernel::u32, kCapacity> kInputKeys{
    4u, 1u, 3u, 1u, 2u, 1u, 0u, 99u, 98u, 97u, 96u, 95u, 94u, 93u, 92u, 91u};
inline constexpr std::array<rund::kernel::u32, kCapacity> kInputValues{
    40u, 10u, 30u, 11u, 20u, 12u, 0u,  99u,
    98u, 97u, 96u, 95u, 94u, 93u, 92u, 91u};
inline constexpr std::array<rund::kernel::u32, kSparseCount> kSortedKeys{
    0u, 1u, 1u, 1u, 2u, 3u, 4u};
inline constexpr std::array<rund::kernel::u32, kSparseCount> kSortedValues{
    0u, 10u, 11u, 12u, 20u, 30u, 40u};

struct Resources final {
  rund::AccelContext context{};
  rund::AccelBuffer keys{};
  rund::AccelBuffer count{};
  rund::AccelBuffer values{};
  rund::AccelBuffer output_keys{};
  rund::AccelBuffer output_values{};
  rund::AccelKernel kernel{};
};

struct Physical final {
  std::uint64_t dispatches = 0u;
  std::uint64_t submits = 0u;
  bool valid = false;
};

[[nodiscard]] constexpr rund::AccelBufferDesc
Buffer(const rund::BufferUsage usage, const std::uint64_t scalar_bytes,
       const std::uint64_t count) noexcept {
  return rund::AccelBufferDesc{
      .scalar_width_bytes = scalar_bytes,
      .count = count,
      .usage = usage,
  };
}

[[nodiscard]] constexpr Physical
ExpectedPhysical(const rund::AccelApi api) noexcept {
  switch (api) {
  case rund::AccelApi::Cpu:
    return Physical{.dispatches = 0u, .submits = 0u, .valid = true};
  case rund::AccelApi::Metal:
    return Physical{.dispatches = 17u, .submits = 1u, .valid = true};
  case rund::AccelApi::Vulkan:
    return Physical{.dispatches = 13u, .submits = 1u, .valid = true};
  default:
    return {};
  }
}

template <typename Count>
[[nodiscard]] constexpr rund::kernel::SortDesc Desc() noexcept {
  static_assert(sizeof(Count) == sizeof(rund::kernel::u32) ||
                sizeof(Count) == sizeof(rund::kernel::u64));
  return rund::kernel::SortDesc{
      .key = rund::kernel::SortKey::U32,
      .value = rund::kernel::SortValue::U32,
      .element_count = kCapacity,
      .radix_bits = 8u,
      .stable = true,
      .count_source = sizeof(Count) == sizeof(rund::kernel::u64)
                          ? rund::kernel::ComputeCountSource::BufferU64
                          : rund::kernel::ComputeCountSource::BufferU32,
  };
}

template <typename Count>
[[nodiscard]] Resources Build(const rund::AccelContext &context,
                              const Count logical_count) {
  Resources out{};
  out.context = context;
  out.keys = rund::node::accel::CreateAccelBuffer(
      context, Buffer(rund::BufferUsage::ReadOnly, sizeof(rund::kernel::u32),
                      kCapacity));
  out.count = rund::node::accel::CreateAccelBuffer(
      context, Buffer(rund::BufferUsage::ReadOnly, sizeof(Count), 1u));
  out.values = rund::node::accel::CreateAccelBuffer(
      context, Buffer(rund::BufferUsage::ReadOnly, sizeof(rund::kernel::u32),
                      kCapacity));
  out.output_keys = rund::node::accel::CreateAccelBuffer(
      context, Buffer(rund::BufferUsage::ReadWrite, sizeof(rund::kernel::u32),
                      kCapacity));
  out.output_values = rund::node::accel::CreateAccelBuffer(
      context, Buffer(rund::BufferUsage::ReadWrite, sizeof(rund::kernel::u32),
                      kCapacity));

  std::array<rund::kernel::u32, kCapacity> output_keys{};
  std::array<rund::kernel::u32, kCapacity> output_values{};
  output_keys.fill(kKeySentinel);
  output_values.fill(kValueSentinel);
  if (!out.keys.check.ok || !out.count.check.ok || !out.values.check.ok ||
      !out.output_keys.check.ok || !out.output_values.check.ok ||
      !rund::node::accel::UploadAccelBuffer(
           context, out.keys, kInputKeys.data(), sizeof(kInputKeys))
           .ok ||
      !rund::node::accel::UploadAccelBuffer(
           context, out.values, kInputValues.data(), sizeof(kInputValues))
           .ok ||
      !rund::node::accel::UploadAccelBuffer(context, out.count, &logical_count,
                                            sizeof(Count))
           .ok ||
      !rund::node::accel::UploadAccelBuffer(
           context, out.output_keys, output_keys.data(), sizeof(output_keys))
           .ok ||
      !rund::node::accel::UploadAccelBuffer(context, out.output_values,
                                            output_values.data(),
                                            sizeof(output_values))
           .ok) {
    return out;
  }

  std::array<rund::AccelGraphBufferRef, 5u> refs{
      rund::AccelGraphBufferRef{.buffer = &out.keys,
                                .role = rund::kernel::BufferRole::Read},
      rund::AccelGraphBufferRef{.buffer = &out.count,
                                .role = rund::kernel::BufferRole::Read},
      rund::AccelGraphBufferRef{.buffer = &out.values,
                                .role = rund::kernel::BufferRole::Read},
      rund::AccelGraphBufferRef{.buffer = &out.output_keys,
                                .role = rund::kernel::BufferRole::Write},
      rund::AccelGraphBufferRef{.buffer = &out.output_values,
                                .role = rund::kernel::BufferRole::Write},
  };
  constexpr rund::kernel::SortDesc desc = Desc<Count>();
  const std::array nodes{rund::AccelSort(refs.data(), refs.size(), desc)};
  out.kernel = rund::node::accel::CompileAccelKernel(
      context, rund::AccelGraph{
                   .nodes = nodes.data(),
                   .node_count = nodes.size(),
                   .scalar = rund::kernel::ComputeScalar::Lane32,
                   .domain = rund::kernel::ComputeDomain::U32,
               });
  return out;
}

[[nodiscard]] inline bool SparseOutputsMatch(const Resources &resources) {
  std::array<rund::kernel::u32, kCapacity> keys{};
  std::array<rund::kernel::u32, kCapacity> values{};
  if (!rund::node::accel::DownloadAccelBuffer(
           resources.context, resources.output_keys, keys.data(), sizeof(keys))
           .ok ||
      !rund::node::accel::DownloadAccelBuffer(resources.context,
                                              resources.output_values,
                                              values.data(), sizeof(values))
           .ok) {
    return false;
  }
  for (std::size_t index = 0u; index < kSparseCount; ++index) {
    if (keys[index] != kSortedKeys[index] ||
        values[index] != kSortedValues[index]) {
      return false;
    }
  }
  for (std::size_t index = kSparseCount; index < kCapacity; ++index) {
    if (keys[index] != kKeySentinel || values[index] != kValueSentinel) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] inline std::array<rund::AccelRunBinding, 5u>
Bindings(Resources &resources) noexcept {
  return {
      rund::AccelRunBinding{.buffer = &resources.keys,
                            .role = rund::kernel::BufferRole::Read},
      rund::AccelRunBinding{.buffer = &resources.count,
                            .role = rund::kernel::BufferRole::Read},
      rund::AccelRunBinding{.buffer = &resources.values,
                            .role = rund::kernel::BufferRole::Read},
      rund::AccelRunBinding{.buffer = &resources.output_keys,
                            .role = rund::kernel::BufferRole::Write},
      rund::AccelRunBinding{.buffer = &resources.output_values,
                            .role = rund::kernel::BufferRole::Write},
  };
}

[[nodiscard]] inline rund::AccelEvidence Run(Resources &resources) {
  auto bindings = Bindings(resources);
  return rund::node::accel::RunAccelKernel(
      resources.context, resources.kernel,
      rund::AccelRun{.bindings = bindings.data(),
                     .binding_count = bindings.size(),
                     .tile_count = kCapacity,
                     .fresh_evidence = true});
}

[[nodiscard]] inline bool OutputsUnchanged(const Resources &resources) {
  std::array<rund::kernel::u32, kCapacity> keys{};
  std::array<rund::kernel::u32, kCapacity> values{};
  if (!rund::node::accel::DownloadAccelBuffer(
           resources.context, resources.output_keys, keys.data(), sizeof(keys))
           .ok ||
      !rund::node::accel::DownloadAccelBuffer(resources.context,
                                              resources.output_values,
                                              values.data(), sizeof(values))
           .ok) {
    return false;
  }
  for (std::size_t index = 0u; index < kCapacity; ++index) {
    if (keys[index] != kKeySentinel || values[index] != kValueSentinel) {
      return false;
    }
  }
  return true;
}

template <typename Count>
[[nodiscard]] bool Rejects(const rund::AccelContext &context,
                           const Count logical_count) {
  Resources resources = Build(context, logical_count);
  const Physical expected = ExpectedPhysical(context.api);
  if (!resources.kernel.check.ok || !expected.valid) {
    return false;
  }
  const rund::AccelEvidence evidence = Run(resources);
  return !evidence.ok &&
         std::string_view{evidence.reason} == "compute_bounded_count_invalid" &&
         evidence.dispatch_count == expected.dispatches &&
         evidence.command_submit_count == expected.submits &&
         OutputsUnchanged(resources);
}

template <typename Count>
[[nodiscard]] bool RunsSparse(const rund::AccelContext &context) {
  Resources resources = Build(context, static_cast<Count>(kSparseCount));
  const Physical expected = ExpectedPhysical(context.api);
  if (!resources.kernel.check.ok || !expected.valid) {
    return false;
  }
  const rund::AccelEvidence evidence = Run(resources);
  return evidence.ok && evidence.dispatch_count == expected.dispatches &&
         evidence.command_submit_count == expected.submits &&
         SparseOutputsMatch(resources);
}

template <typename Count>
[[nodiscard]] bool RunsEmpty(const rund::AccelContext &context) {
  Resources resources = Build(context, Count{0u});
  const Physical expected = ExpectedPhysical(context.api);
  if (!resources.kernel.check.ok || !expected.valid) {
    return false;
  }
  const rund::AccelEvidence evidence = Run(resources);
  return evidence.ok && evidence.dispatch_count == expected.dispatches &&
         evidence.command_submit_count == expected.submits &&
         OutputsUnchanged(resources);
}

[[nodiscard]] inline bool Contract(const rund::AccelDevice &pick) {
  const Physical expected = ExpectedPhysical(pick.api);
  if (!pick.check.ok || !expected.valid) {
    return false;
  }
  const rund::AccelContext context = rund::node::accel::OpenAccel(pick);
  const bool native_indirect = context.api == rund::AccelApi::Cpu ||
                               (RunsEmpty<rund::kernel::u32>(context) &&
                                RunsEmpty<rund::kernel::u64>(context) &&
                                RunsSparse<rund::kernel::u32>(context) &&
                                RunsSparse<rund::kernel::u64>(context));
  return context.check.ok && native_indirect &&
         Rejects<rund::kernel::u32>(
             context, static_cast<rund::kernel::u32>(kCapacity + 1u)) &&
         Rejects<rund::kernel::u64>(context, 1ull << 32u);
}

} // namespace node_accel_contract::collective::bounded_sort
