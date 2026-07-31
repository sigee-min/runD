#include "control.hpp"
#include "lease.hpp"

#include "../../../hash/fnv.hpp"

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

#include "../barrier.hpp"
#include "../buffer/create/telemetry.hpp"
#include "../collective/pipeline.hpp"
#include "../descriptor.hpp"

#include <kernel/program/compute/artifact.hpp>
#include <kernel/program/compute/backend.hpp>

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <new>
#include <string>

namespace rund::node::accel::detail {
namespace {

inline constexpr std::uint32_t kControlThreads = 256u;

struct CanonicalParams final {
  std::uint32_t first{};
  std::uint32_t count{};
  std::uint32_t rule{};
  std::uint32_t success{};
  std::uint32_t mapping_count{};
  std::array<std::uint32_t, 8u> raw_values{};
  std::array<std::uint32_t, 8u> reasons{};
};

static_assert(sizeof(CanonicalParams) == 84u);

struct ControlParams final {
  std::uint32_t first{};
  std::uint32_t count{};
  std::uint32_t declared_step{};
  std::uint32_t phase{};
};

static_assert(sizeof(ControlParams) == 16u);

[[nodiscard]] std::uint64_t SourceHash(const std::string &source) noexcept {
  ::rund::node::hash_detail::Fnv hash{};
  for (const unsigned char byte : source) {
    hash.Byte(byte);
  }
  return hash.Finish();
}

[[nodiscard]] rund::kernel::ComputePlan
PipelinePseudoPlan(const std::string &source) noexcept {
  const std::uint64_t hash = SourceHash(source);
  return rund::kernel::ComputePlan{
      .op_hash_hi = hash,
      .op_hash_lo = hash ^ 0x9e3779b97f4a7c15ull,
      .api = rund::kernel::ComputeApi::Vulkan,
      .scalar = rund::kernel::ComputeScalar::Lane32,
      .ok = true,
      .reason = "ok",
  };
}

[[nodiscard]] rund::kernel::LoweringArtifact
PipelineArtifact(const std::string &source) {
  return rund::kernel::LoweringArtifact{
      .kind = rund::kernel::LoweringArtifactKind::VulkanSource,
      .source_text = source,
      .ok = true,
      .reason = "ok",
  };
}

[[nodiscard]] const std::string &CanonicalSource() {
  static const std::string source = [] {
    std::string text = R"GLSL(#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0, std430) readonly buffer RawStatus {
  uint raw_status[];
};
layout(set = 0, binding = 1, std430) buffer CanonicalStatus {
  uint canonical_status[];
};
layout(push_constant) uniform CanonicalParams {
  uint first;
  uint count;
  uint rule;
  uint success;
  uint mapping_count;
  uint raw_values[8];
  uint reasons[8];
} p;
void main() {
  const uint index = gl_GlobalInvocationID.x;
  if (index >= p.count) { return; }
  const uint raw = raw_status[index];
  uint reason = 0u;
  if (raw != p.success) {
    uint key = raw;
    if (p.rule == 3u) { key &= 1u; }
    reason = )GLSL";
    text += std::to_string(
        static_cast<std::uint32_t>(rund::compute::Reason::ReasonInvalid));
    text += R"GLSL(u;
    if (p.rule == 4u) { reason = p.reasons[0]; }
    for (uint map = 0u; map < p.mapping_count; ++map) {
      const bool match = p.rule == 2u
                             ? (raw & p.raw_values[map]) != 0u
                             : key == p.raw_values[map];
      if (p.rule != 4u && match) { reason = p.reasons[map]; break; }
    }
  }
  canonical_status[p.first + index] = reason;
}
)GLSL";
    return text;
  }();
  return source;
}

[[nodiscard]] std::string
ReduceSource(const PreparedPipelineStatusLayout &layout) {
  if (layout.status_entry_count == 0u) {
    std::string source = R"GLSL(#version 450
layout(local_size_x = 1) in;
layout(set = 0, binding = 0, std430) buffer ControlSummary {
  uint control[];
};
layout(push_constant) uniform Params {
  uint first;
  uint count;
  uint declared_step;
  uint phase;
} p;
void main() {
  if (p.phase == 2u) {
    control[1] = 0u;
    control[2] = 0xffffffffu;
    control[3] = 0u;
    for (uint index = 4u; index < 18u; ++index) { control[index] = 0u; }
    control[18] = 0xffffffffu;
    control[19] = 0xffffffffu;
    return;
  }
  if (p.phase != 1u) { return; }
  control[0] += )GLSL";
    source += std::to_string(layout.generation_stride);
    source += R"GLSL(u;
  if (control[1] == 0u) {
    control[2] = 0xffffffffu;
    control[3] = )GLSL";
    source += std::to_string(layout.declared_step_count);
    source += R"GLSL(u;
  } else {
    control[3] = control[2];
  }
}
)GLSL";
    return source;
  }
  std::string source = R"GLSL(#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0, std430) readonly buffer CanonicalStatus {
  uint status_values[];
};
layout(set = 0, binding = 1, std430) buffer ControlSummary {
  uint control[];
};
layout(push_constant) uniform Params {
  uint first;
  uint count;
  uint declared_step;
  uint phase;
} p;
shared uint ordinals[256];
shared uint reasons[256];
void main() {
  const uint lane = gl_LocalInvocationID.x;
  if (p.phase == 2u) {
    if (lane == 0u) {
      control[1] = 0u;
      control[2] = 0xffffffffu;
      control[3] = 0u;
      for (uint index = 4u; index < 18u; ++index) { control[index] = 0u; }
      control[18] = 0xffffffffu;
      control[19] = 0xffffffffu;
    }
    return;
  }
  if (p.phase == 1u) {
    if (lane == 0u) {
      control[0] += )GLSL";
  source += std::to_string(layout.generation_stride);
  source += R"GLSL(u;
      if (control[1] == 0u) {
        control[2] = 0xffffffffu;
        control[3] = )GLSL";
  source += std::to_string(layout.declared_step_count);
  source += R"GLSL(u;
      } else {
        control[3] = control[2];
      }
    }
    return;
  }
  if (control[1] != 0u) { return; }
  uint best = 0xffffffffu;
  uint reason = 0u;
)GLSL";
  source += R"GLSL(  for (uint index = lane; index < p.count; index += 256u) {
    const uint candidate = status_values[p.first + index];
    if (candidate != 0u && index < best) {
      best = index;
      reason = candidate;
    }
  }
)GLSL";
  source += R"GLSL(
  ordinals[lane] = best;
  reasons[lane] = reason;
  barrier();
  for (uint stride = 128u; stride != 0u; stride >>= 1u) {
    if (lane < stride && ordinals[lane + stride] < ordinals[lane]) {
      ordinals[lane] = ordinals[lane + stride];
      reasons[lane] = reasons[lane + stride];
    }
    barrier();
  }
  if (lane == 0u) {
    if (ordinals[0] != 0xffffffffu) {
      control[1] = reasons[0];
      control[2] = p.declared_step;
    }
)GLSL";
  source += "  }\n}\n";
  return source;
}

void ReleaseDescriptorLeases(VulkanPipelineControlResources &control) {
  for (const VulkanCollectiveDescriptorLease lease :
       control.descriptor_leases) {
    if (lease.pipeline != nullptr &&
        lease.slot < lease.pipeline->descriptor_leased.size()) {
      lease.pipeline->descriptor_leased[lease.slot] = false;
    }
  }
  control.descriptor_leases.clear();
}

[[nodiscard]] bool ValidSource(const VulkanPipelineStatusSource &source) {
  if (source.raw == nullptr || source.raw->buffer == VK_NULL_HANDLE ||
      source.count == 0u || source.mapping_count == 0u ||
      source.mapping_count > source.raw_values.size() ||
      (source.offset & (sizeof(std::uint32_t) - 1u)) != 0u ||
      source.offset > source.raw->bytes ||
      static_cast<std::uint64_t>(source.count) * sizeof(std::uint32_t) >
          source.raw->bytes - source.offset ||
      source.rule == VulkanPipelineStatusRule::None) {
    return false;
  }
  for (std::size_t index = 0u; index < source.mapping_count; ++index) {
    if (!CanonicalReasonStatus(source.reasons[index]) ||
        source.reasons[index] == 0u) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] PreparedMemory ExactMemory(const std::uint64_t bytes) noexcept {
  return PreparedMemory{
      .current = bytes, .peak = bytes, .cumulative = bytes, .budget = bytes};
}

} // namespace

rund::AccelCheck PrepareVulkanPipelineControl(
    VulkanAdapter &adapter,
    const std::span<VulkanPipelineCanonicalStatus> statuses,
    const PreparedPipelineStatusLayout &layout, const bool profile_steps,
    VulkanPipelineControlResources &control, PreparedPipelineMemory &memory) {
  control = {};
  control.adapter = &adapter;
  memory.device = {};
  memory.staging = {};
  const std::uint64_t arena_bytes =
      static_cast<std::uint64_t>(layout.status_entry_count) *
      sizeof(std::uint32_t);
  const std::uint64_t profile_bytes =
      profile_steps ? static_cast<std::uint64_t>(layout.declared_step_count) *
                          PreparedPipelineStepControlBytes
                    : 0u;
  if ((arena_bytes != 0u &&
       !CreateVulkanBuffer(adapter, arena_bytes,
                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, control.arena,
                           nullptr, VulkanMemoryUse::Device)) ||
      !CreateVulkanBuffer(adapter, sizeof(PreparedPipelineControl),
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                          control.summary) ||
      (profile_bytes != 0u &&
       !CreateVulkanBuffer(adapter, profile_bytes,
                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                               VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                           control.profile))) {
    DestroyVulkanPipelineControl(control);
    return rund::AccelCheck{false, VulkanLastError(&adapter)};
  }
  if (!ClearVulkanBuffer(control.summary, sizeof(PreparedPipelineControl))) {
    DestroyVulkanPipelineControl(control);
    return rund::AccelCheck{false, "accel_vulkan_memory_unavailable"};
  }
  control.profile_step_count = profile_steps ? layout.declared_step_count : 0u;

  try {
    control.descriptor_leases.reserve(statuses.size() + 1u);
  } catch (const std::bad_alloc &) {
    DestroyVulkanPipelineControl(control);
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  bool ready = false;
  try {
    VulkanLeaseScope lease_scope{adapter, control.descriptor_leases};
    const std::string &canonical_source = CanonicalSource();
    if (!statuses.empty()) {
      const auto canonical_plan = PipelinePseudoPlan(canonical_source);
      const auto canonical_artifact = PipelineArtifact(canonical_source);
      control.canonical_pipeline =
          AcquireVulkanCollectivePipeline(adapter, 2u, sizeof(CanonicalParams),
                                          canonical_plan, canonical_artifact);
    }
    const std::string reduce_source = ReduceSource(layout);
    const auto reduce_plan = PipelinePseudoPlan(reduce_source);
    const auto reduce_artifact = PipelineArtifact(reduce_source);
    const std::uint32_t reduce_descriptor_count =
        layout.status_entry_count == 0u ? 1u : 2u;
    control.reduce_pipeline = AcquireVulkanCollectivePipeline(
        adapter, reduce_descriptor_count, sizeof(ControlParams), reduce_plan,
        reduce_artifact);
    ready = control.reduce_pipeline != nullptr &&
            (statuses.empty() || control.canonical_pipeline != nullptr);
    std::uint32_t expected_first = 0u;
    for (std::size_t index = 0u; ready && index < statuses.size(); ++index) {
      VulkanPipelineCanonicalStatus &status = statuses[index];
      const std::uint64_t groups =
          (static_cast<std::uint64_t>(status.source.count) + kControlThreads -
           1u) /
          kControlThreads;
      ready = status.first == expected_first && ValidSource(status.source) &&
              status.source.count <=
                  std::numeric_limits<std::uint32_t>::max() - expected_first &&
              groups != 0u && groups <= adapter.max_dispatch_groups &&
              AcquireVulkanCollectiveDescriptorSet(
                  adapter, *control.canonical_pipeline, 2u, status.descriptor);
      if (ready) {
        const std::array<VulkanStorageBinding, 2u> buffers{
            VulkanStorageBinding{
                .buffer = status.source.raw,
                .offset = status.source.offset,
                .range = static_cast<VkDeviceSize>(status.source.count) *
                         sizeof(std::uint32_t)},
            VulkanStorageBindingFor(control.arena)};
        ready = WriteVulkanStorageDescriptorSet(adapter, status.descriptor,
                                                buffers);
        expected_first += status.source.count;
      }
    }
    ready = ready && expected_first == layout.status_entry_count;
    ready = ready && AcquireVulkanCollectiveDescriptorSet(
                         adapter, *control.reduce_pipeline,
                         reduce_descriptor_count, control.reduce_descriptor);
    if (ready) {
      if (layout.status_entry_count == 0u) {
        const std::array<const VulkanBuffer *, 1u> buffers{&control.summary};
        ready = WriteVulkanStorageDescriptorSet(
            adapter, control.reduce_descriptor, buffers);
      } else {
        const std::array<const VulkanBuffer *, 2u> buffers{&control.arena,
                                                           &control.summary};
        ready = WriteVulkanStorageDescriptorSet(
            adapter, control.reduce_descriptor, buffers);
      }
    }
  } catch (const std::bad_alloc &) {
    DestroyVulkanPipelineControl(control);
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  if (!ready) {
    const char *const reason = VulkanLastError(&adapter);
    DestroyVulkanPipelineControl(control);
    return rund::AccelCheck{false, reason == nullptr || reason[0] == '\0'
                                       ? "accel_vulkan_descriptor_unavailable"
                                       : reason};
  }
  memory.device = ExactMemory(control.arena.bytes);
  memory.staging = ExactMemory(control.summary.bytes + control.profile.bytes);
  return rund::AccelCheck{true, "ok"};
}

void DestroyVulkanPipelineControl(
    VulkanPipelineControlResources &control) noexcept {
  if (control.adapter != nullptr) {
    ReleaseDescriptorLeases(control);
    ReleaseVulkanBuffer(*control.adapter, control.arena);
    ReleaseVulkanBuffer(*control.adapter, control.summary);
    ReleaseVulkanBuffer(*control.adapter, control.profile);
  }
  control = {};
}

bool EncodeVulkanPipelineCanonicalStatus(
    const VkCommandBuffer command,
    const VulkanPipelineControlResources &control,
    const VulkanPipelineCanonicalStatus &status) noexcept {
  if (command == VK_NULL_HANDLE || control.canonical_pipeline == nullptr ||
      status.descriptor == VK_NULL_HANDLE) {
    return false;
  }
  const CanonicalParams params{
      .first = status.first,
      .count = status.source.count,
      .rule = static_cast<std::uint32_t>(status.source.rule),
      .success = status.source.success,
      .mapping_count = status.source.mapping_count,
      .raw_values = status.source.raw_values,
      .reasons = status.source.reasons,
  };
  BindVulkanPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                     control.canonical_pipeline->pipeline);
  BindVulkanDescriptors(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                        control.canonical_pipeline->pipeline_layout, 0u, 1u,
                        &status.descriptor, 0u, nullptr);
  PushVulkanConstants(command, control.canonical_pipeline->pipeline_layout,
                      VK_SHADER_STAGE_COMPUTE_BIT, 0u, sizeof(params), &params);
  const std::uint32_t groups =
      (status.source.count + kControlThreads - 1u) / kControlThreads;
  DispatchVulkan(command, groups, 1u, 1u);
  return true;
}

namespace {

[[nodiscard]] bool
EncodeVulkanPipelineControl(const VkCommandBuffer command,
                            const VulkanPipelineControlResources &control,
                            const ControlParams params) noexcept {
  if (command == VK_NULL_HANDLE || control.reduce_pipeline == nullptr ||
      control.reduce_descriptor == VK_NULL_HANDLE ||
      control.summary.buffer == VK_NULL_HANDLE ||
      (params.phase == 0u &&
       (params.count == 0u || control.arena.buffer == VK_NULL_HANDLE))) {
    return false;
  }
  std::array<VkBufferMemoryBarrier, 2u> inputs{};
  std::uint32_t input_count = 0u;
  if (control.arena.buffer != VK_NULL_HANDLE) {
    inputs[input_count++] = VulkanBufferBarrier(
        control.arena, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
  }
  inputs[input_count++] = VulkanBufferBarrier(
      control.summary, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT,
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
  vkCmdPipelineBarrier(command,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                           VK_PIPELINE_STAGE_HOST_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0u, 0u, nullptr,
                       input_count, inputs.data(), 0u, nullptr);
  BindVulkanPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                     control.reduce_pipeline->pipeline);
  BindVulkanDescriptors(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                        control.reduce_pipeline->pipeline_layout, 0u, 1u,
                        &control.reduce_descriptor, 0u, nullptr);
  PushVulkanConstants(command, control.reduce_pipeline->pipeline_layout,
                      VK_SHADER_STAGE_COMPUTE_BIT, 0u, sizeof(params), &params);
  DispatchVulkan(command, 1u, 1u, 1u);
  return true;
}

} // namespace

bool FoldVulkanPipelineControl(const VkCommandBuffer command,
                               const VulkanPipelineControlResources &control,
                               const PreparedProgramStatusSlice slice,
                               const std::uint32_t declared_step) noexcept {
  return EncodeVulkanPipelineControl(
      command, control,
      ControlParams{.first = slice.first,
                    .count = slice.count,
                    .declared_step = declared_step});
}

bool OpenVulkanPipelineControl(
    const VkCommandBuffer command,
    const VulkanPipelineControlResources &control) noexcept {
  return EncodeVulkanPipelineControl(command, control,
                                     ControlParams{.phase = 2u});
}

bool FinishVulkanPipelineControl(
    const VkCommandBuffer command,
    const VulkanPipelineControlResources &control,
    const PreparedPipelineStatusLayout &) noexcept {
  return EncodeVulkanPipelineControl(command, control,
                                     ControlParams{.phase = 1u});
}

bool ResetVulkanPipelineProfile(
    const VkCommandBuffer command,
    const VulkanPipelineControlResources &control) noexcept {
  if (control.profile_step_count == 0u) {
    return control.profile.buffer == VK_NULL_HANDLE;
  }
  const VkDeviceSize field_bytes =
      static_cast<VkDeviceSize>(control.profile_step_count) *
      sizeof(std::uint64_t);
  const VkDeviceSize zero_bytes = 7u * field_bytes;
  const VkDeviceSize overflow_bytes = field_bytes;
  if (command == VK_NULL_HANDLE || control.profile.buffer == VK_NULL_HANDLE ||
      control.profile.bytes < zero_bytes + overflow_bytes) {
    return false;
  }
  vkCmdFillBuffer(command, control.profile.buffer, 0u, zero_bytes, 0u);
  vkCmdFillBuffer(command, control.profile.buffer, zero_bytes, overflow_bytes,
                  std::numeric_limits<std::uint32_t>::max());
  const VkBufferMemoryBarrier visible = VulkanBufferBarrier(
      control.profile, VK_ACCESS_TRANSFER_WRITE_BIT,
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
  vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0u, 0u, nullptr,
                       1u, &visible, 0u, nullptr);
  return true;
}

bool PublishVulkanPipelineControl(
    const VkCommandBuffer command,
    const VulkanPipelineControlResources &control) noexcept {
  if (command == VK_NULL_HANDLE || control.summary.buffer == VK_NULL_HANDLE) {
    return false;
  }
  std::array<VkBufferMemoryBarrier, 2u> visible{};
  std::uint32_t visible_count = 0u;
  visible[visible_count++] = VulkanBufferBarrier(
      control.summary, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
  if (control.profile.buffer != VK_NULL_HANDLE) {
    visible[visible_count++] = VulkanBufferBarrier(
        control.profile,
        VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT,
        VK_ACCESS_HOST_READ_BIT);
  }
  const VkPipelineStageFlags source_stage =
      control.profile.buffer == VK_NULL_HANDLE
          ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
          : VK_PIPELINE_STAGE_TRANSFER_BIT |
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
  vkCmdPipelineBarrier(command, source_stage, VK_PIPELINE_STAGE_HOST_BIT, 0u,
                       0u, nullptr, visible_count, visible.data(), 0u, nullptr);
  return true;
}

bool ReadVulkanPipelineControl(const VulkanPipelineControlResources &resources,
                               PreparedPipelineControl &control) noexcept {
  control = {};
  if (resources.summary.mapped == nullptr ||
      resources.summary.bytes < sizeof(control)) {
    return false;
  }
  std::memcpy(&control, resources.summary.mapped, sizeof(control));
  return true;
}

bool ReadVulkanPipelineProfile(
    const VulkanPipelineControlResources &resources,
    const std::span<PreparedPipelineStepControl> controls) noexcept {
  const std::size_t step_count = resources.profile_step_count;
  const std::uint64_t required_bytes =
      static_cast<std::uint64_t>(step_count) * PreparedPipelineStepControlBytes;
  if (step_count == 0u) {
    return resources.profile.buffer == VK_NULL_HANDLE && controls.empty();
  }
  if (resources.profile.mapped == nullptr ||
      resources.profile.bytes < required_bytes ||
      controls.size() < step_count) {
    return false;
  }
  const auto *const bytes =
      static_cast<const std::uint8_t *>(resources.profile.mapped);
  const auto read = [bytes, step_count](const std::size_t field,
                                        const std::size_t step) noexcept {
    std::uint64_t value{};
    const std::size_t offset =
        (field * step_count + step) * sizeof(std::uint64_t);
    std::memcpy(&value, bytes + offset, sizeof(value));
    return value;
  };
  for (std::size_t step = 0u; step < step_count; ++step) {
    controls[step] = PreparedPipelineStepControl{
        .generated_item_count = read(0u, step),
        .generated_capacity = read(1u, step),
        .indirect_dispatch_count = read(2u, step),
        .indirect_work_item_count = read(3u, step),
        .iteration_count = read(4u, step),
        .skipped_iteration_count = read(5u, step),
        .conflict_count = read(6u, step),
        .overflow_ordinal = read(7u, step),
    };
  }
  return true;
}

} // namespace rund::node::accel::detail

#endif
