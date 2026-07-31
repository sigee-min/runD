#include "resource.hpp"

#include "../barrier.hpp"
#include "../buffer/transfer/copy.hpp"
#include "../scope.hpp"
#include "../status.hpp"

#include <kernel/program/compute/transform/twiddle.hpp>

#include <limits>
#include <string>
#include <utility>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
[[nodiscard]] rund::kernel::LoweringArtifact
NumericArtifact(const std::string &source) noexcept {
  return rund::kernel::LoweringArtifact{
      .kind = rund::kernel::LoweringArtifactKind::VulkanSource,
      .source_text = source,
      .ok = true,
      .reason = "ok",
  };
}

[[nodiscard]] VulkanCollectivePipeline *AcquireNumericPipeline(
    VulkanAdapter &adapter, const std::uint32_t descriptor_count,
    const std::uint32_t push_bytes, const rund::kernel::ComputePlan &pseudo,
    const std::string &source, const NumericPolicy policy) {
  const rund::kernel::LoweringArtifact artifact = NumericArtifact(source);
  const VulkanSpecialization specialization{
      .values = policy.constants(),
      .count = 4u,
  };
  return AcquireVulkanCollectivePipeline(adapter, descriptor_count, push_bytes,
                                         pseudo, artifact, specialization);
}

[[nodiscard]] bool MakeParamsBuffer(VulkanAdapter &adapter,
                                    const NumericParams &params,
                                    VulkanBuffer &out) {
  return CreateVulkanBuffer(adapter, sizeof(params),
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, out) &&
         UploadVulkanBuffer(out, &params, sizeof(params));
}

[[nodiscard]] bool MakeDummyBuffer(VulkanAdapter &adapter, VulkanBuffer &out) {
  const rund::kernel::u32 zero = 0u;
  return CreateVulkanBuffer(adapter, sizeof(zero),
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, out) &&
         UploadVulkanBuffer(out, &zero, sizeof(zero));
}

[[nodiscard]] rund::AccelCheck StatusCheck(VulkanAdapter &adapter,
                                           const VulkanBuffer &status,
                                           const rund::kernel::u64 count) {
  const auto *const data =
      static_cast<const rund::kernel::u32 *>(status.mapped);
  if (data == nullptr) {
    SetVulkanLastError(adapter, "accel_vulkan_memory_unavailable");
    return rund::AccelCheck{false, "accel_vulkan_memory_unavailable"};
  }
  rund::AccelCheck check{true, "ok"};
  for (rund::kernel::u64 index = 0u; index < count; ++index) {
    if (data[index] == 0u) {
      continue;
    }
    if (check.failed_batches == 0u) {
      check.first_failed_batch = index;
      check.first_status = data[index];
    }
    ++check.failed_batches;
  }
  return check;
}

[[nodiscard]] rund::AccelCheck RejectElement(VulkanAdapter &adapter,
                                             const char *const reason) {
  SetVulkanLastError(adapter, reason);
  return rund::AccelCheck{false, reason};
}

[[nodiscard]] VulkanStorageBinding
ResidentBinding(const VulkanResidentBufferResult &resident) noexcept {
  return VulkanStorageBindingFor(resident.device_buffer, resident.ref);
}

void DestroyNumericPrepared(void *const raw) {
  auto *const state = static_cast<VulkanNumericPrepared *>(raw);
  if (state == nullptr) {
    return;
  }
  if (state->adapter != nullptr) {
    ReleaseVulkanBuffer(*state->adapter, state->params);
    ReleaseVulkanBuffer(*state->adapter, state->dummy);
    ReleaseVulkanBuffer(*state->adapter, state->twiddle);
    ReleaseVulkanBuffer(*state->adapter, state->status_readback);
  }
  delete state;
}

[[nodiscard]] bool PrepareTwiddle(VulkanNumericPrepared &state,
                                  const rund::kernel::TransformPlan &plan) {
  if (plan.workspace_bytes == 0u) {
    return true;
  }
  if (!CreateVulkanBuffer(*state.adapter, plan.workspace_bytes,
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                              VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                          state.twiddle, nullptr, VulkanMemoryUse::Device)) {
    return false;
  }
  VulkanBuffer upload_raw{};
  if (!CreateVulkanBuffer(*state.adapter, plan.workspace_bytes,
                          VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                              VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                          upload_raw)) {
    return false;
  }
  ScopedBuffer upload{*state.adapter, upload_raw, plan.workspace_bytes};
  const bool filled =
      plan.element_bytes == sizeof(rund::kernel::i64)
          ? rund::kernel::transform_twiddle::Fill(
                static_cast<rund::kernel::i64 *>(upload.buffer.mapped),
                plan.element_count, plan.direction, plan.fixed_format)
          : rund::kernel::transform_twiddle::Fill(
                static_cast<rund::kernel::i32 *>(upload.buffer.mapped),
                plan.element_count, plan.direction, plan.fixed_format);
  return filled && UploadVulkanCopy(*state.adapter, std::move(upload),
                                    state.twiddle, 0u, plan.workspace_bytes);
}

[[nodiscard]] rund::AccelCheck
LookupPrepared(const rund::AccelDevice &pick, VulkanNumericPrepared &state,
               const std::span<VulkanResidentReq> requests) {
  LookupVulkanResidentBatch(pick, requests.data(), requests.size(),
                            "accel_vulkan_resident_id_unavailable");
  for (std::size_t index = 0u; index < requests.size(); ++index) {
    if (!state.resident[index].check.ok) {
      return state.resident[index].check;
    }
  }
  return rund::AccelCheck{true, "ok"};
}

[[nodiscard]] rund::AccelCheck
FinalizePrepared(VulkanNumericPrepared &state, const NumericParams &params,
                 const std::uint32_t descriptor_count,
                 const rund::kernel::ComputePlan &pseudo,
                 const std::string &source, const NumericPolicy policy,
                 const KernelPreparationMode mode,
                 const std::uint32_t push_bytes) {
  if (state.status_count > std::numeric_limits<rund::kernel::u64>::max() /
                               sizeof(rund::kernel::u32) ||
      (state.status_count != 0u && state.status == nullptr)) {
    return rund::AccelCheck{false, "compute_plan_invalid"};
  }
  const rund::kernel::u64 status_bytes =
      state.status_count * sizeof(rund::kernel::u32);
  if (status_bytes != 0u && (state.status_binding.buffer != state.status ||
                             status_bytes > state.status_binding.range)) {
    return rund::AccelCheck{false, "compute_plan_invalid"};
  }
  state.pipeline_private = IsPipelinePrivatePreparation(mode);
  if (status_bytes != 0u && !state.pipeline_private &&
      !CreateVulkanBuffer(*state.adapter, status_bytes,
                          VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                          state.status_readback)) {
    return rund::AccelCheck{false, VulkanLastError(state.adapter)};
  }
  state.pipeline = AcquireNumericPipeline(*state.adapter, descriptor_count,
                                          push_bytes, pseudo, source, policy);
  if (state.pipeline == nullptr ||
      !MakeParamsBuffer(*state.adapter, params, state.params) ||
      !MakeDummyBuffer(*state.adapter, state.dummy) ||
      !AcquireVulkanCollectiveDescriptorSet(*state.adapter, *state.pipeline,
                                            descriptor_count,
                                            state.descriptor)) {
    return rund::AccelCheck{false, VulkanLastError(state.adapter)};
  }
  state.bindings[0] = VulkanStorageBindingFor(state.params);
  for (std::size_t index = 1u; index < state.binding_count; ++index) {
    if (state.bindings[index].buffer == &state.dummy) {
      state.bindings[index] = VulkanStorageBindingFor(state.dummy);
    } else if (state.bindings[index].buffer == &state.twiddle) {
      state.bindings[index] = VulkanStorageBindingFor(state.twiddle);
    }
    if (state.bindings[index].buffer == nullptr ||
        state.bindings[index].range == 0u) {
      return rund::AccelCheck{false, "compute_plan_invalid"};
    }
  }
  return WriteVulkanStorageDescriptorSet(
             *state.adapter, state.descriptor, state.bindings.data(),
             static_cast<std::uint32_t>(state.binding_count))
             ? rund::AccelCheck{true, "ok"}
             : rund::AccelCheck{false, VulkanLastError(state.adapter)};
}
#endif

} // namespace rund::node::accel::detail
