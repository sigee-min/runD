#include "status.hpp"

#include "../../pipeline/named.hpp"
#include "source.hpp"

#include <algorithm>
#include <limits>
#include <string>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
[[nodiscard]] bool to_u32(const std::uint64_t value,
                          std::uint32_t &out) noexcept {
  if (value > std::numeric_limits<std::uint32_t>::max()) {
    return false;
  }
  out = static_cast<std::uint32_t>(value);
  return true;
}

[[nodiscard]] bool
CollectMetalStatus(MetalKernelResources &resources,
                   const std::uint32_t declared_step,
                   std::vector<MetalPipelineStatusBindingRecord> &bindings,
                   std::vector<MetalPipelineStatusSourceMeta> &sources,
                   std::uint32_t &raw_count, std::uint32_t &entry_count) {
  const std::size_t program_binding_begin = bindings.size();
  for (std::size_t index = 0u; index < resources.size(); ++index) {
    MetalKernelEntry *const entry = resources.entry(index);
    if (entry == nullptr) {
      return false;
    }
    if (entry->ops.pipeline_status == nullptr) {
      continue;
    }
    MetalPipelineStatusBindings described{};
    if (!entry->ops.pipeline_status(entry->resource, described) ||
        described.size > described.values.size()) {
      return false;
    }
    for (std::size_t binding_index = 0u; binding_index < described.size;
         ++binding_index) {
      const MetalPipelineStatusBinding &binding =
          described.values[binding_index];
      std::uint32_t words = 0u;
      if (binding.buffer == nullptr || binding.bytes == 0u ||
          (binding.bytes & (sizeof(std::uint32_t) - 1u)) != 0u ||
          binding.offset >
              std::numeric_limits<std::uint64_t>::max() - binding.bytes ||
          binding.offset > std::numeric_limits<NSUInteger>::max() ||
          binding.bytes > std::numeric_limits<NSUInteger>::max() ||
          !to_u32(binding.bytes / sizeof(std::uint32_t), words) ||
          binding.observed_count == 0u || binding.observed >= words ||
          binding.observed_count > words - binding.observed ||
          words > std::numeric_limits<std::uint32_t>::max() - raw_count ||
          sources.size() >= std::numeric_limits<std::uint32_t>::max() ||
          binding.observed_count >
              std::numeric_limits<std::uint32_t>::max() - entry_count) {
        return false;
      }
      for (std::size_t prior = program_binding_begin; prior < bindings.size();
           ++prior) {
        const MetalPipelineStatusBinding &other = bindings[prior].binding;
        if (other.buffer == binding.buffer &&
            binding.offset < other.offset + other.bytes &&
            other.offset < binding.offset + binding.bytes) {
          return false;
        }
      }
      MetalPipelineStatusSourceMeta source{
          .encoding = static_cast<std::uint32_t>(binding.encoding),
          .declared_step = declared_step,
          .limit_low = static_cast<std::uint32_t>(binding.limit),
          .limit_high = static_cast<std::uint32_t>(binding.limit >> 32u),
          .telemetry = static_cast<std::uint32_t>(binding.telemetry),
          .indirect_dispatch_count = binding.indirect_dispatch_count,
          .work_item_count_low =
              static_cast<std::uint32_t>(binding.work_item_count),
          .work_item_count_high =
              static_cast<std::uint32_t>(binding.work_item_count >> 32u),
      };
      constexpr std::uint32_t field_max =
          std::numeric_limits<std::uint16_t>::max();
      for (std::size_t reason_index = 0u; reason_index < source.policies.size();
           ++reason_index) {
        const std::uint32_t reason = binding.reasons[reason_index];
        const std::uint32_t priority =
            reason == 0u ? field_max : CanonicalReasonPriority(reason);
        if (reason > field_max || priority > field_max) {
          return false;
        }
        source.policies[reason_index] = reason | (priority << 16u);
      }
      sources.push_back(source);
      bindings.push_back(MetalPipelineStatusBindingRecord{
          .binding = binding,
          .raw_offset = raw_count,
          .raw_count = words,
      });
      raw_count += words;
      entry_count += binding.observed_count;
    }
  }
  return true;
}

[[nodiscard]] bool
ValidMetalReset(const std::span<const MetalPipelineResetMeta> resets,
                const std::uint32_t raw_count) noexcept {
  if (raw_count == 0u) {
    return resets.empty();
  }
  if (resets.empty() || resets.front().raw_offset != 0u ||
      resets.back().raw_offset >= raw_count) {
    return false;
  }
  for (std::size_t index = 1u; index < resets.size(); ++index) {
    if (resets[index - 1u].raw_offset >= resets[index].raw_offset) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool PrepareMetalStatus(
    MetalAdapter &adapter, const bool need_status, const bool need_reset,
    const bool need_import, const bool need_telemetry, const bool profile_steps,
    std::shared_ptr<void> &reset, std::shared_ptr<void> &import,
    std::shared_ptr<void> &reduce, std::shared_ptr<void> &complete,
    std::shared_ptr<void> &telemetry, const bool need_publish,
    std::shared_ptr<void> &publish, const bool need_advance,
    std::shared_ptr<void> &advance) {
  const char *const reduce_key =
      profile_steps ? "pipeline.status.fold.profile" : "pipeline.status.fold";
  const char *const reduce_function =
      profile_steps ? "rund_pipeline_status_reduce_profiled"
                    : "rund_pipeline_status_reduce";
  const char *const complete_key =
      profile_steps ? "pipeline.status.close.profile" : "pipeline.status.close";
  const char *const complete_function =
      profile_steps ? "rund_pipeline_status_complete_profiled"
                    : "rund_pipeline_status_complete";
  const char *const telemetry_key = profile_steps
                                        ? "pipeline.telemetry.fold.profile"
                                        : "pipeline.telemetry.fold";
  const char *const telemetry_function =
      profile_steps ? "rund_pipeline_telemetry_accumulate_profiled"
                    : "rund_pipeline_telemetry_accumulate";
  if (need_reset) {
    reset = LookupMetalNamedPipeline(adapter, "pipeline.status.reset");
  }
  if (need_status) {
    reduce = LookupMetalNamedPipeline(adapter, reduce_key);
  }
  complete = LookupMetalNamedPipeline(adapter, complete_key);
  if (need_import) {
    import = LookupMetalNamedPipeline(adapter, "pipeline.status.import");
  }
  if (need_telemetry) {
    telemetry = LookupMetalNamedPipeline(adapter, telemetry_key);
  }
  if (need_publish) {
    publish = LookupMetalNamedPipeline(adapter, "pipeline.publish");
  }
  if (need_advance) {
    advance = LookupMetalNamedPipeline(adapter, "pipeline.advance");
  }
  const bool make_reset = need_reset && reset == nullptr;
  const bool make_import = need_import && import == nullptr;
  const bool make_reduce = need_status && reduce == nullptr;
  const bool make_complete = complete == nullptr;
  const bool make_telemetry = need_telemetry && telemetry == nullptr;
  const bool make_publish = need_publish && publish == nullptr;
  const bool make_advance = need_advance && advance == nullptr;
  if (!make_reset && !make_import && !make_reduce && !make_complete &&
      !make_telemetry && !make_publish && !make_advance) {
    return true;
  }
  id<MTLDevice> const device = (__bridge id<MTLDevice>)adapter.device.get();
  const std::uint64_t begin = MonotonicNanoseconds();
  const std::shared_ptr<void> library_owner =
      AcquireMetalLibrary(adapter, MetalPipelineSource());
  id<MTLLibrary> const library = (__bridge id<MTLLibrary>)library_owner.get();
  if (library == nil ||
      (make_reset &&
       !MakeNamedMetalPipeline(device, library, "rund_pipeline_status_reset",
                               reset)) ||
      (make_import &&
       !MakeNamedMetalPipeline(device, library, "rund_pipeline_status_import",
                               import)) ||
      (make_reduce &&
       !MakeNamedMetalPipeline(device, library, reduce_function, reduce)) ||
      (make_complete &&
       !MakeNamedMetalPipeline(device, library, complete_function, complete)) ||
      (make_telemetry && !MakeNamedMetalPipeline(
                             device, library, telemetry_function, telemetry)) ||
      (make_publish &&
       !MakeNamedMetalPipeline(device, library, "rund_pipeline_publish",
                               publish)) ||
      (make_advance &&
       !MakeNamedMetalPipeline(device, library, "rund_pipeline_advance",
                               advance))) {
    return false;
  }
  const std::uint64_t elapsed = MonotonicNanoseconds() - begin;
  bool accounted = false;
  const auto store = [&](const bool created, const char *const name,
                         const std::shared_ptr<void> &pipeline) {
    if (!created) {
      return;
    }
    StoreMetalNamedPipeline(adapter, name, pipeline, accounted ? 0u : elapsed);
    accounted = true;
  };
  store(make_reset, "pipeline.status.reset", reset);
  store(make_import, "pipeline.status.import", import);
  store(make_reduce, reduce_key, reduce);
  store(make_complete, complete_key, complete);
  store(make_telemetry, telemetry_key, telemetry);
  store(make_publish, "pipeline.publish", publish);
  store(make_advance, "pipeline.advance", advance);
  return true;
}

#endif

} // namespace rund::node::accel::detail
