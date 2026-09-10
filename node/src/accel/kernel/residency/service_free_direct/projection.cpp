#include "projection.hpp"

#include "../../recurrence.hpp"

#include <kernel/core/checked.hpp>

#include <algorithm>
#include <limits>
#include <new>

namespace rund::node::accel::detail {
namespace {

[[nodiscard]] ServiceFreeDirectIdentity
identity(const std::uint64_t pipeline_hi, const std::uint64_t pipeline_lo,
         const MapRecurrence &recurrence) noexcept {
  std::uint64_t hi = pipeline_hi ^ 0x736572766963652dull;
  std::uint64_t lo = pipeline_lo ^ 0x667265652d646972ull;
  const auto mix = [](std::uint64_t &hash, const std::uint64_t value) {
    hash ^= value + 0x9e3779b97f4a7c15ull + (hash << 6u) + (hash >> 2u);
  };
  mix(hi, recurrence.iterations);
  mix(lo, recurrence.plan.op_hash_hi);
  mix(hi, recurrence.plan.op_hash_lo);
  mix(lo, recurrence.bindings.resident_inputs.count);
  mix(hi, recurrence.bindings.resident_outputs.count);
  if (hi == 0u && lo == 0u) {
    lo = 1u;
  }
  return ServiceFreeDirectIdentity{.hi = hi, .lo = lo};
}

[[nodiscard]] bool copy_range(
    const rund::kernel::ResidentBindingRange &source,
    std::array<rund::kernel::ResidentBufferRef,
               ServiceFreeDirectProof::BindingCapacity> &refs,
    std::array<std::shared_ptr<void>, ServiceFreeDirectProof::BindingCapacity>
        &handles) noexcept {
  if (source.count == 0u ||
      source.count > ServiceFreeDirectProof::BindingCapacity ||
      !source.has_refs() || !source.has_handles()) {
    return false;
  }
  for (std::uint64_t index = 0u; index < source.count; ++index) {
    const rund::kernel::ResidentBufferRef *const ref = source.ref(index);
    const std::shared_ptr<void> *const handle = source.handle(index);
    if (ref == nullptr || handle == nullptr || *handle == nullptr) {
      return false;
    }
    refs[static_cast<std::size_t>(index)] = *ref;
    handles[static_cast<std::size_t>(index)] = *handle;
  }
  return true;
}

[[nodiscard]] bool
valid_view(const rund::kernel::ResidentBufferRef &view) noexcept {
  std::uint64_t last_offset = 0u;
  std::uint64_t end = 0u;
  return view.id != 0u && view.bytes != 0u && view.count != 0u &&
         view.element_bytes != 0u && view.stride_bytes >= view.element_bytes &&
         (view.usage == rund::kernel::kResidentUsageRead ||
          view.usage == rund::kernel::kResidentUsageWrite) &&
         view.offset_bytes <= view.bytes &&
         rund::kernel::checked::mul(view.count - 1u, view.stride_bytes,
                                    last_offset) &&
         rund::kernel::checked::add(view.offset_bytes, last_offset, end) &&
         rund::kernel::checked::add(end, view.element_bytes, end) &&
         end <= view.bytes;
}

[[nodiscard]] bool append_state(ServiceFreeDirectProof &proof,
                                const rund::kernel::ResidentBufferRef &view,
                                const std::shared_ptr<void> &handle) noexcept {
  if (!valid_view(view) || handle == nullptr) {
    return false;
  }
  for (std::size_t index = 0u; index < proof.state_count; ++index) {
    rund::kernel::ResidentBufferRef &state = proof.states[index];
    if (state.id == view.id) {
      if (state.bytes != view.bytes || proof.state_handles[index] != handle) {
        return false;
      }
      state.usage |= view.usage;
      return true;
    }
    if (proof.state_handles[index] == handle) {
      return false;
    }
  }
  if (proof.state_count >= proof.StateCapacity) {
    return false;
  }
  const std::size_t index = proof.state_count++;
  proof.states[index] = rund::kernel::ResidentBufferRef{
      .id = view.id,
      .bytes = view.bytes,
      .offset_bytes = 0u,
      .element_bytes = 1u,
      .stride_bytes = 1u,
      .count = view.bytes,
      .usage = view.usage,
  };
  proof.state_handles[index] = handle;
  return true;
}

[[nodiscard]] bool
collect_states(ServiceFreeDirectProof &proof,
               const std::span<const BackendBatchEntry> entries) noexcept {
  if (entries.size() != proof.iterations) {
    return false;
  }
  for (const BackendBatchEntry &entry : entries) {
    if (entry.run == nullptr || entry.run->steps == nullptr ||
        entry.run->step_count != 1u) {
      return false;
    }
    const rund::kernel::BindingSet bindings =
        MapBindingFor(entry.run->steps[0]);
    if (!bindings.ok || !bindings.resident_inputs.has_refs() ||
        !bindings.resident_inputs.has_handles() ||
        !bindings.resident_outputs.has_refs() ||
        !bindings.resident_outputs.has_handles()) {
      return false;
    }
    for (const rund::kernel::ResidentBindingRange range :
         {bindings.resident_inputs, bindings.resident_outputs}) {
      for (std::uint64_t position = 0u; position < range.count; ++position) {
        const rund::kernel::ResidentBufferRef *const view = range.ref(position);
        const std::shared_ptr<void> *const handle = range.handle(position);
        if (view == nullptr || handle == nullptr ||
            !append_state(proof, *view, *handle)) {
          return false;
        }
      }
    }
  }
  return proof.state_count != 0u;
}

} // namespace

ServiceFreeDirectProjection ProjectServiceFreeDirectProof(
    const MapRecurrence &recurrence,
    const std::span<const BackendBatchEntry> entries,
    std::shared_ptr<const void> semantic_owner,
    const std::uint64_t pipeline_fingerprint_hi,
    const std::uint64_t pipeline_fingerprint_lo) noexcept {
  ServiceFreeDirectProjection result{};
  const std::uint64_t input_count = recurrence.bindings.resident_inputs.count;
  const std::uint64_t output_count = recurrence.bindings.resident_outputs.count;
  if (!recurrence.ready() || semantic_owner == nullptr ||
      recurrence.canonical_artifact == nullptr ||
      recurrence.windows == nullptr || recurrence.window_count == 0u ||
      recurrence.iterations < 2u ||
      recurrence.iterations > std::numeric_limits<std::uint32_t>::max() ||
      input_count > std::numeric_limits<std::uint32_t>::max() ||
      output_count > std::numeric_limits<std::uint32_t>::max() ||
      recurrence.bindings.param_data_bytes != recurrence.bindings.param_bytes ||
      (recurrence.bindings.param_bytes != 0u &&
       recurrence.bindings.param_data == nullptr)) {
    return result;
  }

  try {
    auto proof = std::make_shared<ServiceFreeDirectProof>();
    proof->identity =
        identity(pipeline_fingerprint_hi, pipeline_fingerprint_lo, recurrence);
    proof->semantic_owner = std::move(semantic_owner);
    proof->artifact = recurrence.canonical_artifact;
    proof->plan = recurrence.plan;
    if (!copy_range(recurrence.bindings.resident_inputs, proof->inputs,
                    proof->input_handles) ||
        !copy_range(recurrence.bindings.resident_outputs, proof->outputs,
                    proof->output_handles)) {
      return result;
    }
    proof->windows = recurrence.windows;
    proof->parameters =
        static_cast<const std::byte *>(recurrence.bindings.param_data);
    proof->iterations = recurrence.iterations;
    proof->window_count = recurrence.window_count;
    proof->parameter_bytes = recurrence.bindings.param_bytes;
    proof->input_count = static_cast<std::uint32_t>(input_count);
    proof->output_count = static_cast<std::uint32_t>(output_count);
    if (!collect_states(*proof, entries)) {
      return result;
    }
    proof->fixed_common_storage = true;
    proof->retention = recurrence.writes_each_iteration()
                           ? ServiceFreeDirectRetention::History
                           : ServiceFreeDirectRetention::Terminal;
    if (recurrence.history != nullptr) {
      if (recurrence.history->count != output_count) {
        return result;
      }
      for (std::uint64_t index = 0u; index < output_count; ++index) {
        proof->output_pitch_bytes[static_cast<std::size_t>(index)] =
            recurrence.history->pitch_bytes[static_cast<std::size_t>(index)];
      }
    }
    result.proof = std::move(proof);
    result.check = rund::AccelCheck{true, "ok"};
    return result;
  } catch (const std::bad_alloc &) {
    result.check.reason = "compute_pipeline_capacity";
    return result;
  }
}

} // namespace rund::node::accel::detail
