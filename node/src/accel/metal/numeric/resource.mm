#include "resource.hpp"

#include "../adapter.hpp"
#include "../numeric/source.hpp"
#include "../object.hpp"
#include "../pipeline/cache.hpp"
#include "../pipeline/named.hpp"

#include <kernel/program/compute/transform/twiddle.hpp>

#include <cstddef>
#include <cstring>
#include <string>
#include <utility>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
namespace {

[[nodiscard]] rund::AccelCheck LookupPipeline(MetalAdapter &adapter,
                                              const char *const name,
                                              const NumericPolicy policy,
                                              std::shared_ptr<void> &out) {
  const auto constants = policy.constants();
  std::string key{name};
  for (const std::uint32_t value : constants) {
    key += '.';
    key += std::to_string(value);
  }
  out = LookupMetalNamedPipeline(adapter, key);
  if (out != nullptr) {
    return rund::AccelCheck{true, "ok"};
  }
  id<MTLDevice> device = (__bridge id<MTLDevice>)adapter.device.get();
  const std::uint64_t begin = MonotonicNanoseconds();
  std::shared_ptr<void> library_owner =
      AcquireMetalLibrary(adapter, MetalNumericSource());
  id<MTLLibrary> library = (__bridge id<MTLLibrary>)library_owner.get();
  if (library == nil ||
      !MakeNamedMetalPipeline(device, library, name, constants, out)) {
    SetMetalLastError(adapter, "accel_metal_pipeline_unavailable");
    return rund::AccelCheck{false, "accel_metal_pipeline_unavailable"};
  }
  StoreMetalNamedPipeline(adapter, std::move(key), out,
                          MonotonicNanoseconds() - begin);
  return rund::AccelCheck{true, "ok"};
}

} // namespace

id<MTLBuffer> ToMetalBuffer(const MetalResidentBufferResult &buffer) {
  return (__bridge id<MTLBuffer>)buffer.device_buffer.get();
}

void ClearStatus(const MetalResidentBufferResult &status,
                 const rund::kernel::u64 count) {
  auto *const base =
      static_cast<std::byte *>(MetalBufferContents(MetalRuntimeBuffer{
          .bytes = status.ref.bytes, .buffer = status.device_buffer}));
  auto *const data = base == nullptr ? nullptr
                                     : reinterpret_cast<rund::kernel::u32 *>(
                                           base + status.ref.offset_bytes);
  if (data != nullptr) {
    std::memset(data, 0,
                static_cast<std::size_t>(count * sizeof(rund::kernel::u32)));
  }
}

rund::AccelCheck StatusCheck(MetalAdapter &adapter,
                             const MetalResidentBufferResult &status,
                             const rund::kernel::u64 count,
                             const rund::kernel::u64 dispatches) {
  auto *const base =
      static_cast<const std::byte *>(MetalBufferContents(MetalRuntimeBuffer{
          .bytes = status.ref.bytes, .buffer = status.device_buffer}));
  auto *const data = base == nullptr
                         ? nullptr
                         : reinterpret_cast<const rund::kernel::u32 *>(
                               base + status.ref.offset_bytes);
  if (data == nullptr) {
    SetMetalLastError(adapter, "accel_metal_buffer_unavailable");
    return rund::AccelCheck{false, "accel_metal_buffer_unavailable"};
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
  RecordMetalDispatches(adapter, dispatches);
  SetMetalLastError(adapter, "ok");
  return check;
}

rund::AccelCheck RejectElement(MetalAdapter &adapter,
                               const char *const reason) {
  SetMetalLastError(adapter, reason);
  return rund::AccelCheck{false, reason};
}

MetalNumericPrepared::~MetalNumericPrepared() {
  if (adapter != nullptr) {
    ReleaseMetalBuffer(*adapter, std::move(twiddle));
  }
}

bool PrepareTwiddle(MetalNumericPrepared &state,
                    const rund::kernel::TransformPlan &plan) {
  if (plan.workspace_bytes == 0u) {
    return true;
  }
  state.twiddle = AcquireMetalBuffer(*state.adapter, plan.workspace_bytes,
                                     MetalBufferUsage::Input);
  void *const raw = MetalBufferContents(state.twiddle);
  if (raw == nullptr) {
    return false;
  }
  return plan.element_bytes == sizeof(rund::kernel::i64)
             ? rund::kernel::transform_twiddle::Fill(
                   static_cast<rund::kernel::i64 *>(raw), plan.element_count,
                   plan.direction, plan.fixed_format)
             : rund::kernel::transform_twiddle::Fill(
                   static_cast<rund::kernel::i32 *>(raw), plan.element_count,
                   plan.direction, plan.fixed_format);
}

rund::AccelCheck PreparedPipeline(MetalNumericPrepared &state,
                                  const char *const name,
                                  const NumericPolicy policy,
                                  const MetalKernelImmutablePipelines *const
                                      pipelines) {
  if (pipelines != nullptr) {
    if (!pipelines->ready(1u)) {
      return {false, "accel_metal_pipeline_unavailable"};
    }
    state.pipeline = pipelines->stages[0u];
    return {true, "ok"};
  }
  return LookupPipeline(*state.adapter, name, policy, state.pipeline);
}

rund::AccelCheck PreparedBuffers(const rund::AccelDevice &pick,
                                 MetalNumericPrepared &state,
                                 const std::span<MetalResidentReq> requests) {
  LookupMetalResidentBatch(pick, requests.data(), requests.size(),
                           "accel_metal_resident_id_unavailable");
  for (std::size_t index = 0u; index < requests.size(); ++index) {
    if (!state.buffers[index].check.ok) {
      return state.buffers[index].check;
    }
  }
  state.buffer_count = requests.size();
  return rund::AccelCheck{true, "ok"};
}

#endif

} // namespace rund::node::accel::detail
