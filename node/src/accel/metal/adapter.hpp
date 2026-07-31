#pragma once

#include <kernel/program/compute/artifact.hpp>
#include <kernel/program/compute/backend.hpp>
#include <kernel/program/compute/binding/model.hpp>
#include <kernel/program/compute/model.hpp>

#include <cstdint>
#include <memory>

namespace rund {
struct AccelDevice;
}

namespace rund::node::accel::detail {

enum class MetalBufferUsage : std::uint8_t;
struct MetalAdapter;
struct MetalRuntimeBuffer;

[[nodiscard]] bool MetalPickOwnsAdapter(const rund::AccelDevice &pick) noexcept;

[[nodiscard]] bool
ExecuteMetal(void *context, const rund::kernel::ComputePlan &plan,
             const rund::kernel::LoweringArtifact &artifact,
             const rund::kernel::ComputeDispatchWindow *windows,
             rund::kernel::u64 window_count,
             const rund::kernel::BindingSet &bindings);

[[nodiscard]] std::shared_ptr<void>
MetalPipelineForArtifact(MetalAdapter &adapter,
                         const rund::kernel::LoweringArtifact &artifact);

[[nodiscard]] MetalRuntimeBuffer AcquireMetalBuffer(MetalAdapter &adapter,
                                                    rund::kernel::u64 bytes,
                                                    MetalBufferUsage usage);
void ReleaseMetalBuffer(MetalAdapter &adapter, MetalRuntimeBuffer buffer);
[[nodiscard]] bool UploadMetalBuffer(MetalAdapter &adapter,
                                     const MetalRuntimeBuffer &buffer,
                                     const void *data, rund::kernel::u64 bytes);
[[nodiscard]] void *MetalBufferContents(const MetalRuntimeBuffer &buffer);
void RecordMetalHostToDeviceBytes(MetalAdapter &adapter,
                                  rund::kernel::u64 bytes);
void RecordMetalDeviceToHostBytes(MetalAdapter &adapter,
                                  rund::kernel::u64 bytes);
void RecordMetalDispatch(MetalAdapter &adapter);
void RecordMetalDispatches(MetalAdapter &adapter, rund::kernel::u64 count);
void RecordMetalCommandSubmitWaitNs(MetalAdapter &adapter,
                                    std::uint64_t elapsed_ns);
[[nodiscard]] std::uint64_t
RecordMetalComputeKernelSeconds(MetalAdapter &adapter, double start_seconds,
                                double end_seconds);

} // namespace rund::node::accel::detail
