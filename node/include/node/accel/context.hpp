#pragma once

#include <accel/check.hpp>
#include <accel/context/buffer.hpp>
#include <accel/context/buffer/descriptor.hpp>
#include <accel/context/value.hpp>
#include <accel/graph/value.hpp>
#include <accel/kernel/evidence.hpp>
#include <accel/kernel/run.hpp>
#include <accel/kernel/value.hpp>
#include <node/accel/buffer.hpp>

#include <cstdint>

namespace rund::node::accel {

[[nodiscard]] rund::AccelContext OpenAccel(const rund::AccelDevice& pick);

[[nodiscard]] rund::AccelBuffer OpenAccelBuffer(
    const rund::AccelContext& context, const rund::Buffer& buffer,
    rund::AccelBufferDesc desc);

[[nodiscard]] rund::AccelBuffer CreateAccelBuffer(
    const rund::AccelContext& context, rund::AccelBufferDesc desc);

[[nodiscard]] rund::AccelCheck UploadAccelBuffer(
    const rund::AccelContext& context, const rund::AccelBuffer& buffer,
    const void* data, std::uint64_t bytes, std::uint64_t offset = 0u);

[[nodiscard]] rund::AccelCheck DownloadAccelBuffer(
    const rund::AccelContext& context, const rund::AccelBuffer& buffer,
    void* data, std::uint64_t bytes, std::uint64_t offset = 0u);

[[nodiscard]] rund::AccelKernel CompileAccelKernel(
    const rund::AccelContext& context, const rund::AccelGraph& graph);

[[nodiscard]] rund::AccelEvidence RunAccelKernel(
    const rund::AccelContext& context, const rund::AccelKernel& kernel,
    const rund::AccelRun& run);

}  // namespace rund::node::accel
