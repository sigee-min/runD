#pragma once

#include <accel/buffer.hpp>
#include <accel/device.hpp>

#include <kernel/program/compute/binding/model.hpp>

#include <cstdint>
#include <memory>

namespace rund::node::accel::detail {

[[nodiscard]] rund::Buffer
MakeBuffer(const rund::AccelDevice &pick, const rund::BufferDesc &desc,
           const rund::AccelCheck &check,
           const rund::kernel::ResidentBufferRef &resident,
           std::shared_ptr<void> handle, std::uint64_t storage_bytes,
           bool storage_reused) noexcept;

} // namespace rund::node::accel::detail
