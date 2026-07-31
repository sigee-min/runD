#pragma once

#include <accel/api.hpp>
#include <accel/check.hpp>
#include <accel/context/buffer.hpp>
#include <accel/context/buffer/descriptor.hpp>
#include <accel/device.hpp>

#include <node/accel/context.hpp>

#include "../backend/token.hpp"
#include "internal.hpp"
#include <kernel/program/compute/model.hpp>

#include <cstdint>
#include <memory>
#include <utility>

namespace rund::node::accel::detail {

struct ContextToken final {
  ContextToken(const std::uint64_t value,
               std::shared_ptr<PickToken> backend) noexcept
      : id(value), api(backend->raw.api), caps(backend->raw.caps),
        pick(std::move(backend)) {}

  const std::uint64_t id;
  const rund::AccelApi api;
  const rund::kernel::ComputeCaps caps;
  const std::shared_ptr<PickToken> pick;
};

struct ContextTokenAdmission {
  rund::AccelCheck check{};
  std::shared_ptr<ContextToken> token{};

  [[nodiscard]] const rund::AccelDevice *raw() const noexcept {
    return token == nullptr || token->pick == nullptr ? nullptr
                                                      : &token->pick->raw;
  }

  [[nodiscard]] const BackendOps *ops() const noexcept {
    return token == nullptr || token->pick == nullptr ? nullptr
                                                      : token->pick->ops;
  }
};

struct AccelBufferToken final {
  AccelBufferToken(std::shared_ptr<ContextToken> admitted_context,
                   rund::Buffer backend_buffer,
                   const rund::kernel::ResidentBufferRef resident_ref,
                   const std::uint64_t extent,
                   const rund::AccelBufferDesc &desc) noexcept
      : context(std::move(admitted_context)),
        backend_check(backend_buffer.check), backend_id(backend_buffer.id),
        backend_bytes(backend_buffer.bytes),
        backend_element_bytes(backend_buffer.element_bytes),
        backend_stride_bytes(backend_buffer.stride_bytes),
        backend_count(backend_buffer.count),
        backend_storage_bytes(backend_buffer.storage_bytes),
        backend_storage_reused(backend_buffer.storage_reused),
        backend_usage(backend_buffer.usage),
        backend_handle(std::move(backend_buffer.handle)),
        resident(resident_ref),
        byte_extent(extent), scalar_width_bytes(desc.scalar_width_bytes),
        count(desc.count), usage(desc.usage) {}

  const std::shared_ptr<ContextToken> context;
  const rund::AccelCheck backend_check;
  const std::uint64_t backend_id;
  const std::uint64_t backend_bytes;
  const std::uint64_t backend_element_bytes;
  const std::uint64_t backend_stride_bytes;
  const std::uint64_t backend_count;
  const std::uint64_t backend_storage_bytes;
  const bool backend_storage_reused;
  const rund::BufferUsage backend_usage;
  const std::shared_ptr<void> backend_handle;
  const rund::kernel::ResidentBufferRef resident;
  const std::uint64_t byte_extent;
  const std::uint64_t scalar_width_bytes;
  const std::uint64_t count;
  const rund::BufferUsage usage;
};

[[nodiscard]] std::shared_ptr<void>
PublicTokenOwner(const std::shared_ptr<ContextToken> &token) noexcept;

[[nodiscard]] std::shared_ptr<ContextToken>
MintContextToken(rund::AccelApi api, const rund::kernel::ComputeCaps &caps,
                 std::shared_ptr<PickToken> pick);

[[nodiscard]] std::shared_ptr<ContextToken>
LookupContextToken(const std::shared_ptr<void> &owner);

[[nodiscard]] std::shared_ptr<AccelBufferToken>
MakeAccelBufferToken(const ContextTokenAdmission &admission,
                     rund::Buffer backend,
                     const rund::kernel::ResidentBufferRef &resident,
                     std::uint64_t byte_extent,
                     const rund::AccelBufferDesc &desc) noexcept;

[[nodiscard]] std::shared_ptr<void>
PublicBufferTokenOwner(const std::shared_ptr<AccelBufferToken> &token) noexcept;

[[nodiscard]] std::shared_ptr<AccelBufferToken>
LookupAccelBufferToken(const std::shared_ptr<void> &owner) noexcept;

[[nodiscard]] bool
AccelBufferTokenMatches(const ContextAdmission &admission,
                        const rund::AccelBuffer &buffer,
                        const std::shared_ptr<AccelBufferToken> &token) noexcept;

} // namespace rund::node::accel::detail
