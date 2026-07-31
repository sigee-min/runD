#pragma once

#include <accel/buffer.hpp>
#include <accel/check.hpp>
#include <accel/context/buffer.hpp>
#include <accel/context/buffer/descriptor.hpp>
#include <accel/context/value.hpp>
#include <accel/device.hpp>

#include <node/accel/context.hpp>

#include "../backend/result.hpp"
#include "../backend/token.hpp"
#include "capability.hpp"
#include "internal.hpp"
#include <kernel/program/compute/binding/model.hpp>
#include <kernel/program/compute/model.hpp>

#include <cstdint>
#include <memory>

namespace rund::node::accel::detail {

struct SupportBufferAdmission {
  rund::AccelCheck check{};
  BackendLookup lookup{};
  std::uint64_t byte_extent = 0u;
};

struct TransferAdmission {
  rund::AccelCheck check{};
  std::shared_ptr<PickToken> pick{};
  BackendLookup route{};
  std::uint64_t byte_extent = 0u;
};

struct OpenBufferAdmission {
  rund::AccelCheck check{};
  ContextTokenAdmission context{};
  std::shared_ptr<void> handle{};
  rund::kernel::ResidentBufferRef resident{};
  std::uint64_t byte_extent = 0u;
};

[[nodiscard]] ContextTokenAdmission
AdmitContextToken(const rund::AccelContext &context);

[[nodiscard]] rund::AccelCheck
CheckDesc(const rund::AccelBufferDesc &desc) noexcept;

[[nodiscard]] OpenBufferAdmission
AdmitAccelBufferOpen(const rund::AccelContext &context,
                     const rund::Buffer &buffer,
                     const rund::AccelBufferDesc &desc);

[[nodiscard]] OpenBufferAdmission
AdmitAccelBufferOpen(const ContextTokenAdmission &context,
                     const rund::Buffer &buffer,
                     const rund::AccelBufferDesc &desc);

[[nodiscard]] ContextAdmission
SupportAdmissionFrom(const ContextTokenAdmission &admission);

[[nodiscard]] SupportBufferAdmission
AdmitAccelBufferForSupport(const ContextAdmission &admission,
                           const rund::AccelBuffer &buffer);

[[nodiscard]] TransferAdmission
AdmitAccelBufferTransfer(const rund::AccelContext &context,
                         const rund::AccelBuffer &buffer);

[[nodiscard]] TransferAdmission
AdmitAccelBufferTransfer(const ContextTokenAdmission &context,
                         const rund::AccelBuffer &buffer);

[[nodiscard]] rund::AccelCheck
TransferCheckFrom(rund::AccelCheck check, const char *overflow_reason) noexcept;

} // namespace rund::node::accel::detail
