#include <accel/api.hpp>
#include <accel/buffer.hpp>

#include "src/accel/backend/token.hpp"
#include "src/accel/cpu/buffer.hpp"
#include "src/accel/metal/resident/model.hpp"
#include "src/accel/resident/ref.hpp"
#include "src/accel/resident/validation.hpp"
#include "src/accel/vulkan/buffer/resident/model.hpp"

#include <node/accel/pick.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string_view>
#include <type_traits>

namespace node_accel_contract {
namespace {

namespace detail = rund::node::accel::detail;

static_assert(std::is_base_of_v<detail::ResidentEntry, detail::CpuBuffer>);
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
static_assert(
    std::is_base_of_v<detail::ResidentEntry, detail::MetalResidentBuffer>);
#endif
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
static_assert(
    std::is_base_of_v<detail::ResidentEntry, detail::VulkanResidentBuffer>);
#endif

[[nodiscard]] detail::ResidentEntry Entry(const std::shared_ptr<void> &owner) {
  return detail::ResidentEntry{
      .id = 7u,
      .bytes = 64u,
      .element_bytes = 4u,
      .stride_bytes = 4u,
      .count = 16u,
      .usage = rund::kernel::kResidentUsageRead,
      .read_capable = true,
      .write_capable = true,
      .owner = owner,
  };
}

[[nodiscard]] const char *Reason(const detail::ResidentEntry &entry,
                                 const std::shared_ptr<void> &owner,
                                 const rund::kernel::ResidentBufferRef &ref,
                                 const std::shared_ptr<void> &handle,
                                 const bool allow_stride = false) {
  const char *reason = "ok";
  (void)detail::ResidentRefFits(entry, owner, ref, handle,
                                "compute_resident_id_invalid", reason,
                                allow_stride);
  return reason;
}

template <std::size_t N>
[[nodiscard]] bool ReasonsMatch(detail::ResidentEntry *const (&entries)[N],
                                const std::shared_ptr<void> &owner,
                                const rund::kernel::ResidentBufferRef &ref,
                                const std::shared_ptr<void> &handle,
                                const std::string_view expected,
                                const bool allow_stride = false) {
  for (const detail::ResidentEntry *const entry : entries) {
    if (entry == nullptr ||
        std::string_view{Reason(*entry, owner, ref, handle, allow_stride)} !=
            expected) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool CommonResidentBoundaryContract() {
  const std::shared_ptr<void> owner = std::make_shared<std::uint32_t>(7u);
  detail::CpuBuffer cpu{};
  static_cast<detail::ResidentEntry &>(cpu) = Entry(owner);
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  detail::MetalResidentBuffer metal{};
  static_cast<detail::ResidentEntry &>(metal) = Entry(owner);
#endif
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  detail::VulkanResidentBuffer vulkan{};
  static_cast<detail::ResidentEntry &>(vulkan) = Entry(owner);
#endif

  detail::ResidentEntry *const entries[] = {
      &cpu,
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
      &metal,
#endif
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
      &vulkan,
#endif
  };
  const rund::kernel::ResidentBufferRef dense = detail::RefFromResident(cpu);
  if (!ReasonsMatch(entries, owner, dense, owner, "ok")) {
    return false;
  }

  rund::kernel::ResidentBufferRef ref = dense;
  ref.id = 0u;
  if (!ReasonsMatch(entries, owner, ref, owner,
                    "compute_resident_id_invalid")) {
    return false;
  }
  ref = dense;
  ref.bytes = 0u;
  if (!ReasonsMatch(entries, owner, ref, owner,
                    "compute_resident_bytes_invalid")) {
    return false;
  }
  ref = dense;
  ref.bytes = 65u;
  if (!ReasonsMatch(entries, owner, ref, owner,
                    "compute_resident_bytes_invalid")) {
    return false;
  }
  ref = dense;
  ref.element_bytes = 0u;
  if (!ReasonsMatch(entries, owner, ref, owner,
                    "compute_resident_stride_invalid")) {
    return false;
  }
  ref = dense;
  ref.stride_bytes = 2u;
  if (!ReasonsMatch(entries, owner, ref, owner,
                    "compute_resident_stride_invalid")) {
    return false;
  }
  ref = dense;
  ref.stride_bytes = 8u;
  ref.count = 8u;
  if (!ReasonsMatch(entries, owner, ref, owner,
                    "compute_resident_stride_invalid") ||
      !ReasonsMatch(entries, owner, ref, owner, "ok", true)) {
    return false;
  }
  ref = dense;
  ref.count = 0u;
  if (!ReasonsMatch(entries, owner, ref, owner,
                    "compute_resident_stride_invalid")) {
    return false;
  }
  ref = dense;
  ref.count = std::numeric_limits<std::uint64_t>::max();
  ref.stride_bytes = 8u;
  if (!ReasonsMatch(entries, owner, ref, owner,
                    "compute_resident_stride_invalid", true)) {
    return false;
  }
  ref = dense;
  ref.offset_bytes = std::numeric_limits<std::uint64_t>::max();
  if (!ReasonsMatch(entries, owner, ref, owner,
                    "compute_resident_bytes_invalid")) {
    return false;
  }
  ref = dense;
  ref.offset_bytes = 60u;
  ref.count = 2u;
  if (!ReasonsMatch(entries, owner, ref, owner,
                    "compute_resident_bytes_invalid")) {
    return false;
  }
  ref = dense;
  ref.usage = 99u;
  if (!ReasonsMatch(entries, owner, ref, owner,
                    "compute_resident_usage_invalid")) {
    return false;
  }
  for (detail::ResidentEntry *const entry : entries) {
    entry->read_capable = false;
  }
  if (!ReasonsMatch(entries, owner, dense, owner,
                    "compute_resident_usage_invalid")) {
    return false;
  }
  for (detail::ResidentEntry *const entry : entries) {
    entry->read_capable = true;
    entry->write_capable = false;
  }
  ref = dense;
  ref.usage = rund::kernel::kResidentUsageWrite;
  if (!ReasonsMatch(entries, owner, ref, owner,
                    "compute_resident_usage_invalid")) {
    return false;
  }
  for (detail::ResidentEntry *const entry : entries) {
    entry->write_capable = true;
  }

  const std::shared_ptr<void> foreign = std::make_shared<std::uint32_t>(9u);
  if (!ReasonsMatch(entries, owner, dense, foreign,
                    "accel_buffer_unavailable") ||
      !ReasonsMatch(entries, nullptr, dense, owner,
                    "accel_buffer_unavailable")) {
    return false;
  }
  std::uint32_t alias = 0u;
  const std::shared_ptr<void> alias_handle{owner, &alias};
  const std::shared_ptr<void> foreign_owner{owner.get(), [](void *) {}};
  if (!ReasonsMatch(entries, owner, dense, alias_handle,
                    "accel_buffer_unavailable") ||
      !ReasonsMatch(entries, owner, dense, foreign_owner,
                    "accel_buffer_unavailable")) {
    return false;
  }
  for (detail::ResidentEntry *const entry : entries) {
    entry->owner = foreign;
  }
  return ReasonsMatch(entries, owner, dense, owner, "accel_buffer_unavailable");
}

[[nodiscard]] bool CpuResidentBoundaryContract() {
  const rund::AccelDevice cpu = rund::node::accel::PickAccel(rund::AccelPolicy{
      .preferred = {rund::AccelApi::Cpu, rund::AccelApi::Auto,
                    rund::AccelApi::Auto},
      .preferred_count = 1u,
  });
  const detail::PickAdmission admission = detail::AdmitPick(cpu);
  const rund::AccelDevice *const raw = admission.raw();
  if (!admission.check.ok || raw == nullptr) {
    return false;
  }
  detail::CpuBufferResult created = detail::CreateCpuResidentBuffer(
      *raw, rund::BufferDesc{.bytes = 64u,
                             .usage = rund::BufferUsage::ReadWrite,
                             .alignment = 16u});
  if (!created.check.ok) {
    return false;
  }
  rund::kernel::ResidentBufferRef ref = created.ref;
  ref.usage = rund::kernel::kResidentUsageRead;
  ref.stride_bytes = 8u;
  ref.count = 8u;
  const detail::CpuBufferResult valid = detail::LookupCpuResidentView(
      *raw, ref, created.buffer, rund::kernel::kResidentUsageRead);
  if (!valid.check.ok) {
    return false;
  }
  ref.offset_bytes = 12u;
  ref.count = 8u;
  const detail::CpuBufferResult range = detail::LookupCpuResidentView(
      *raw, ref, created.buffer, rund::kernel::kResidentUsageRead);
  ref = created.ref;
  ref.usage = 99u;
  const detail::CpuBufferResult usage =
      detail::LookupCpuResidentView(*raw, ref, created.buffer, 99u);
  ref = created.ref;
  std::uint32_t alias = 0u;
  const std::shared_ptr<void> alias_handle{created.buffer, &alias};
  const detail::CpuBufferResult owner = detail::LookupCpuResidentView(
      *raw, ref, alias_handle, rund::kernel::kResidentUsageWrite);
  ref.id = 0u;
  const detail::CpuBufferResult id = detail::LookupCpuResidentView(
      *raw, ref, created.buffer, rund::kernel::kResidentUsageWrite);
  return !range.check.ok &&
         std::string_view{range.check.reason} ==
             "compute_resident_bytes_invalid" &&
         !usage.check.ok &&
         std::string_view{usage.check.reason} ==
             "compute_resident_usage_invalid" &&
         !owner.check.ok &&
         std::string_view{owner.check.reason} == "accel_buffer_unavailable" &&
         !id.check.ok &&
         std::string_view{id.check.reason} == "compute_resident_id_invalid";
}

} // namespace

bool ResidentValidationContract() {
  return CommonResidentBoundaryContract() && CpuResidentBoundaryContract();
}

} // namespace node_accel_contract
