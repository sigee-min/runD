#pragma once

#include "../registry.hpp"

namespace rund::compute::detail::residency {

// Stateless owner for the CPU Graph reservation, epoch, and quarantine
// protocol. Authority remains the sole owner of the gate, frame table, CPU
// state, and credentials; this facet only borrows and authenticates them.
class CpuGraphOwner final {
public:
  explicit CpuGraphOwner(Authority &) noexcept;

  CpuGraphOwner(const CpuGraphOwner &) noexcept = default;
  CpuGraphOwner &operator=(const CpuGraphOwner &) = delete;

  [[nodiscard]] std::uint64_t cpu_owner_id() const noexcept;
  [[nodiscard]] bool confirm_cpu_epoch(CpuReservationKey,
                                       std::uint64_t token,
                                       std::uint64_t generation) noexcept;
  [[nodiscard]] bool close_cpu_epoch(CpuReservationKey, std::uint64_t token,
                                     std::uint64_t generation, bool success,
                                     bool invalidate_all = false,
                                     CloseInfo *info = nullptr) noexcept;
  [[nodiscard]] bool retain_cpu_quarantine(
      std::shared_ptr<graph_reduce::CpuGraphQuarantine>) noexcept;
  [[nodiscard]] bool discard_cpu_quarantine(
      const std::shared_ptr<graph_reduce::CpuGraphQuarantine> &,
      const void *pipeline_identity, const void *pool_identity) noexcept;
  [[nodiscard]] bool cpu_quarantine_active() const noexcept;

private:
  friend class graph_reduce::CpuReceiptBook;
  [[nodiscard]] CpuReservationKey reserve_cpu_key(std::uint64_t domain,
                                                  std::uint64_t role_slot,
                                                  std::uint64_t nonce) noexcept;
  [[nodiscard]] bool cancel_cpu_reservation(CpuReservationKey) noexcept;

  Authority &authority_;
};

} // namespace rund::compute::detail::residency
