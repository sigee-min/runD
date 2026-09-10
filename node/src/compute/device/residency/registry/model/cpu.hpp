#pragma once

#include "../../../../pipeline/residency/model.hpp"
#include "../../execution/graph_persist/capacity.hpp"
#include "../credentials/cpu.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace rund::compute::detail::residency {
class Authority;
}

namespace rund::compute::detail::graph_reduce {
class CpuReceiptBook;
} // namespace rund::compute::detail::graph_reduce

namespace rund::compute::detail::residency::registry_model {

// Stack transaction used while publishing one CPU reservation. It carries no
// Authority storage; its destructor only clears the single pending key.
struct PendingCpu final {
  Authority *owner{};
  CpuReservationKey key{};
  bool active{};

  ~PendingCpu() noexcept;
  void publish() noexcept;
};

// Retry transaction over the Authority-owned GraphPersist slots. Arrays are
// bounded scratch on the call stack/object; no second frame or epoch table is
// introduced by this model.
struct RetryTxn final {
  Authority &owner;
  std::array<std::size_t, execution::GraphPersistSlotCapacity> rows{};
  std::array<std::uint32_t, execution::GraphPersistAggregateCapacity> frames{};
  std::size_t row_count{};
  std::size_t frame_count{};
  bool active{};

  explicit RetryTxn(Authority &value) noexcept : owner(value) {}
  ~RetryTxn() noexcept;
  [[nodiscard]] bool stage(std::uint64_t,
                           const GraphPersistIdentity &) noexcept;
  void commit() noexcept;
  void rollback() noexcept;
};

// CPU quarantine validation consumes the Authority's one gate and physical
// records directly. The owner keeps only an Authority reference; snapshots
// and bounded disposition arrays are private implementation records in the
// quarantine source, never a second mutable registry.
class CpuQuarantineOwner final {
public:
  explicit CpuQuarantineOwner(Authority &) noexcept;
  CpuQuarantineOwner(const CpuQuarantineOwner &) = delete;
  CpuQuarantineOwner &operator=(const CpuQuarantineOwner &) = delete;

  [[nodiscard]] bool discard(graph_reduce::CpuReceiptBook &,
                             const void *quarantine_identity) noexcept;

private:
  struct BookSnapshot;
  struct Plan;

  [[nodiscard]] static BookSnapshot
  snap_q(const graph_reduce::CpuReceiptBook &) noexcept;
  [[nodiscard]] bool check_epoch_q(const BookSnapshot &, std::size_t,
                                   Plan &) const noexcept;
  [[nodiscard]] bool check_unknown_q(const BookSnapshot &, std::size_t,
                                     Plan &) const noexcept;
  [[nodiscard]] bool check_retry_q(const BookSnapshot &, std::size_t,
                                   Plan &) const noexcept;
  [[nodiscard]] bool check_q(const BookSnapshot &, Plan &) const noexcept;
  void commit_q(const Plan &) noexcept;
  void scrub_q(graph_reduce::CpuReceiptBook &, const BookSnapshot &) noexcept;

  Authority &authority_;
};

} // namespace rund::compute::detail::residency::registry_model
