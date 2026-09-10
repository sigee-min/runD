#pragma once

#include "../../../device/residency/execution/graph_persist.hpp"
#include "../../../device/residency/pool.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace rund::compute::detail {
struct PipelineState;
namespace residency {
class Authority;
struct CloseInfo;
} // namespace residency
} // namespace rund::compute::detail

namespace rund::compute::detail::residency::registry_model {
class CpuQuarantineOwner;
} // namespace rund::compute::detail::residency::registry_model

namespace rund::compute::detail::graph_reduce {

enum class CpuReceiptRole : std::uint8_t {
  Prefix,
  Supply,
  Middle,
  Collective,
};

class CpuReceiptBook;
struct CpuGraphQuarantine;

enum class CpuReserveResult : std::uint8_t {
  Reserved,
  Busy,
  Invalid,
};

enum class CpuBindResult : std::uint8_t {
  Bound,
  Closed,
  Retained,
};

enum class CpuReceiptBindResult : std::uint8_t {
  Bound,
  Busy,
  Invalid,
};

class CpuEpochPermit final {
public:
  CpuEpochPermit() = default;
  CpuEpochPermit(const CpuEpochPermit &) = delete;
  CpuEpochPermit &operator=(const CpuEpochPermit &) = delete;
  CpuEpochPermit(CpuEpochPermit &&other) noexcept;
  CpuEpochPermit &operator=(CpuEpochPermit &&other) noexcept;
  ~CpuEpochPermit() noexcept;

  [[nodiscard]] explicit operator bool() const noexcept {
    return book_ != nullptr;
  }
  void cancel() noexcept;
  [[nodiscard]] residency::CpuReservationKey key() const noexcept {
    return key_;
  }
  [[nodiscard]] bool
  bound_to(const residency::Authority &authority) const noexcept;

private:
  friend class CpuReceiptBook;
  friend CpuBindResult bind_cpu_epoch(residency::Authority &, CpuEpochPermit &,
                                      std::uint64_t token,
                                      std::uint64_t generation,
                                      residency::CloseInfo *) noexcept;

  void set(CpuReceiptBook &book, std::size_t slot,
           residency::Authority &authority,
           residency::CpuReservationKey key) noexcept;
  void prepare(std::uint64_t token, std::uint64_t generation) noexcept;
  [[nodiscard]] bool finalize(std::uint64_t token,
                              std::uint64_t generation) noexcept;
  void release_closed() noexcept;
  void detach() noexcept;
  void clear_local() noexcept;

  CpuReceiptBook *book_{};
  residency::Authority *authority_{};
  std::size_t slot_{};
  residency::CpuReservationKey key_{};
};

// A Ticket carries only this fixed handle. The credential itself lives in the
// run-owned book, so ticket reconstruction cannot silently drop an armed
// epoch.
struct CpuEpochReceipt final {
public:
  CpuEpochReceipt() = default;
  CpuEpochReceipt(const CpuEpochReceipt &) = delete;
  CpuEpochReceipt &operator=(const CpuEpochReceipt &) = delete;
  CpuEpochReceipt(CpuEpochReceipt &&other) noexcept;
  ~CpuEpochReceipt() noexcept;
  CpuEpochReceipt &operator=(CpuEpochReceipt &&other) noexcept;

  [[nodiscard]] explicit operator bool() const noexcept;
  [[nodiscard]] bool occupied() const noexcept;
  [[nodiscard]] std::uint64_t token() const noexcept;
  [[nodiscard]] std::uint64_t generation() const noexcept;
  [[nodiscard]] residency::CpuReservationKey key() const noexcept;
  [[nodiscard]] CpuReserveResult reserve(residency::Authority &,
                                         CpuEpochPermit &) noexcept;
  void detach() noexcept;
  [[nodiscard]] bool bound_to(const residency::Authority &) const noexcept;

private:
  friend class CpuReceiptBook;
  friend bool close_cpu_epoch(residency::Authority &, CpuEpochReceipt &, bool,
                              bool, residency::CloseInfo *) noexcept;

  void attach(CpuReceiptBook &, std::size_t) noexcept;
  [[nodiscard]] bool empty_handle() const noexcept;
  [[nodiscard]] bool has_snapshot() const noexcept;
  void set_key(residency::CpuReservationKey) noexcept;
  void set_credential(std::uint64_t token, std::uint64_t generation) noexcept;
  void clear() noexcept;

  CpuReceiptBook *book_{};
  std::size_t slot_{};
  residency::CpuReservationKey key_{};
  std::uint64_t token_{};
  std::uint64_t generation_{};
};

class CpuReceiptBook final {
public:
  static constexpr std::size_t RoleCount = 4u;
  static constexpr std::size_t SlotCount =
      residency::Pool::BankCount * RoleCount;

  CpuReceiptBook() noexcept;
  CpuReceiptBook(const CpuReceiptBook &) = delete;
  CpuReceiptBook &operator=(const CpuReceiptBook &) = delete;
  CpuReceiptBook(CpuReceiptBook &&) noexcept = delete;
  CpuReceiptBook &operator=(CpuReceiptBook &&) noexcept = delete;

  [[nodiscard]] bool valid() const noexcept { return domain_ != 0u; }
  [[nodiscard]] std::uint64_t domain() const noexcept { return domain_; }

  enum class SlotState : std::uint8_t {
    Free,
    Reserved,
    Prepared,
    Armed,
  };

  struct Slot final {
    residency::Authority *authority{};
    CpuEpochReceipt *handle{};
    CpuEpochPermit *permit{};
    std::uint64_t token{};
    std::uint64_t generation{};
    residency::CpuReservationKey key{};
    std::size_t bank{};
    CpuReceiptRole role{CpuReceiptRole::Prefix};
    SlotState state{SlotState::Free};
  };

  // A live entry is only an immutable locator for an Unknown Authority row.
  // The row owns pages, transitions, and rollback state; the book never
  // mirrors that journal.
  struct UnknownCred final {
    residency::Authority *authority{};
    residency::Identity plan{};
    residency::GraphPersistIdentity identity{};
    residency::FrameRegion region{};
    std::size_t page_count{};
    std::uint64_t domain{};
    std::uint64_t token{};
    std::uint64_t generation{};
    std::uint64_t coordinate{};
    bool terminal{};
    bool quarantined{};
    bool live{};

    void clear() noexcept;
  };

  static constexpr std::size_t UnknownCapacity =
      residency::execution::GraphPersistSlotCapacity + 2u;

  [[nodiscard]] CpuReceiptBindResult bind(CpuEpochReceipt &receipt,
                                          residency::Authority &authority,
                                          std::size_t bank,
                                          CpuReceiptRole role) noexcept;
  [[nodiscard]] CpuReserveResult reserve(std::size_t slot,
                                         residency::Authority &authority,
                                         CpuEpochPermit &permit) noexcept;
  [[nodiscard]] bool reserved_to(std::size_t slot,
                                 residency::CpuReservationKey key,
                                 const residency::Authority &authority) const
      noexcept;
  [[nodiscard]] bool bound(std::size_t slot,
                           residency::CpuReservationKey key,
                           const residency::Authority &authority) const noexcept;
  [[nodiscard]] bool owns(std::size_t slot,
                          const CpuEpochReceipt *handle) const noexcept;
  [[nodiscard]] bool cancel(std::size_t slot,
                            residency::CpuReservationKey key,
                            residency::Authority &authority) noexcept;
  void prepare(std::size_t slot, residency::CpuReservationKey key,
               std::uint64_t token, std::uint64_t generation) noexcept;
  [[nodiscard]] bool finalize(std::size_t slot,
                              residency::CpuReservationKey key,
                              std::uint64_t token,
                              std::uint64_t generation) noexcept;
  [[nodiscard]] bool
  occupied(residency::CpuReservationKey key) const noexcept;
  [[nodiscard]] bool
  credential(residency::CpuReservationKey key) const noexcept;
  [[nodiscard]] std::uint64_t
  token(residency::CpuReservationKey key) const noexcept;
  [[nodiscard]] std::uint64_t
  generation(residency::CpuReservationKey key) const noexcept;
  [[nodiscard]] bool idle() const noexcept;
  [[nodiscard]] std::size_t live_receipts() const noexcept;
  [[nodiscard]] std::size_t live_permits() const noexcept;
  [[nodiscard]] bool can_hold_unknown(
      residency::Authority &authority,
      const residency::execution::GraphPersist &ticket) const noexcept;
  [[nodiscard]] bool
  hold_unknown(residency::Authority &authority,
               residency::execution::GraphPersist &ticket) noexcept;
  [[nodiscard]] bool
  recover(residency::Authority &authority,
          residency::CloseInfo *info = nullptr) noexcept;

private:
  friend class ::rund::compute::detail::residency::Authority;
  friend class ::rund::compute::detail::residency::registry_model::
      CpuQuarantineOwner;
  friend struct CpuEpochReceipt;
  friend class CpuEpochPermit;
  friend struct CpuGraphQuarantine;

  void drop_receipt() noexcept;
  void detach_handle(std::size_t slot, CpuEpochReceipt *handle) noexcept;
  void move_handle(std::size_t slot, CpuEpochReceipt *from,
                   CpuEpochReceipt *to) noexcept;
  void move_permit(std::size_t slot, CpuEpochPermit *from,
                   CpuEpochPermit *to) noexcept;
  void detach_permit(std::size_t slot, CpuEpochPermit *permit) noexcept;
  void release_permit(Slot &entry) noexcept;
  void drop_handles() noexcept;
  [[nodiscard]] bool clear_closed(std::size_t slot,
                                  residency::CpuReservationKey key) noexcept;
  void clear(const CpuEpochReceipt &receipt) noexcept;
  [[nodiscard]] Slot *find(residency::CpuReservationKey key) noexcept;
  [[nodiscard]] const Slot *
  find(residency::CpuReservationKey key) const noexcept;
  [[nodiscard]] residency::CpuReservationKey key(std::size_t slot) const
      noexcept;
  void reset_cred(Slot &entry) noexcept;
  void reset(Slot &entry) noexcept;
  [[nodiscard]] static constexpr std::size_t
  index(std::size_t bank, CpuReceiptRole role) noexcept {
    return bank * RoleCount + static_cast<std::size_t>(role);
  }

  std::uint64_t domain_{};
  std::array<Slot, SlotCount> slots_{};
  std::uint64_t next_reservation_{1u};
  std::array<UnknownCred, UnknownCapacity> unknowns_{};
  std::size_t live_receipts_{};
  std::size_t live_permits_{};
};

struct CpuGraphQuarantine final {
  enum class Phase : std::uint8_t { Ready, Armed, Held };

  explicit CpuGraphQuarantine(std::shared_ptr<CpuReceiptBook> value) noexcept;
  CpuGraphQuarantine(const CpuGraphQuarantine &) = delete;
  CpuGraphQuarantine &operator=(const CpuGraphQuarantine &) = delete;
  ~CpuGraphQuarantine() noexcept;

  [[nodiscard]] bool empty() const noexcept;
  [[nodiscard]] bool bind(const PipelineState *pipeline_value,
                          const residency::Pool *pool_value,
                          residency::Authority *authority_value) noexcept;
  [[nodiscard]] bool
  arm(const std::shared_ptr<CpuGraphQuarantine> &owner) noexcept;
  void commit() noexcept;
  void clear() noexcept;
  void drop_handles() noexcept;

  std::shared_ptr<CpuReceiptBook> book;
  const PipelineState *pipeline{};
  const residency::Pool *pool{};
  residency::Authority *authority{};
  Phase phase{Phase::Ready};
  std::shared_ptr<CpuGraphQuarantine> self;
};

} // namespace rund::compute::detail::graph_reduce
