#pragma once

#include "model.hpp"

#include <rund/compute/virtual.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <vector>

namespace rund_node_test_virtual::product {

struct MemoryVirtualBackingCohort final
    : public rund::compute::VirtualReadCohort {
  std::uint64_t owner{};
  std::uint64_t generation{};

  [[nodiscard]] rund::compute::VirtualCohortId
  cohort_id() const noexcept override;
  [[nodiscard]] std::uint32_t lane_limit() const noexcept override;
  [[nodiscard]] rund::compute::Status read_cohort(
      std::span<rund::compute::VirtualCohortRead>,
      rund::compute::VirtualCohortResult &) noexcept override;
};

[[nodiscard]] std::shared_ptr<MemoryVirtualBackingCohort>
make_memory_backing_cohort() noexcept;

// Public-product backing fixture. size_bytes() exposes only logical bytes;
// rounded guard storage remains poisoned so a partial terminal page cannot be
// silently written as a full page.
class MemoryVirtualBacking final
    : public rund::compute::VirtualBacking,
      public rund::compute::VirtualBackingTransaction,
      public rund::compute::VirtualBackingReadCohort {
public:
  MemoryVirtualBacking(std::size_t logical_bytes, std::size_t page_bytes,
                       rund::compute::VirtualBackingTier tier =
                           rund::compute::VirtualBackingTier::Host,
                       std::shared_ptr<MemoryVirtualBackingCohort>
                           cohort = {});

  [[nodiscard]] std::uint64_t size_bytes() const noexcept override {
    return logical_bytes_;
  }
  [[nodiscard]] rund::compute::VirtualBackingTier
  tier() const noexcept override {
    return tier_;
  }
  [[nodiscard]] rund::compute::Status
  read(std::uint64_t offset, std::span<std::byte> output) noexcept override;
  [[nodiscard]] rund::compute::Status
  write(std::uint64_t offset,
        std::span<const std::byte> input) noexcept override;

  [[nodiscard]] rund::compute::Status
  begin(rund::compute::VirtualBackingTransactionSpec,
        rund::compute::VirtualBackingTransactionToken &) noexcept override;
  [[nodiscard]] rund::compute::Status
  stage(rund::compute::VirtualBackingTransactionToken &,
        std::span<const rund::compute::VirtualWrite>) noexcept override;
  [[nodiscard]] rund::compute::Status prepare_commit(
      const rund::compute::VirtualBackingTransactionToken &) noexcept override;
  [[nodiscard]] rund::compute::VirtualBackingTransactionResult
  commit(rund::compute::VirtualBackingTransactionToken &) noexcept override;
  void abort_known(
      rund::compute::VirtualBackingTransactionToken &&) noexcept override;
  void quarantine_unknown(
      rund::compute::VirtualBackingTransactionToken &&) noexcept override;

  [[nodiscard]] std::shared_ptr<rund::compute::VirtualReadCohort>
  cohort() const noexcept override;

  [[nodiscard]] bool seed(std::span<const std::byte> input) noexcept;
  [[nodiscard]] bool observe(std::span<std::byte> output) noexcept;
  void reset(std::byte value) noexcept;
  void fail_next_read(rund::compute::Reason reason) noexcept;
  void fail_next_write(rund::compute::Reason reason) noexcept;
  void fail_next_write_after(std::size_t prefix_bytes,
                             rund::compute::Reason reason) noexcept;
  void fail_next_transaction_unknown() noexcept;
  [[nodiscard]] bool tail_poisoned() const noexcept;
  [[nodiscard]] bool transaction_quarantined() const noexcept;
  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] const void *identity() const noexcept { return bytes_.data(); }
  [[nodiscard]] BackingFacts facts() const noexcept { return facts_; }

private:
  struct TransactionState final {
    rund::compute::VirtualBackingTransactionSpec spec{};
    std::vector<std::byte> shadow{};
    std::vector<bool> covered{};
    std::uint64_t generation{};
    std::uint64_t staged_bytes{};
    bool prepared{};
  };

  [[nodiscard]] bool contains(std::uint64_t offset,
                              std::size_t bytes) const noexcept;
  [[nodiscard]] TransactionState *valid_transaction(
      const rund::compute::VirtualBackingTransactionToken &) const noexcept;

  mutable std::mutex gate_;
  std::vector<std::byte> bytes_;
  std::size_t logical_bytes_{};
  std::size_t page_bytes_{};
  rund::compute::VirtualBackingTier tier_{
      rund::compute::VirtualBackingTier::Host};
  std::shared_ptr<MemoryVirtualBackingCohort> cohort_{};
  BackingFacts facts_{};
  rund::compute::Reason read_failure_{rund::compute::Reason::Ok};
  rund::compute::Reason write_failure_{rund::compute::Reason::Ok};
  std::size_t write_failure_prefix_{};
  std::shared_ptr<TransactionState> transaction_{};
  std::uint64_t next_generation_{};
  bool transaction_unknown_once_{};
  bool transaction_quarantined_{};

  friend struct MemoryVirtualBackingCohort;
};

} // namespace rund_node_test_virtual::product
