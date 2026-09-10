#pragma once

#include <rund/compute/abi/virtual.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <type_traits>
#include <utility>

namespace rund::compute {

struct VirtualRead final {
  std::uint64_t offset{};
  std::span<std::byte> bytes;
};

struct VirtualWrite final {
  std::uint64_t offset{};
  std::span<const std::byte> bytes;
};

// A backing may opt into all-or-none output publication without changing the
// VirtualBacking vtable. The provider owns the shadow generation; this
// descriptor binds the run to one backing version, physical backing geometry,
// and the logical write tiling used by the output service.
struct VirtualBackingTransactionSpec final {
  std::uint64_t backing_id{};
  std::uint64_t base_version{};
  std::uint64_t logical_bytes{};
  std::uint64_t backing_page_bytes{};
  std::uint64_t backing_page_count{};
  std::uint64_t write_page_bytes{};
  std::uint64_t write_page_count{};

  [[nodiscard]] constexpr bool
  operator==(const VirtualBackingTransactionSpec &) const noexcept = default;
};

enum class VirtualBackingTransactionResult : std::uint8_t {
  Success,
  KnownNoWrite,
  UnknownMayWrite,
};

class VirtualBacking;

// Optional additive side capability for the fixed Host GraphPersist ring.
// VirtualBacking's vtable remains unchanged; the capability value is stable
// for the backing lifetime and never authorizes more than two write lanes.
class VirtualWriteLanes {
public:
  virtual ~VirtualWriteLanes() = default;
  [[nodiscard]] virtual std::uint32_t write_lanes() const noexcept = 0;
};

// A read cohort is an opt-in, synchronous join for a bounded set of staged
// Graph inputs. It is deliberately a side interface so VirtualBacking's
// existing vtable remains unchanged.
struct VirtualCohortId final {
  std::uint64_t owner{};
  std::uint64_t generation{};

  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return owner != 0u && generation != 0u;
  }
  [[nodiscard]] constexpr bool
  operator==(const VirtualCohortId &) const noexcept = default;
};

struct VirtualCohortRead final {
  VirtualBacking *backing{};
  std::uint64_t backing_id{};
  std::uint64_t version{};
  std::uint64_t page_bytes{};
  std::uint64_t page_count{};
  std::span<std::byte> destination{};
};

struct VirtualCohortResult final {
  // `joined` becomes true only after the provider has validated the complete
  // descriptor set. On success both failure fields retain this sentinel.
  bool joined{};
  std::uint64_t completed_bytes{};
  std::uint64_t failed_member{std::numeric_limits<std::uint64_t>::max()};
  std::uint64_t failed_page{std::numeric_limits<std::uint64_t>::max()};
};

class VirtualReadCohort {
public:
  virtual ~VirtualReadCohort() = default;
  [[nodiscard]] virtual VirtualCohortId cohort_id() const noexcept = 0;
  [[nodiscard]] virtual std::uint32_t lane_limit() const noexcept = 0;
  // The call is a synchronous join: it returns only after every descriptor
  // has either filled its whole destination or reported its exact page. The
  // provider does not retain descriptor spans or reenter backing callbacks.
  [[nodiscard]] virtual Status read_cohort(std::span<VirtualCohortRead>,
                                           VirtualCohortResult &) noexcept = 0;
};

// A backing is only a cohort member. Its shared provider stays alive through
// the complete synchronous join and is independently reauthenticated before
// and after the call.
class VirtualBackingReadCohort {
public:
  virtual ~VirtualBackingReadCohort() = default;
  [[nodiscard]] virtual std::shared_ptr<VirtualReadCohort>
  cohort() const noexcept = 0;
};

class VirtualBackingTransactionToken final {
public:
  VirtualBackingTransactionToken() = default;
  VirtualBackingTransactionToken(std::shared_ptr<void> opaque,
                                 VirtualBackingTransactionSpec spec,
                                 std::uint64_t generation) noexcept
      : opaque_(std::move(opaque)), spec_(spec), generation_(generation) {}
  VirtualBackingTransactionToken(const VirtualBackingTransactionToken &) =
      delete;
  VirtualBackingTransactionToken &
  operator=(const VirtualBackingTransactionToken &) = delete;
  VirtualBackingTransactionToken(VirtualBackingTransactionToken &&) noexcept =
      default;
  VirtualBackingTransactionToken &
  operator=(VirtualBackingTransactionToken &&) noexcept = default;

  [[nodiscard]] explicit operator bool() const noexcept {
    return opaque_ != nullptr && generation_ != 0u;
  }
  [[nodiscard]] const std::shared_ptr<void> &opaque() const noexcept {
    return opaque_;
  }
  [[nodiscard]] VirtualBackingTransactionSpec spec() const noexcept {
    return spec_;
  }
  [[nodiscard]] std::uint64_t generation() const noexcept {
    return generation_;
  }

private:
  std::shared_ptr<void> opaque_{};
  VirtualBackingTransactionSpec spec_{};
  std::uint64_t generation_{};
};

// Optional side interface. VirtualBacking remains ABI/vtable compatible;
// callers discover this capability with dynamic_cast at the run boundary.
class VirtualBackingTransaction {
public:
  virtual ~VirtualBackingTransaction() = default;
  [[nodiscard]] virtual Status
  begin(VirtualBackingTransactionSpec,
        VirtualBackingTransactionToken &) noexcept = 0;
  [[nodiscard]] virtual Status
  stage(VirtualBackingTransactionToken &,
        std::span<const VirtualWrite>) noexcept = 0;
  [[nodiscard]] virtual Status
  prepare_commit(const VirtualBackingTransactionToken &) noexcept = 0;
  [[nodiscard]] virtual VirtualBackingTransactionResult
  commit(VirtualBackingTransactionToken &) noexcept = 0;
  virtual void abort_known(VirtualBackingTransactionToken &&) noexcept = 0;
  virtual void
  quarantine_unknown(VirtualBackingTransactionToken &&) noexcept = 0;
};

// Physical latency class of the logical backing. This selects bounded
// prefetch distance only; it never changes ordering, cache identity, or
// correctness. Host is the default for existing memory-backed owners.
enum class VirtualBackingTier : std::uint8_t { Host, Persistent };

namespace detail {
struct VirtualBackingState;
struct VirtualBackingAccess;
} // namespace detail

// VirtualBacking is the logical dataset authority. Implementations may use
// memory, a file, object storage, or deterministic generation; runD requests
// bounded byte ranges and never assumes that the complete dataset is mapped.
class VirtualBacking {
public:
  VirtualBacking();
  VirtualBacking(const VirtualBacking &) = delete;
  VirtualBacking &operator=(const VirtualBacking &) = delete;
  virtual ~VirtualBacking();

  [[nodiscard]] virtual std::uint64_t size_bytes() const noexcept = 0;
  [[nodiscard]] virtual VirtualBackingTier tier() const noexcept {
    return VirtualBackingTier::Host;
  }
  // One is the serialized default. A read-only backing may explicitly admit
  // bounded parallel range reads; runD never exceeds two read lanes.
  [[nodiscard]] virtual std::uint32_t max_parallel_reads() const noexcept {
    return 1u;
  }
  [[nodiscard]] virtual Status read(std::uint64_t offset,
                                    std::span<std::byte> output) noexcept = 0;
  [[nodiscard]] virtual Status
  write(std::uint64_t offset, std::span<const std::byte> input) noexcept = 0;
  // Batch is the physical page-store interface. The default preserves custom
  // backings by issuing the exact ordered scalar callbacks; NVMe/object-store
  // implementations may override it with vectored I/O without changing page
  // order or partial-failure semantics.
  [[nodiscard]] virtual Status
  read_batch(std::span<const VirtualRead> ranges) noexcept;
  [[nodiscard]] virtual Status
  write_batch(std::span<const VirtualWrite> ranges) noexcept;
  // External mutation must be published through this boundary before the
  // backing is used again. It advances the cache generation while serialized
  // with every run using this backing; hidden mutation is a contract error.
  [[nodiscard]] Status invalidate() noexcept;

protected:
  // A transaction provider calls this only while the owning run holds the
  // backing gate and after its shadow has become externally visible.
  void publish_transaction_version() noexcept;

private:
  friend struct detail::VirtualBackingAccess;

  std::unique_ptr<detail::VirtualBackingState> state_;
};

static_assert(std::is_trivially_copyable_v<ResidencyConfig>);
static_assert(std::is_trivially_copyable_v<VirtualRead>);
static_assert(std::is_trivially_copyable_v<VirtualWrite>);

} // namespace rund::compute
