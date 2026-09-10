#include "backing.hpp"

#include "../../../../../src/compute/virtual/backing.hpp"
#include "../../../../../src/compute/virtual/state.hpp"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <limits>
#include <new>

namespace rund_node_test_virtual::product {
namespace {

std::atomic<std::uint64_t> next_cohort_owner{1u};

[[nodiscard]] std::size_t rounded_capacity(const std::size_t logical,
                                           const std::size_t page) noexcept {
  if (page == 0u) {
    return 0u;
  }
  const std::size_t remainder = logical % page;
  const std::size_t padding = remainder == 0u ? page : page - remainder;
  return logical <= std::numeric_limits<std::size_t>::max() - padding
             ? logical + padding
             : 0u;
}

} // namespace

std::shared_ptr<MemoryVirtualBackingCohort>
make_memory_backing_cohort() noexcept {
  try {
    auto cohort = std::make_shared<MemoryVirtualBackingCohort>();
    cohort->owner = next_cohort_owner.fetch_add(1u, std::memory_order_relaxed);
    if (cohort->owner == 0u) {
      cohort->owner =
          next_cohort_owner.fetch_add(1u, std::memory_order_relaxed);
    }
    cohort->generation = 1u;
    return cohort->owner == 0u || cohort->generation == 0u ? nullptr
                                                           : std::move(cohort);
  } catch (const std::bad_alloc &) {
    return {};
  }
}

MemoryVirtualBacking::MemoryVirtualBacking(
    const std::size_t logical_bytes, const std::size_t page_bytes,
    const rund::compute::VirtualBackingTier tier,
    std::shared_ptr<MemoryVirtualBackingCohort> cohort)
    : bytes_(rounded_capacity(logical_bytes, page_bytes), TailPoison),
      logical_bytes_(logical_bytes), page_bytes_(page_bytes), tier_(tier),
      cohort_(std::move(cohort)) {}

bool MemoryVirtualBacking::contains(const std::uint64_t offset,
                                    const std::size_t bytes) const noexcept {
  return offset <= logical_bytes_ &&
         bytes <= logical_bytes_ - static_cast<std::size_t>(offset);
}

rund::compute::Status
MemoryVirtualBacking::read(const std::uint64_t offset,
                           const std::span<std::byte> output) noexcept {
  std::lock_guard lock{gate_};
  if (read_failure_ != rund::compute::Reason::Ok) {
    const rund::compute::Reason reason = read_failure_;
    read_failure_ = rund::compute::Reason::Ok;
    ++facts_.read_failure_count;
    return rund::compute::Status::fail(reason);
  }
  if (!contains(offset, output.size())) {
    return rund::compute::Status::fail(rund::compute::Reason::ShapeMismatch);
  }
  if (!output.empty()) {
    std::memcpy(output.data(), bytes_.data() + offset, output.size());
  }
  ++facts_.read_count;
  facts_.read_bytes += output.size();
  return rund::compute::Status::success();
}

rund::compute::Status
MemoryVirtualBacking::write(const std::uint64_t offset,
                            const std::span<const std::byte> input) noexcept {
  std::lock_guard lock{gate_};
  if (write_failure_ != rund::compute::Reason::Ok) {
    const rund::compute::Reason reason = write_failure_;
    write_failure_ = rund::compute::Reason::Ok;
    const std::size_t changed = std::min(write_failure_prefix_, input.size());
    write_failure_prefix_ = 0u;
    if (changed != 0u && contains(offset, changed)) {
      std::memcpy(bytes_.data() + offset, input.data(), changed);
      facts_.partial_write_bytes += changed;
    }
    ++facts_.write_failure_count;
    return rund::compute::Status::fail(reason);
  }
  if (!contains(offset, input.size())) {
    return rund::compute::Status::fail(rund::compute::Reason::ShapeMismatch);
  }
  if (!input.empty()) {
    std::memcpy(bytes_.data() + offset, input.data(), input.size());
  }
  ++facts_.write_count;
  facts_.write_bytes += input.size();
  return rund::compute::Status::success();
}

rund::compute::VirtualCohortId
MemoryVirtualBackingCohort::cohort_id() const noexcept {
  return rund::compute::VirtualCohortId{owner, generation};
}

std::uint32_t MemoryVirtualBackingCohort::lane_limit() const noexcept {
  return 2u;
}

rund::compute::Status MemoryVirtualBackingCohort::read_cohort(
    const std::span<rund::compute::VirtualCohortRead> requests,
    rund::compute::VirtualCohortResult &result) noexcept {
  result = {};
  const rund::compute::VirtualCohortId id = cohort_id();
  if (!id || requests.empty() ||
      requests.size() >
          rund::compute::detail::VirtualPipelineState::InputCapacity) {
    return rund::compute::Status::fail(
        rund::compute::Reason::PipelineInvalid);
  }
  auto *const first =
      requests.front().backing == nullptr
          ? nullptr
          : dynamic_cast<MemoryVirtualBacking *>(requests.front().backing);
  if (first == nullptr || first->page_bytes_ == 0u ||
      first->logical_bytes_ == 0u) {
    return rund::compute::Status::fail(
        rund::compute::Reason::PipelineInvalid);
  }
  const std::size_t expected_pages =
      first->logical_bytes_ / first->page_bytes_ +
      (first->logical_bytes_ % first->page_bytes_ == 0u ? 0u : 1u);
  for (std::size_t index = 0u; index < requests.size(); ++index) {
    rund::compute::VirtualCohortRead &request = requests[index];
    auto *const peer =
        request.backing == nullptr
            ? nullptr
            : dynamic_cast<MemoryVirtualBacking *>(request.backing);
    const std::shared_ptr<rund::compute::VirtualReadCohort> member =
        peer == nullptr ? std::shared_ptr<rund::compute::VirtualReadCohort>{}
                        : peer->cohort();
    if (peer == nullptr || member == nullptr || member.get() != this ||
        peer->logical_bytes_ != first->logical_bytes_ ||
        peer->page_bytes_ != first->page_bytes_ || request.backing_id == 0u ||
        request.version == 0u ||
        rund::compute::detail::VirtualBackingAccess::id(*peer) !=
            request.backing_id ||
        rund::compute::detail::VirtualBackingAccess::version(*peer) !=
            request.version ||
        request.page_bytes != first->page_bytes_ ||
        request.page_count != expected_pages ||
        request.destination.size() != first->logical_bytes_ ||
        request.destination.data() == nullptr) {
      return rund::compute::Status::fail(
          rund::compute::Reason::PipelineInvalid);
    }
  }
  result.joined = true;
  for (std::size_t index = 0u; index < requests.size(); ++index) {
    rund::compute::VirtualCohortRead &request = requests[index];
    auto *const peer = static_cast<MemoryVirtualBacking *>(request.backing);
    for (std::uint64_t page = 0u; page < request.page_count; ++page) {
      if (page > std::numeric_limits<std::uint64_t>::max() /
                         request.page_bytes) {
        return rund::compute::Status::fail(
            rund::compute::Reason::PipelineCapacity);
      }
      const std::uint64_t offset64 = page * request.page_bytes;
      if (offset64 > first->logical_bytes_) {
        return rund::compute::Status::fail(
            rund::compute::Reason::PipelineCapacity);
      }
      const std::size_t offset = static_cast<std::size_t>(offset64);
      const std::size_t count =
          std::min(first->page_bytes_, first->logical_bytes_ - offset);
      const rund::compute::Status status = peer->read(
          offset, request.destination.subspan(offset, count));
      if (!status) {
        result.failed_member = index;
        result.failed_page = page;
        return status;
      }
      result.completed_bytes += count;
    }
  }
  return rund::compute::Status::success();
}

std::shared_ptr<rund::compute::VirtualReadCohort>
MemoryVirtualBacking::cohort() const noexcept {
  return cohort_;
}

rund::compute::Status MemoryVirtualBacking::begin(
    const rund::compute::VirtualBackingTransactionSpec spec,
    rund::compute::VirtualBackingTransactionToken &token) noexcept {
  std::lock_guard lock{gate_};
  const std::size_t expected_backing_pages =
      spec.backing_page_bytes == 0u
          ? 0u
          : static_cast<std::size_t>(
                spec.logical_bytes / spec.backing_page_bytes +
                (spec.logical_bytes % spec.backing_page_bytes != 0u));
  const std::size_t expected_write_pages =
      spec.write_page_bytes == 0u
          ? 0u
          : static_cast<std::size_t>(
                spec.logical_bytes / spec.write_page_bytes +
                (spec.logical_bytes % spec.write_page_bytes != 0u));
  if (transaction_quarantined_ || transaction_ != nullptr ||
      spec.logical_bytes != logical_bytes_ ||
      spec.backing_page_bytes == 0u ||
      spec.backing_page_count != expected_backing_pages ||
      spec.write_page_count != expected_write_pages ||
      spec.write_page_bytes == 0u ||
      expected_backing_pages == 0u || expected_write_pages == 0u) {
    return rund::compute::Status::fail(rund::compute::Reason::PipelineCapacity);
  }
  try {
    auto next = std::make_shared<TransactionState>();
    next->spec = spec;
    next->generation = ++next_generation_;
    next->shadow = bytes_;
    next->covered.assign(logical_bytes_, false);
    transaction_ = std::move(next);
    token = rund::compute::VirtualBackingTransactionToken{
        transaction_, spec, transaction_->generation};
  } catch (const std::bad_alloc &) {
    return rund::compute::Status::fail(rund::compute::Reason::PipelineCapacity);
  }
  return rund::compute::Status::success();
}

rund::compute::Status MemoryVirtualBacking::stage(
    rund::compute::VirtualBackingTransactionToken &token,
    const std::span<const rund::compute::VirtualWrite> ranges) noexcept {
  std::lock_guard lock{gate_};
  TransactionState *const state = valid_transaction(token);
  if (state == nullptr || state->prepared || ranges.empty()) {
    return rund::compute::Status::fail(rund::compute::Reason::PipelineInvalid);
  }
  if (write_failure_ != rund::compute::Reason::Ok) {
    const rund::compute::Reason reason = write_failure_;
    write_failure_ = rund::compute::Reason::Ok;
    write_failure_prefix_ = 0u;
    ++facts_.write_failure_count;
    return rund::compute::Status::fail(reason);
  }
  for (const rund::compute::VirtualWrite range : ranges) {
    if (range.bytes.empty() || !contains(range.offset, range.bytes.size())) {
      return rund::compute::Status::fail(
          rund::compute::Reason::TransferInvalid);
    }
    for (std::size_t index = 0u; index < range.bytes.size(); ++index) {
      if (state->covered[static_cast<std::size_t>(range.offset) + index]) {
        return rund::compute::Status::fail(
            rund::compute::Reason::PipelineInvalid);
      }
    }
  }
  for (const rund::compute::VirtualWrite range : ranges) {
    const std::size_t offset = static_cast<std::size_t>(range.offset);
    std::memcpy(state->shadow.data() + offset, range.bytes.data(),
                range.bytes.size());
    std::fill(state->covered.begin() + offset,
              state->covered.begin() + offset + range.bytes.size(), true);
    state->staged_bytes += range.bytes.size();
  }
  return rund::compute::Status::success();
}

rund::compute::Status MemoryVirtualBacking::prepare_commit(
    const rund::compute::VirtualBackingTransactionToken &token) noexcept {
  std::lock_guard lock{gate_};
  TransactionState *const state = valid_transaction(token);
  if (state == nullptr || state->staged_bytes != state->spec.logical_bytes ||
      std::find(state->covered.begin(), state->covered.end(), false) !=
          state->covered.end()) {
    return rund::compute::Status::fail(rund::compute::Reason::PipelineInvalid);
  }
  state->prepared = true;
  return rund::compute::Status::success();
}

rund::compute::VirtualBackingTransactionResult MemoryVirtualBacking::commit(
    rund::compute::VirtualBackingTransactionToken &token) noexcept {
  std::lock_guard lock{gate_};
  TransactionState *const state = valid_transaction(token);
  if (state == nullptr || !state->prepared) {
    return rund::compute::VirtualBackingTransactionResult::KnownNoWrite;
  }
  if (transaction_unknown_once_) {
    transaction_unknown_once_ = false;
    return rund::compute::VirtualBackingTransactionResult::UnknownMayWrite;
  }
  bytes_.swap(state->shadow);
  ++facts_.write_count;
  facts_.write_bytes += state->spec.logical_bytes;
  transaction_.reset();
  token = rund::compute::VirtualBackingTransactionToken{};
  publish_transaction_version();
  return rund::compute::VirtualBackingTransactionResult::Success;
}

void MemoryVirtualBacking::abort_known(
    rund::compute::VirtualBackingTransactionToken &&token) noexcept {
  std::lock_guard lock{gate_};
  if (valid_transaction(token) != nullptr) {
    transaction_.reset();
  }
  token = rund::compute::VirtualBackingTransactionToken{};
}

void MemoryVirtualBacking::quarantine_unknown(
    rund::compute::VirtualBackingTransactionToken &&token) noexcept {
  std::lock_guard lock{gate_};
  if (valid_transaction(token) != nullptr) {
    transaction_.reset();
    transaction_quarantined_ = true;
  }
  token = rund::compute::VirtualBackingTransactionToken{};
}

bool MemoryVirtualBacking::seed(
    const std::span<const std::byte> input) noexcept {
  std::lock_guard lock{gate_};
  if (input.size() != logical_bytes_) {
    return false;
  }
  if (!input.empty()) {
    std::memcpy(bytes_.data(), input.data(), input.size());
  }
  return true;
}

bool MemoryVirtualBacking::observe(const std::span<std::byte> output) noexcept {
  if (output.size() != logical_bytes_) {
    return false;
  }
  if (!output.empty()) {
    std::memcpy(output.data(), bytes_.data(), output.size());
  }
  ++facts_.observation_count;
  facts_.observation_bytes += output.size();
  return true;
}

void MemoryVirtualBacking::reset(const std::byte value) noexcept {
  std::lock_guard lock{gate_};
  std::fill(bytes_.begin(), bytes_.end(), value);
}

void MemoryVirtualBacking::fail_next_read(
    const rund::compute::Reason reason) noexcept {
  read_failure_ = reason == rund::compute::Reason::Ok
                      ? rund::compute::Reason::BackendFailed
                      : reason;
}

void MemoryVirtualBacking::fail_next_write(
    const rund::compute::Reason reason) noexcept {
  fail_next_write_after(0u, reason);
}

void MemoryVirtualBacking::fail_next_write_after(
    const std::size_t prefix_bytes,
    const rund::compute::Reason reason) noexcept {
  write_failure_ = reason == rund::compute::Reason::Ok
                       ? rund::compute::Reason::BackendFailed
                       : reason;
  write_failure_prefix_ = prefix_bytes;
}

void MemoryVirtualBacking::fail_next_transaction_unknown() noexcept {
  std::lock_guard lock{gate_};
  transaction_unknown_once_ = true;
}

bool MemoryVirtualBacking::tail_poisoned() const noexcept {
  return logical_bytes_ <= bytes_.size() &&
         std::all_of(bytes_.begin() +
                         static_cast<std::ptrdiff_t>(logical_bytes_),
                     bytes_.end(),
                     [](const std::byte value) { return value == TailPoison; });
}

bool MemoryVirtualBacking::transaction_quarantined() const noexcept {
  std::lock_guard lock{gate_};
  return transaction_quarantined_;
}

bool MemoryVirtualBacking::valid() const noexcept {
  return page_bytes_ != 0u && logical_bytes_ <= bytes_.size() &&
         bytes_.size() - logical_bytes_ != 0u &&
         bytes_.size() % page_bytes_ == 0u && tail_poisoned();
}

MemoryVirtualBacking::TransactionState *MemoryVirtualBacking::valid_transaction(
    const rund::compute::VirtualBackingTransactionToken &token) const noexcept {
  if (!token || transaction_ == nullptr || token.spec() != transaction_->spec ||
      token.generation() != transaction_->generation ||
      token.opaque().get() != transaction_.get()) {
    return nullptr;
  }
  return transaction_.get();
}

} // namespace rund_node_test_virtual::product
