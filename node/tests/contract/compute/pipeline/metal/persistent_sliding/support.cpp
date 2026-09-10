#include "local.hpp"

#if !defined(RUND_NODE_TEST_BACKEND_CPU) && \
    !defined(RUND_NODE_TEST_BACKEND_VULKAN)

namespace rund_node_test_pipeline_metal_persistent {

[[nodiscard]] bool DeviceVsmQueueCount(const rund::AccelDevice &pick,
                                       std::uint64_t &count) noexcept {
  const std::shared_ptr<accel::PickToken> token = accel::AdmitPick(pick);
  accel::MetalAdapter *const adapter =
      token == nullptr ? nullptr : accel::MetalAdapterFromPick(token->raw);
  if (adapter == nullptr) {
    return false;
  }
  std::lock_guard lock{adapter->mutex};
  count = adapter->stats.runtime.run.work.command_submit_count;
  return true;
}

[[nodiscard]] bool ProductQueueCount(
    const std::shared_ptr<rund::compute::detail::VirtualPipelineState> &state,
    std::uint64_t &count) noexcept {
  count = 0u;
  auto *const common = state == nullptr || state->pipeline == nullptr ||
                               !state->pipeline->prepared.ok
                           ? nullptr
                           : static_cast<accel::prepared::PipelineState *>(
                                 state->pipeline->prepared.owner.get());
  accel::MetalResidencySlidingDiagnostics diagnostics{};
  if (common == nullptr || common->backend == nullptr ||
      !accel::InspectMetalResidencySliding(common->backend, diagnostics)) {
    return false;
  }
  count = diagnostics.command_submit_count;
  return true;
}

void IgnorePersistentFinal(void *,
                           accel::PersistentResidencySlidingFinal &&) noexcept {
}

// This fixture calls the backend service directly, so it supplies the same
// acceptance transition that the common compute submitter owns in product
// runs.  The backend only activates the control and reports native acceptance.
[[nodiscard]] bool AcceptOneSubmit(
    const accel::PersistentResidencySlidingRequest &request,
    accel::PersistentResidencySlidingControl &control) noexcept {
  std::lock_guard lock{control.gate};
  if (!control.active || control.quarantined ||
      control.mode != accel::PersistentResidencySlidingMode::OneSubmit ||
      request.chunk_count != 0u ||
      control.coordinate_count != request.coordinate_count ||
      control.accepted_end != 0u || control.native_submit_count != 0u ||
      control.queue_calls != 0u || control.chunk_submit_count != 0u) {
    return false;
  }
  control.accepted_end = request.coordinate_count;
  control.native_submit_count = 1u;
  control.queue_calls = 1u;
  control.chunk_submit_count = 1u;
  return true;
}

struct PersistentBacking::TransactionState final {
  PersistentBacking::TransactionSpec spec{};
  std::vector<std::byte> shadow{};
  std::vector<bool> covered{};
  std::uint64_t generation{};
  std::uint64_t staged_bytes{};
  bool prepared{};
};

PersistentBacking::PersistentBacking(const std::size_t bytes,
                                     const std::uint32_t max_parallel_reads)
    : bytes_(bytes), max_parallel_reads_(max_parallel_reads),
      write_count_{}, write_bytes_{}, transaction_{}, next_generation_{},
      transaction_quarantined_{}, fail_next_stage_{} {}

  [[nodiscard]] std::uint64_t PersistentBacking::size_bytes() const noexcept {
    return bytes_.size();
  }

  [[nodiscard]] std::uint32_t PersistentBacking::max_parallel_reads() const noexcept {
    return max_parallel_reads_;
  }

  [[nodiscard]] rund::compute::Status
  PersistentBacking::read(const std::uint64_t offset,
       const std::span<std::byte> output) noexcept {
    std::lock_guard lock{gate_};
    if (offset > bytes_.size() || output.size() > bytes_.size() - offset) {
      return rund::compute::Status::fail(
          rund::compute::Reason::TransferInvalid);
    }
    std::memcpy(output.data(), bytes_.data() + offset, output.size());
    return rund::compute::Status::success();
  }

  [[nodiscard]] rund::compute::Status
  PersistentBacking::write(const std::uint64_t offset,
        const std::span<const std::byte> input) noexcept {
    std::lock_guard lock{gate_};
    if (offset > bytes_.size() || input.size() > bytes_.size() - offset) {
      return rund::compute::Status::fail(
          rund::compute::Reason::TransferInvalid);
    }
    std::memcpy(bytes_.data() + offset, input.data(), input.size());
    write_count_.fetch_add(1u, std::memory_order_relaxed);
    write_bytes_.fetch_add(input.size(), std::memory_order_relaxed);
    return rund::compute::Status::success();
  }

  [[nodiscard]] rund::compute::Status
  PersistentBacking::begin(const TransactionSpec spec, TransactionToken &token) noexcept {
    std::lock_guard lock{gate_};
    constexpr std::uint64_t backing_page_bytes = 64u;
    constexpr std::uint64_t write_page_bytes = 48u;
    const std::uint64_t expected_backing_pages =
        spec.logical_bytes / backing_page_bytes +
        (spec.logical_bytes % backing_page_bytes != 0u);
    const std::uint64_t expected_write_pages =
        spec.logical_bytes / write_page_bytes +
        (spec.logical_bytes % write_page_bytes != 0u);
    if (transaction_quarantined_ || transaction_ != nullptr ||
        spec.logical_bytes != bytes_.size() ||
        spec.backing_page_bytes != backing_page_bytes ||
        spec.backing_page_count != expected_backing_pages ||
        spec.write_page_bytes != write_page_bytes ||
        spec.write_page_count != expected_write_pages ||
        expected_backing_pages == 0u) {
      return rund::compute::Status::fail(
          rund::compute::Reason::PipelineCapacity);
    }
    try {
      auto next = std::make_shared<TransactionState>();
      next->spec = spec;
      next->generation = ++next_generation_;
      next->shadow = bytes_;
      next->covered.assign(bytes_.size(), false);
      transaction_ = std::move(next);
      token = TransactionToken{transaction_, spec, transaction_->generation};
    } catch (const std::bad_alloc &) {
      return rund::compute::Status::fail(
          rund::compute::Reason::PipelineCapacity);
    }
    return rund::compute::Status::success();
  }

  [[nodiscard]] rund::compute::Status
  PersistentBacking::stage(TransactionToken &token,
        const std::span<const rund::compute::VirtualWrite> ranges) noexcept {
    std::lock_guard lock{gate_};
    TransactionState *const state = valid_transaction(token);
    if (state == nullptr || state->prepared || ranges.empty()) {
      return rund::compute::Status::fail(
          rund::compute::Reason::PipelineInvalid);
    }
    if (fail_next_stage_) {
      fail_next_stage_ = false;
      return rund::compute::Status::fail(
          rund::compute::Reason::TileCallbackFailed);
    }
    for (const rund::compute::VirtualWrite range : ranges) {
      if (range.bytes.empty() || range.offset > state->spec.logical_bytes ||
          range.bytes.size() > state->spec.logical_bytes - range.offset) {
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

  [[nodiscard]] rund::compute::Status
  PersistentBacking::prepare_commit(const TransactionToken &token) noexcept {
    std::lock_guard lock{gate_};
    TransactionState *const state = valid_transaction(token);
    if (state == nullptr || state->staged_bytes != state->spec.logical_bytes ||
        std::find(state->covered.begin(), state->covered.end(), false) !=
            state->covered.end()) {
      return rund::compute::Status::fail(
          rund::compute::Reason::PipelineInvalid);
    }
    state->prepared = true;
    return rund::compute::Status::success();
  }

  [[nodiscard]] PersistentBacking::TransactionResult
  PersistentBacking::commit(TransactionToken &token) noexcept {
    std::lock_guard lock{gate_};
    TransactionState *const state = valid_transaction(token);
    if (state == nullptr || !state->prepared) {
      return TransactionResult::KnownNoWrite;
    }
    bytes_.swap(state->shadow);
    write_count_.fetch_add(1u, std::memory_order_relaxed);
    write_bytes_.fetch_add(state->spec.logical_bytes,
                           std::memory_order_relaxed);
    transaction_.reset();
    token = TransactionToken{};
    publish_transaction_version();
    return TransactionResult::Success;
  }

  void PersistentBacking::abort_known(TransactionToken &&token) noexcept {
    std::lock_guard lock{gate_};
    if (valid_transaction(token) != nullptr) {
      transaction_.reset();
    }
    token = TransactionToken{};
  }

  void PersistentBacking::quarantine_unknown(TransactionToken &&token) noexcept {
    std::lock_guard lock{gate_};
    if (valid_transaction(token) != nullptr) {
      transaction_.reset();
      transaction_quarantined_ = true;
    }
    token = TransactionToken{};
  }

  void PersistentBacking::fail_next_transaction_stage() noexcept {
    std::lock_guard lock{gate_};
    fail_next_stage_ = true;
  }

  [[nodiscard]] std::uint64_t PersistentBacking::write_count() const noexcept {
    return write_count_.load(std::memory_order_relaxed);
  }

  [[nodiscard]] std::uint64_t PersistentBacking::write_bytes() const noexcept {
    return write_bytes_.load(std::memory_order_relaxed);
  }

  [[nodiscard]] PersistentBacking::TransactionState *
  PersistentBacking::valid_transaction(const TransactionToken &token) const noexcept {
    if (!token || transaction_ == nullptr ||
        token.spec() != transaction_->spec ||
        token.generation() != transaction_->generation ||
        token.opaque().get() != transaction_.get()) {
      return nullptr;
    }
    return transaction_.get();
  }

void CompletePersistent(
    void *const raw, accel::PersistentResidencySlidingFinal &&final) noexcept {
  auto *const wait = static_cast<FinalWait *>(raw);
  if (wait == nullptr) {
    return;
  }
  wait->final = std::move(final);
  wait->callback_count.fetch_add(1u, std::memory_order_release);
}

[[nodiscard]] bool SameFusedDirectStorage(
    const accel::MetalFusedDirectRecurrenceDiagnostics &left,
    const accel::MetalFusedDirectRecurrenceDiagnostics &right) noexcept {
  return left.native_command_count == right.native_command_count &&
         left.native_dispatch_count == right.native_dispatch_count &&
         left.native_storage_bytes == right.native_storage_bytes &&
         left.route_host_bytes == right.route_host_bytes &&
         left.retained_bytes == right.retained_bytes &&
         left.iteration_argument_bytes == right.iteration_argument_bytes;
}

[[nodiscard]] bool
SameCurrentCommonMemory(const accel::PreparedPipelineMemory &left,
                        const accel::PreparedPipelineMemory &right) noexcept {
  return left.host.current == right.host.current &&
         left.device.current == right.device.current &&
         left.staging.current == right.staging.current;
}

} // namespace rund_node_test_pipeline_metal_persistent

#endif
