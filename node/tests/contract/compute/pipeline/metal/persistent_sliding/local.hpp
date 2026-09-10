#pragma once

#include "../../local.hpp"

#if !defined(RUND_NODE_TEST_BACKEND_CPU) && \
    !defined(RUND_NODE_TEST_BACKEND_VULKAN)

#include "../../device_vsm_product/local.hpp"
#include "../../persistent_product/local.hpp"
#include "../../residency/device_vsm/actual.hpp"
#include "../../residency/service_free_direct/product.hpp"

#include "src/accel/kernel/prepared/model.hpp"
#include "src/accel/metal/buffer/owner.hpp"
#include "src/accel/metal/kernel.hpp"
#include "src/accel/metal/state.hpp"
#include "src/compute/backend.hpp"
#include "src/compute/pipeline/state.hpp"
#include "src/compute/virtual/backing.hpp"
#include "src/compute/virtual/state.hpp"

#include <node/runtime/compute/access.hpp>
#include <rund/compute/virtual.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <span>
#include <utility>
#include <vector>

namespace rund_node_test_pipeline_metal_persistent {

namespace accel = rund::node::accel::detail;

[[nodiscard]] bool DeviceVsmQueueCount(const rund::AccelDevice &pick,
                                       std::uint64_t &count) noexcept;

[[nodiscard]] bool ProductQueueCount(
    const std::shared_ptr<rund::compute::detail::VirtualPipelineState> &state,
    std::uint64_t &count) noexcept;

void IgnorePersistentFinal(
    void *, accel::PersistentResidencySlidingFinal &&) noexcept;

[[nodiscard]] bool AcceptOneSubmit(
    const accel::PersistentResidencySlidingRequest &request,
    accel::PersistentResidencySlidingControl &control) noexcept;

class PersistentBacking final
    : public rund::compute::VirtualBacking,
      public rund::compute::VirtualBackingTransaction {
public:
  using TransactionResult =
      rund::compute::VirtualBackingTransactionResult;
  using TransactionSpec = rund::compute::VirtualBackingTransactionSpec;
  using TransactionToken =
      rund::compute::VirtualBackingTransactionToken;

  explicit PersistentBacking(
      std::size_t bytes, std::uint32_t max_parallel_reads = 1u);

  [[nodiscard]] std::uint64_t size_bytes() const noexcept override;
  [[nodiscard]] std::uint32_t max_parallel_reads() const noexcept override;

  [[nodiscard]] rund::compute::Status
  read(std::uint64_t offset, std::span<std::byte> output) noexcept override;

  [[nodiscard]] rund::compute::Status
  write(std::uint64_t offset,
        std::span<const std::byte> input) noexcept override;

  [[nodiscard]] rund::compute::Status
  begin(TransactionSpec spec, TransactionToken &token) noexcept override;

  [[nodiscard]] rund::compute::Status
  stage(TransactionToken &token,
        std::span<const rund::compute::VirtualWrite> ranges) noexcept override;

  [[nodiscard]] rund::compute::Status
  prepare_commit(const TransactionToken &token) noexcept override;

  [[nodiscard]] TransactionResult
  commit(TransactionToken &token) noexcept override;

  void abort_known(TransactionToken &&token) noexcept override;
  void quarantine_unknown(TransactionToken &&token) noexcept override;

  void fail_next_transaction_stage() noexcept;

  [[nodiscard]] std::uint64_t write_count() const noexcept;
  [[nodiscard]] std::uint64_t write_bytes() const noexcept;

private:
  struct TransactionState;

  [[nodiscard]] TransactionState *
  valid_transaction(const TransactionToken &token) const noexcept;

  mutable std::mutex gate_;
  std::vector<std::byte> bytes_;
  std::uint32_t max_parallel_reads_;
  std::atomic<std::uint64_t> write_count_;
  std::atomic<std::uint64_t> write_bytes_;
  std::shared_ptr<TransactionState> transaction_;
  std::uint64_t next_generation_;
  bool transaction_quarantined_;
  bool fail_next_stage_;
};

struct FinalWait final {
  accel::PersistentResidencySlidingRequest request{};
  accel::PersistentResidencySlidingFinal final{};
  std::atomic<std::uint64_t> callback_count{};
};

void CompletePersistent(
    void *, accel::PersistentResidencySlidingFinal &&) noexcept;

[[nodiscard]] bool RejectOversizedPreparation(
    const accel::PersistentResidencySlidingRequest &request) noexcept;

[[nodiscard]] bool SameFusedDirectStorage(
    const accel::MetalFusedDirectRecurrenceDiagnostics &left,
    const accel::MetalFusedDirectRecurrenceDiagnostics &right) noexcept;

[[nodiscard]] bool SameCurrentCommonMemory(
    const accel::PreparedPipelineMemory &left,
    const accel::PreparedPipelineMemory &right) noexcept;

[[nodiscard]] bool RunPublicSpatialWindowN48();
[[nodiscard]] bool RunNaturalFallbackWindowN53();
[[nodiscard]] bool CheckFusedDirectRecurrence();
[[nodiscard]] bool CheckFusedDirectHistory();

[[nodiscard]] bool RunPersistentCase(std::uint64_t coordinate_count,
                                      std::uint64_t identity,
                                      bool known_failure,
                                      bool unknown_failure,
                                      bool failed_admission_only);

} // namespace rund_node_test_pipeline_metal_persistent
#endif
