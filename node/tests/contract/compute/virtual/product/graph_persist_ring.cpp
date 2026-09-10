#include "local.hpp"

#include "backing.hpp"
#include "graph_pointwise/internal.hpp"
#include "graph_pointwise_shape/evidence.hpp"

#include "../../../target/selection.hpp"

#include "src/compute/device/residency/pool.hpp"
#include "src/compute/virtual/backing.hpp"
#include "src/compute/virtual/state.hpp"

#include <rund/compute.hpp>
#include <rund/compute/virtual.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <span>
#include <vector>

namespace rund_node_test_virtual::product {
namespace {

class ParallelWriteBacking final : public rund::compute::VirtualBacking,
                                   public rund::compute::VirtualWriteLanes {
public:
  explicit ParallelWriteBacking(const std::size_t bytes) : bytes_(bytes) {}

  [[nodiscard]] std::uint64_t size_bytes() const noexcept override {
    return bytes_.size();
  }

  [[nodiscard]] std::uint32_t write_lanes() const noexcept override {
    return 2u;
  }

  [[nodiscard]] rund::compute::Status
  read(std::uint64_t, std::span<std::byte>) noexcept override {
    return rund::compute::Status::fail(rund::compute::Reason::PipelineInvalid);
  }

  [[nodiscard]] rund::compute::Status
  write(const std::uint64_t offset,
        const std::span<const std::byte> input) noexcept override {
    std::unique_lock lock{gate_};
    if (offset > bytes_.size() || input.size() > bytes_.size() - offset) {
      return rund::compute::Status::fail(rund::compute::Reason::ShapeMismatch);
    }
    ++started_;
    ++active_;
    max_active_ = std::max(max_active_, active_);
    if (active_ >= 2u) {
      paired_ = true;
      ready_.notify_all();
    }
    if (started_ <= 2u && !ready_.wait_for(lock, std::chrono::seconds{5},
                                           [this] { return paired_; })) {
      --active_;
      ready_.notify_all();
      return rund::compute::Status::fail(rund::compute::Reason::BackendFailed);
    }
    std::memcpy(bytes_.data() + offset, input.data(), input.size());
    ++completed_;
    written_bytes_ += input.size();
    --active_;
    ready_.notify_all();
    return rund::compute::Status::success();
  }

  [[nodiscard]] bool observe(const std::span<std::byte> output) noexcept {
    std::lock_guard lock{gate_};
    if (output.size() != bytes_.size() || active_ != 0u) {
      return false;
    }
    std::memcpy(output.data(), bytes_.data(), bytes_.size());
    return true;
  }

  [[nodiscard]] std::size_t max_active() const noexcept {
    std::lock_guard lock{gate_};
    return max_active_;
  }

  [[nodiscard]] std::size_t completed() const noexcept {
    std::lock_guard lock{gate_};
    return completed_;
  }

  [[nodiscard]] std::size_t written_bytes() const noexcept {
    std::lock_guard lock{gate_};
    return written_bytes_;
  }

private:
  mutable std::mutex gate_;
  std::condition_variable ready_;
  std::vector<std::byte> bytes_;
  std::size_t started_{};
  std::size_t completed_{};
  std::size_t written_bytes_{};
  std::size_t active_{};
  std::size_t max_active_{};
  bool paired_{};
};

} // namespace

using GraphPublicationGenerations =
    std::array<std::uint64_t,
               graph_pointwise::StageCount *
                   rund::compute::detail::residency::Pool::BankCount>;

[[nodiscard]] GraphPublicationGenerations graph_publication_generations(
    const std::shared_ptr<rund::compute::detail::VirtualPipelineState>
        &state) noexcept {
  using rund::compute::detail::PipelineState;
  using rund::compute::detail::residency::Pool;
  GraphPublicationGenerations result{};
  result.fill(std::numeric_limits<std::uint64_t>::max());
  if (state == nullptr || state->graph_pipelines.size() !=
                              graph_pointwise::StageCount * Pool::BankCount) {
    return result;
  }

  std::array<std::uint64_t, graph_pointwise::StageCount> primary{};
  graph_pointwise_shape::stage_generations(state, primary);
  for (std::size_t stage = 0u; stage < graph_pointwise::StageCount; ++stage) {
    result[stage * Pool::BankCount] = primary[stage];
    for (std::size_t bank = 1u; bank < Pool::BankCount; ++bank) {
      const std::shared_ptr<PipelineState> &pipeline =
          state->graph_pipelines[stage * Pool::BankCount + bank];
      if (pipeline == nullptr || pipeline->publication == nullptr) {
        result.fill(std::numeric_limits<std::uint64_t>::max());
        return result;
      }
      std::lock_guard state_lock{pipeline->gate};
      std::lock_guard publication_lock{pipeline->publication->gate};
      result[stage * Pool::BankCount + bank] =
          pipeline->publication->generation;
    }
  }
  return result;
}

int CheckProductGraphPersistRing(const rund::compute::Backend backend) {
  using namespace rund::compute;
  if (backend == Backend::Cpu) {
    return 0;
  }
  auto opened = open(rund::node::test_contract::target_for(backend));
  if (!opened) {
    return opened.reason() == Reason::AdapterUnavailable ? 0 : 1;
  }
  auto program = graph_pointwise::build_program(*opened);
  if (!program || !graph_pointwise::validate_program(*program)) {
    return 2;
  }

  constexpr std::size_t PageCount = 5u;
  constexpr std::size_t ElementCount =
      PageCount * graph_pointwise::FrameElements -
      graph_pointwise::TailElements;
  std::vector<std::uint64_t> input_values(ElementCount);
  std::vector<std::uint64_t> expected(ElementCount);
  for (std::size_t index = 0u; index < ElementCount; ++index) {
    input_values[index] = index * 13u + 7u;
    expected[index] =
        input_values[index] * (2u * graph_pointwise::StageLeafCount) +
        (2u * graph_pointwise::StageLeafCount) *
            (2u * graph_pointwise::StageLeafCount + 1u) / 2u;
  }
  const std::size_t bytes = ElementCount * sizeof(std::uint64_t);
  auto input_backing = std::make_shared<MemoryVirtualBacking>(
      bytes, graph_pointwise::FrameElements * sizeof(std::uint64_t));
  auto output_backing = std::make_shared<ParallelWriteBacking>(bytes);
  if (!input_backing->seed(std::as_bytes(std::span{input_values}))) {
    return 3;
  }
  auto input = virtual_buffer<std::uint64_t>(ElementCount, input_backing);
  auto output = virtual_buffer<std::uint64_t>(ElementCount, output_backing);
  auto pipeline =
      input && output
          ? virtual_pipeline(*program, *input, *output, ResidencyConfig{})
          : Result<VirtualPipeline<std::uint64_t(std::uint64_t)>>::fail(
                Reason::PipelineInvalid);
  const std::shared_ptr<detail::VirtualPipelineState> state_before =
      pipeline ? detail::VirtualPipelineAccess::state(*pipeline) : nullptr;
  const GraphPublicationGenerations before_publications =
      graph_publication_generations(state_before);
  const std::uint64_t initial_version =
      detail::VirtualBackingAccess::version(*output_backing);
  const Status status =
      pipeline ? pipeline->run() : Status::fail(pipeline.reason());
  const std::shared_ptr<detail::VirtualPipelineState> state =
      pipeline ? detail::VirtualPipelineAccess::state(*pipeline) : nullptr;
  const GraphPublicationGenerations after_publications =
      graph_publication_generations(state);
  std::vector<std::uint64_t> observed(ElementCount);
  const ResidencyStats residency =
      pipeline ? pipeline->stats().pipeline.residency : ResidencyStats{};
  const std::uint64_t capacity =
      pipeline ? pipeline->plan().residency.frame_capacity : 0u;
  const std::uint64_t batches =
      capacity == 0u ? 0u : (PageCount + capacity - 1u) / capacity;
  bool publication_generations = true;
  for (std::size_t stage = 0u; stage < graph_pointwise::StageCount; ++stage) {
    for (std::size_t bank = 0u;
         bank < rund::compute::detail::residency::Pool::BankCount; ++bank) {
      const std::size_t index =
          stage * rund::compute::detail::residency::Pool::BankCount + bank;
      const std::uint64_t expected_delta =
          batches <= bank
              ? 0u
              : 1u + (batches - 1u - bank) /
                         rund::compute::detail::residency::Pool::BankCount;
      publication_generations = publication_generations &&
                                before_publications[index] !=
                                    std::numeric_limits<std::uint64_t>::max() &&
                                after_publications[index] ==
                                    before_publications[index] + expected_delta;
    }
  }
  const bool route_evidence =
      state != nullptr &&
      state->geometry.route == detail::VirtualRoute::GraphPointwise &&
      !state->geometry.device_vsm_required &&
      state->device_vsm_product_cache == nullptr &&
      residency.window_handoff_count == 0u &&
      residency.window_batch_count == 0u &&
      residency.window_queue_call_count == 0u;
  const bool exact =
      pipeline && status &&
      output_backing->observe(std::as_writable_bytes(std::span{observed})) &&
      observed == expected && route_evidence && publication_generations &&
      capacity == 2u &&
      residency.epoch_count == batches * graph_pointwise::StageCount &&
      residency.page_in_count == PageCount &&
      residency.backing_read_bytes == bytes &&
      output_backing->max_active() == 2u &&
      output_backing->completed() == PageCount &&
      output_backing->written_bytes() == bytes &&
      residency.page_out_count == PageCount &&
      residency.backing_write_bytes == bytes &&
      detail::VirtualBackingAccess::recovery_bytes(*output_backing) == 0u &&
      detail::VirtualBackingAccess::version(*output_backing) ==
          initial_version + 1u;
  if (!exact) {
    std::fprintf(
        stderr,
        "Graph Persist ring backend=%u reason=%.*s active=%zu "
        "completed=%zu bytes=%zu/%zu page_in=%llu read=%llu "
        "page_out=%llu backing=%llu route=%u cache=%u version=%llu/%llu "
        "exact=%u\n",
        static_cast<unsigned>(backend), static_cast<int>(status.error().size()),
        status.error().data(), output_backing->max_active(),
        output_backing->completed(), output_backing->written_bytes(), bytes,
        static_cast<unsigned long long>(residency.page_in_count),
        static_cast<unsigned long long>(residency.backing_read_bytes),
        static_cast<unsigned long long>(residency.page_out_count),
        static_cast<unsigned long long>(residency.backing_write_bytes),
        route_evidence ? 1u : 0u,
        state != nullptr && state->device_vsm_product_cache != nullptr ? 1u
                                                                       : 0u,
        static_cast<unsigned long long>(
            detail::VirtualBackingAccess::version(*output_backing)),
        static_cast<unsigned long long>(initial_version + 1u),
        observed == expected ? 1u : 0u);
    return 4;
  }
  return 0;
}

} // namespace rund_node_test_virtual::product
