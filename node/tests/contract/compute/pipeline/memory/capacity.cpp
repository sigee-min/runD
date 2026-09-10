#include "local.hpp"

#include <node/runtime/compute/access.hpp>

#include "src/compute/memory/arena.hpp"
#include "src/compute/backend.hpp"
#include "src/compute/pipeline/plan/arena.hpp"
#include "../../allocation.hpp"
#include "src/compute/pipeline/plan/local.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace rund_node_test_pipeline::memory {

namespace {

int CheckScratchPages() {
  using namespace rund::compute;
  detail::DeviceState device;
  device.backend = Backend::Metal;
  device.storage.emplace<detail::AccelDeviceState>();
  auto *native = detail::accel_device(device);
  native->pick.caps.storage_alignment = 256u;
  native->pick.backend_info.storage_bytes = 4096u;
  detail::DeviceOps ops{};
  // Exercise the common page projection without allocating native buffers.
  ops.plan_scratch = [](const detail::DeviceState &, const rund::AccelKernel &,
                        std::uint64_t, const std::uint64_t page) {
    rund::node::accel::detail::KernelScratchPlan result{};
    result.ok = true;
    result.page_count = 1024u;
    result.last_bytes = 256u;
    result.backing_bytes = 1023u * page + result.last_bytes;
    result.payload_bytes = result.backing_bytes;
    return result;
  };
  device.ops = &ops;
  detail::ProgramState program;
  program.accel = std::make_unique<detail::AccelProgram>();
  const detail::ProgramState *pointer = &program;
  detail::PipelineMemoryPlan plan;
  plan.view_slots.resize(7u);
  node_compute_allocation::Start();
  const auto status = detail::plan_pipeline_scratch(device, {&pointer, 1u}, plan);
  node_compute_allocation::Stop();
  if (!status || node_compute_allocation::Count() != 1u ||
      plan.scratch.size() != 1024u ||
      plan.summary.scratch_bytes != 1023u * 4096u + 256u ||
      plan.summary.scratch_payload_bytes != plan.summary.scratch_bytes ||
      plan.summary.scratch_count != 1024u) {
    return 41;
  }
  for (std::size_t i = 0u; i < plan.scratch.size(); ++i) {
    if (plan.scratch[i].slot != 7u + i ||
        plan.scratch[i].bytes != (i == 1023u ? 256u : 4096u)) {
      return 42;
    }
  }
  plan.summary.prepared_buffer_bytes = 0u;
  node_compute_allocation::Start();
  const auto reused = detail::plan_pipeline_scratch(device, {&pointer, 1u}, plan);
  node_compute_allocation::Stop();
  return reused && node_compute_allocation::Count() == 0u ? 0 : 43;
}

} // namespace

[[nodiscard]] int CheckCapacity(rund::compute::Device &device) {
  using namespace rund::compute;
  if (const int scratch = CheckScratchPages(); scratch != 0) {
    return scratch;
  }
  const auto device_state = detail::DeviceAccess::state(device);

  const std::array<std::size_t, 2u> chunked_counts{
      static_cast<std::size_t>(detail::memory::ChunkWords -
                               detail::memory::AlignmentWords),
      static_cast<std::size_t>(2u * detail::memory::AlignmentWords)};
  auto chunked = MakeProgram(device, chunked_counts);
  detail::PipelineBuildState chunked_build{};
  chunked_build.device = device_state;
  detail::PipelineBuildStep chunked_step{};
  chunked_step.program = chunked;
  chunked_step.logical_step = 0u;
  chunked_build.steps.push_back(std::move(chunked_step));
  const auto chunked_plan = detail::plan_memory(chunked_build);
  const std::size_t chunked_base = static_cast<std::size_t>(
      detail::memory::ChunkWords - detail::memory::AlignmentWords);
  if (!chunked_plan || (*chunked_plan)->summary.allocation_count != 2u ||
      (*chunked_plan)->summary.transient_bytes !=
          static_cast<std::uint64_t>(detail::memory::ChunkWords +
                                     detail::memory::AlignmentWords) *
              sizeof(std::uint32_t) ||
      (*chunked_plan)->summary.reuse_count != 0u ||
      (*chunked_plan)->steps != std::vector<std::size_t>{0u, 2u} ||
      (*chunked_plan)->owners != std::vector<std::size_t>{0u, 1u} ||
      (*chunked_plan)->offsets != std::vector<std::size_t>{0u, 0u} ||
      (*chunked_plan)->chunks !=
          std::vector<std::size_t>{chunked_base,
                                   2u * detail::memory::AlignmentWords}) {
    return 13;
  }

  const std::array<std::size_t, 2u> wide_counts{100u, 100u};
  const std::array<std::size_t, 2u> tall_counts{150u, 1u};
  auto wide = MakeProgram(device, wide_counts);
  auto tall = MakeProgram(device, tall_counts);
  detail::PipelineBuildState packed{};
  packed.device = device_state;
  detail::PipelineBuildStep wide_step{};
  wide_step.program = wide;
  wide_step.logical_step = 0u;
  packed.steps.push_back(std::move(wide_step));
  detail::PipelineBuildStep tall_step{};
  tall_step.program = tall;
  tall_step.logical_step = 1u;
  packed.steps.push_back(std::move(tall_step));
  const auto packed_plan = detail::plan_memory(packed);
  if (!packed_plan ||
      (*packed_plan)->summary.transient_bytes !=
          228u * sizeof(std::uint32_t) ||
      (*packed_plan)->summary.reuse_count != 3u ||
      (*packed_plan)->steps != std::vector<std::size_t>{0u, 2u, 4u} ||
      (*packed_plan)->owners != std::vector<std::size_t>{0u, 0u, 0u, 0u} ||
      (*packed_plan)->offsets != std::vector<std::size_t>{0u, 128u, 0u, 192u} ||
      (*packed_plan)->chunks != std::vector<std::size_t>{228u}) {
    return 14;
  }

  constexpr std::size_t large_words = static_cast<std::size_t>(
      detail::memory::ChunkWords + detail::memory::AlignmentWords);
  constexpr std::size_t larger_words =
      large_words + static_cast<std::size_t>(detail::memory::AlignmentWords);
  const std::array<std::size_t, 2u> large_counts{large_words, 128u};
  const std::array<std::size_t, 2u> larger_counts{larger_words, 192u};
  auto large = MakeProgram(device, large_counts);
  auto larger = MakeProgram(device, larger_counts);
  detail::PipelineBuildState dedicated{};
  dedicated.device = device_state;
  detail::PipelineBuildStep large_step{};
  large_step.program = large;
  large_step.logical_step = 3u;
  large_step.iteration = 2u;
  dedicated.steps.push_back(std::move(large_step));
  detail::PipelineBuildStep larger_step{};
  larger_step.program = larger;
  larger_step.logical_step = 4u;
  larger_step.iteration = 7u;
  dedicated.steps.push_back(std::move(larger_step));
  const auto dedicated_plan = detail::plan_memory(dedicated);
  const std::uint64_t larger_bytes =
      static_cast<std::uint64_t>(larger_words) * sizeof(std::uint32_t);
  if (!dedicated_plan ||
      (*dedicated_plan)->summary.transient_bytes !=
          static_cast<std::uint64_t>(larger_words + 192u) *
              sizeof(std::uint32_t) ||
      (*dedicated_plan)->summary.allocation_count != 2u ||
      (*dedicated_plan)->summary.reuse_count != 2u ||
      (*dedicated_plan)->summary.largest_bytes != larger_bytes ||
      (*dedicated_plan)->summary.largest_step != 4u ||
      (*dedicated_plan)->summary.largest_iteration != 7u ||
      (*dedicated_plan)->summary.largest_chunk != 0u ||
      (*dedicated_plan)->steps != std::vector<std::size_t>{0u, 2u, 4u} ||
      (*dedicated_plan)->owners != std::vector<std::size_t>{0u, 1u, 0u, 1u} ||
      (*dedicated_plan)->offsets != std::vector<std::size_t>{0u, 0u, 0u, 0u} ||
      (*dedicated_plan)->chunks !=
          std::vector<std::size_t>{larger_words, 192u}) {
    return 15;
  }

  constexpr std::size_t scale_owners = 8u;
  const std::uint64_t storage_words =
      detail::memory::arena_bytes(*device_state) / detail::memory::Word;
  const std::size_t scale_words = static_cast<std::size_t>(
      std::min(storage_words, detail::memory::ChunkWords));
  std::array<std::size_t, scale_owners> scale_counts{};
  scale_counts.fill(scale_words);
  auto scale = MakeProgram(device, scale_counts);
  detail::PipelineBuildState scale_build{};
  scale_build.device = device_state;
  detail::PipelineBuildStep scale_step{};
  scale_step.program = scale;
  scale_step.logical_step = 5u;
  scale_build.steps.push_back(std::move(scale_step));
  const auto scale_plan = detail::plan_memory(scale_build);
  const std::uint64_t scale_bytes =
      static_cast<std::uint64_t>(scale_words) * detail::memory::Word;
  if (!scale_plan ||
      (*scale_plan)->summary.transient_bytes !=
          scale_owners * scale_bytes ||
      (*scale_plan)->summary.allocation_count != scale_owners ||
      (*scale_plan)->summary.reuse_count != 0u ||
      (*scale_plan)->summary.largest_bytes != scale_bytes ||
      (*scale_plan)->chunks.size() != scale_owners) {
    return 19;
  }
  for (std::size_t index = 0u; index < scale_owners; ++index) {
    if ((*scale_plan)->chunks[index] != scale_words ||
        (*scale_plan)->owners[index] != index ||
        (*scale_plan)->offsets[index] != 0u) {
      return 20;
    }
  }
  return 0;
}

} // namespace rund_node_test_pipeline::memory
