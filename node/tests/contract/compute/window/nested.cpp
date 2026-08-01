#include "../pipeline/local.hpp"
#include "local.hpp"
#include "nested/local.hpp"

#include <node/runtime/compute/access.hpp>

#include "src/compute/pipeline/state.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <span>
#include <utility>
#include <vector>

namespace rund::node::test_contract::window {
namespace {

constexpr std::size_t kMaximum = 10u;
constexpr std::size_t kTile = 4u;
constexpr std::size_t kInner = 3u;
constexpr std::size_t kDomain = 64u;
constexpr std::size_t kOuter = CeilDiv(kMaximum, kTile);
constexpr std::size_t kTemplates = kOuter + kInner + 3u;
constexpr std::size_t kCommands = kOuter * (kInner + 2u);
constexpr std::uint32_t kOuterSeed = 7u;
constexpr std::uint32_t kSentinel = 0xA5A55A5Au;
constexpr std::array<std::uint32_t, kMaximum> kQueue{63u, 0u,  47u, 47u, 1u,
                                                     62u, 31u, 5u,  60u, 2u};
constexpr auto kDomainValues = [] {
  std::array<std::uint32_t, kDomain> values{};
  for (std::size_t index = 0u; index < values.size(); ++index) {
    values[index] = static_cast<std::uint32_t>(3u * index + 1u);
  }
  return values;
}();
constexpr std::array<std::uint32_t, 7u> kCounts{
    0u,
    1u,
    static_cast<std::uint32_t>(kTile - 1u),
    static_cast<std::uint32_t>(kTile),
    static_cast<std::uint32_t>(kTile + 1u),
    static_cast<std::uint32_t>(kMaximum),
    static_cast<std::uint32_t>(kMaximum + 1u)};

static_assert(kOuter == 3u);
static_assert(kOuter * kInner < rund::compute::PipelineIterationCapacity);

template <std::size_t Maximum, std::size_t Tile>
[[nodiscard]] auto SeedProgram(rund::compute::Device &device) {
  using namespace rund::compute;
  return on(device)
      .input<std::uint32_t>(Maximum)
      .zip_input<std::uint32_t>(kDomain)
      .zip_input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(1u)
      .branch([](auto queue, auto domain, auto total, auto ordinal) {
        auto current = resident<Maximum, Tile>(total, ordinal);
        auto active_ordinals = queue.gather(current.items());
        auto sum = domain.gather(active_ordinals).reduce(Reduce::Sum);
        return outputs(sum, current.count());
      })
      .compile();
}

[[nodiscard]] auto ActionProgram(rund::compute::Device &device) {
  using namespace rund::compute;
  return on(device)
      .input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(1u)
      .branch([](auto value, auto tile_count) {
        return value.combine(
            "nested-window-action", tile_count,
            [](auto current, auto count) { return current + count; });
      })
      .compile();
}

[[nodiscard]] auto MemoryActionProgram(rund::compute::Device &device) {
  using namespace rund::compute;
  return on(device)
      .input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(1u)
      .branch([](auto value, auto tile_count) {
        auto prefix = value.scan(Scan::InclusiveSum);
        return prefix.combine(
            "nested-window-memory-action", tile_count,
            [](auto current, auto count) { return current + count; });
      })
      .compile();
}

[[nodiscard]] auto FoldProgram(rund::compute::Device &device) {
  using namespace rund::compute;
  return on(device)
      .input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(1u)
      .branch([](auto outer, auto tile, auto tile_count) {
        auto checked = tile.combine(
            "nested-window-fold-tail", tile_count,
            [](auto value, auto count) { return value + count * 0u; });
        return outer.combine(
            "nested-window-fold", checked,
            [](auto left, auto right) { return left + right; });
      })
      .compile();
}

[[nodiscard]] auto TerminalSeedProgram(rund::compute::Device &device) {
  using namespace rund::compute;
  return on(device)
      .input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(1u)
      .branch([](auto total, auto ordinal) {
        (void)total;
        return ordinal.map("nested-window-terminal-seed",
                           [](auto value) { return value + 1u; });
      })
      .compile();
}

[[nodiscard]] auto TerminalFoldProgram(rund::compute::Device &device) {
  using namespace rund::compute;
  return on(device)
      .input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(1u)
      .branch([](auto outer, auto terminal, auto tile) {
        auto next =
            outer.combine("nested-window-terminal-fold", tile,
                          [](auto left, auto right) { return left + right; });
        auto stopped = terminal.map("nested-window-terminal-value",
                                    [](auto value) { return value * 0u + 7u; });
        return outputs(next, stopped);
      })
      .compile();
}

template <bool Fault>
[[nodiscard]] auto FailureSeedProgram(rund::compute::Device &device) {
  using namespace rund::compute;
  return on(device)
      .input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(1u)
      .branch([](auto total, auto ordinal) {
        (void)total;
        if constexpr (Fault) {
          auto fault =
              ordinal.map("nested-window-seed-fault-index", [](auto value) {
                return select(value == 1u, 1u, 0u);
              });
          auto checked = ordinal.gather(fault).scalar();
          return checked.map("nested-window-seed-fault-value",
                             [](auto value) { return value * 16u; });
        } else {
          return ordinal.map("nested-window-failure-seed",
                             [](auto value) { return value * 16u; });
        }
      })
      .compile();
}

template <std::uint32_t Target>
[[nodiscard]] auto FailureActionProgram(rund::compute::Device &device) {
  using namespace rund::compute;
  return on(device)
      .input<std::uint32_t>(1u)
      .branch([](auto value) {
        auto fault =
            value.map("nested-window-action-fault-index", [](auto current) {
              return select(current == Target, 1u, 0u);
            });
        auto checked = value.gather(fault).scalar();
        return checked.map("nested-window-action-fault-value",
                           [](auto current) { return current + 1u; });
      })
      .compile();
}

template <bool Fault>
[[nodiscard]] auto FailureFoldProgram(rund::compute::Device &device) {
  using namespace rund::compute;
  return on(device)
      .input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(1u)
      .branch([](auto outer, auto tile) {
        if constexpr (Fault) {
          auto fault =
              tile.map("nested-window-fold-fault-index",
                       [](auto value) { return select(value == 19u, 1u, 0u); });
          auto checked = tile.gather(fault).scalar();
          return outer.combine(
              "nested-window-fold-fault", checked,
              [](auto left, auto right) { return left + right; });
        } else {
          return outer.combine(
              "nested-window-failure-fold", tile,
              [](auto left, auto right) { return left + right; });
        }
      })
      .compile();
}

[[nodiscard]] constexpr std::uint32_t
SerialOracle(const std::uint32_t count) noexcept {
  std::uint32_t result = kOuterSeed;
  for (std::size_t index = 0u; index < count; ++index) {
    result += kDomainValues[kQueue[index]];
  }
  result += count * static_cast<std::uint32_t>(kInner);
  return result;
}

[[nodiscard]] bool WarmSetupClean(const rund::compute::Stats &stats) noexcept {
  return stats.pipeline_compiles == 0u && stats.buffer_allocations == 0u &&
         stats.pipeline_cache_evictions == 0u &&
         stats.descriptor_pool_creations == 0u &&
         stats.descriptor_set_allocations == 0u && stats.uploaded_bytes == 0u &&
         stats.download_events == 0u && stats.downloaded_bytes == 0u &&
         stats.output_hash == 0u;
}

template <class Seed, class Action, class Fold>
[[nodiscard]] bool PlanShape(const rund::compute::PipelinePlan &plan,
                             const Seed &seed, const Action &action,
                             const Fold &fold) noexcept {
  constexpr std::uint64_t internal_bytes =
      (kOuter + 5u) * sizeof(std::uint32_t);
  const std::uint64_t infrastructure = plan.state_bytes + plan.prepared_bytes;
  const std::uint64_t logical_workspace =
      kOuter * (seed.graph().memory.logical_bytes +
                kInner * action.graph().memory.logical_bytes +
                fold.graph().memory.logical_bytes);
  const std::uint64_t live_workspace = std::max(
      {seed.graph().memory.live_bytes, action.graph().memory.live_bytes,
       fold.graph().memory.live_bytes});
  return plan.outer_window_count == kOuter && plan.tile_capacity == kTile &&
         plan.inner_iteration_count == kInner &&
         plan.prepared_template_count == kTemplates &&
         plan.prepared_command_count == kCommands &&
         plan.barrier_count == kTemplates - 1u && plan.resource_count == 11u &&
         plan.state_bytes == internal_bytes && plan.publish_count == 1u &&
         plan.publish_bytes == sizeof(std::uint32_t) &&
         plan.logical_bytes == infrastructure + logical_workspace &&
         plan.live_bytes == infrastructure + live_workspace &&
         plan.physical_bytes == infrastructure + plan.transient_bytes &&
         plan.physical_bytes == plan.peak_bytes &&
         plan.total_bytes == plan.persistent_bytes + plan.physical_bytes;
}

[[nodiscard]] bool PreparedShape(const rund::compute::Pipeline &pipeline) {
  using namespace rund::compute::detail;
  const std::shared_ptr<PipelineState> &state =
      PipelineStateAccess::state(pipeline);
  if (state == nullptr || state->steps.size() != kTemplates ||
      state->windows.size() != 1u || state->resources.size() != 11u ||
      state->publications.size() != 1u) {
    return false;
  }
  const PipelineWindow &nested = state->windows[0u];
  if (!nested.nested || nested.begin != 0u || nested.end != kTemplates ||
      nested.seed_first != 0u || nested.seed_count != kOuter ||
      nested.action_first != kOuter || nested.action_count != kInner ||
      nested.fold_first != kOuter + kInner || nested.maximum != kMaximum ||
      nested.tile != kTile) {
    return false;
  }
  for (std::size_t index = 0u; index < state->steps.size(); ++index) {
    const PipelineRoute expected =
        index < kOuter ? PipelineRoute::NestedSeed
                       : (index < kOuter + kInner ? PipelineRoute::NestedAction
                                                  : PipelineRoute::NestedFold);
    if (state->steps[index].route != expected ||
        state->steps[index].job == nullptr) {
      return false;
    }
  }

  std::array<const JobState *, kTemplates> owners{};
  std::size_t owner_count = 0u;
  for (const PipelineStep &step : state->steps) {
    const JobState *const owner = step.job.get();
    const auto found =
        std::find(owners.begin(), owners.begin() + owner_count, owner);
    if (found == owners.begin() + owner_count) {
      owners[owner_count++] = owner;
    }
  }
  constexpr std::size_t expected_owners =
      kOuter + (kInner == 1u ? 1u : 2u) + 3u;
  return owner_count == expected_owners &&
         state->steps[kOuter].job == state->steps[kOuter + 2u].job;
}

[[nodiscard]] std::size_t
ActionOwnerCount(const rund::compute::Pipeline &pipeline,
                 const rund::compute::graph::Fingerprint action) {
  using namespace rund::compute::detail;
  const std::shared_ptr<PipelineState> &state =
      PipelineStateAccess::state(pipeline);
  if (state == nullptr || state->windows.size() != 1u) {
    return std::numeric_limits<std::size_t>::max();
  }
  const PipelineWindow &nested = state->windows[0u];
  std::array<const JobState *, 2u> owners{};
  std::size_t owner_count = 0u;
  for (std::size_t index = nested.action_first;
       index < nested.action_first + nested.action_count; ++index) {
    if (index >= state->steps.size() ||
        state->steps[index].program == nullptr ||
        state->steps[index].job == nullptr ||
        state->steps[index].program->graph_info.fingerprint != action) {
      return std::numeric_limits<std::size_t>::max();
    }
    const JobState *const owner = state->steps[index].job.get();
    const auto found =
        std::find(owners.begin(), owners.begin() + owner_count, owner);
    if (found == owners.begin() + owner_count) {
      if (owner_count == owners.size()) {
        return std::numeric_limits<std::size_t>::max();
      }
      owners[owner_count++] = owner;
    }
  }
  return owner_count;
}

// Test-only structural oracle. Warm production execution never performs this
// compact owner walk; the snapshot makes zero rebinding_count independently
// falsifiable without adding an O(K + N) scan to the runtime path.
struct JobViewIdentity final {
  std::size_t offset{};
  std::size_t count{};
  std::size_t stride{};
  std::size_t element_bytes{};
  std::size_t alignment{};

  [[nodiscard]] bool
  operator==(const JobViewIdentity &) const noexcept = default;
};

struct KernelViewIdentity final {
  std::uint64_t binding{};
  std::size_t slot{};
  std::uint64_t bytes{};

  [[nodiscard]] bool
  operator==(const KernelViewIdentity &) const noexcept = default;
};

struct ArenaSlotIdentity final {
  std::size_t words{};
  std::size_t owner{};
  std::size_t offset_words{};

  [[nodiscard]] bool
  operator==(const ArenaSlotIdentity &) const noexcept = default;
};

struct ScratchIdentity final {
  std::size_t slot{};
  std::uint64_t bytes{};

  [[nodiscard]] bool
  operator==(const ScratchIdentity &) const noexcept = default;
};

struct ResidentBindingIdentity final {
  const void *handle{};
  std::uint64_t id{};
  std::uint64_t bytes{};
  std::uint64_t offset_bytes{};
  std::uint64_t element_bytes{};
  std::uint64_t stride_bytes{};
  std::uint64_t count{};
  std::uint32_t usage{};

  [[nodiscard]] bool
  operator==(const ResidentBindingIdentity &) const noexcept = default;
};

struct CpuTransferIdentity final {
  const rund::compute::detail::BufferState *external{};
  const rund::compute::detail::BufferState *staging{};
  JobViewIdentity view{};

  [[nodiscard]] bool
  operator==(const CpuTransferIdentity &) const noexcept = default;
};

struct JobBindingIdentity final {
  const rund::compute::detail::JobState *owner{};
  const rund::compute::detail::ProgramState *program{};
  rund::compute::graph::Fingerprint program_fingerprint{};
  const rund::compute::detail::JobWorkspace *workspace{};
  const rund::compute::detail::ProgramState *workspace_program{};
  const rund::compute::detail::JobArena *arena{};
  const void *prepared_owner{};
  const void *write_prepared_owner{};
  bool prepared_ok{};
  bool write_prepared_ok{};
  bool arena_bound{};
  bool arena_binds_heap{};
  bool arena_binds_ok{};
  std::vector<const rund::compute::detail::BufferState *> inputs;
  std::vector<const rund::compute::detail::BufferState *> write_inputs;
  std::vector<const rund::compute::detail::BufferState *> graph_buffers;
  std::vector<const rund::compute::detail::BufferState *> effective_graph;
  std::vector<const rund::compute::detail::BufferState *> outputs;
  std::vector<const rund::compute::detail::BufferState *> cpu_view_buffers;
  std::vector<const rund::compute::detail::BufferState *> workspace_buffers;
  std::vector<const rund::compute::detail::BufferState *> arena_buffers;
  std::vector<std::size_t> workspace_offsets;
  std::vector<JobViewIdentity> input_views;
  std::vector<JobViewIdentity> output_views;
  std::vector<KernelViewIdentity> kernel_views;
  std::vector<CpuTransferIdentity> cpu_inputs;
  std::vector<CpuTransferIdentity> cpu_outputs;
  std::vector<ArenaSlotIdentity> arena_slots;
  std::vector<ResidentBindingIdentity> arena_bindings;
  std::vector<ScratchIdentity> scratch;

  [[nodiscard]] bool
  operator==(const JobBindingIdentity &) const noexcept = default;
};

struct StepBindingIdentity final {
  const rund::compute::detail::ProgramState *program{};
  const rund::compute::detail::JobState *normal{};
  const rund::compute::detail::JobState *alternate{};
  rund::compute::detail::PipelineRoute route{
      rund::compute::detail::PipelineRoute::Ordinary};
  std::uint32_t logical_step{};
  std::uint32_t iteration{};
  std::uint16_t window{};

  [[nodiscard]] bool
  operator==(const StepBindingIdentity &) const noexcept = default;
};

struct PipelineBindingIdentity final {
  const rund::compute::detail::PipelineState *owner{};
  const void *prepared_owner{};
  const void *alternate_prepared_owner{};
  bool prepared_ok{};
  bool alternate_prepared_ok{};
  bool transactional{};
  bool valid{};
  std::vector<StepBindingIdentity> steps;
  std::vector<const rund::compute::detail::JobState *> normal_jobs;
  std::vector<const rund::compute::detail::JobState *> alternate_jobs;
  std::vector<JobBindingIdentity> jobs;
  std::vector<const rund::compute::detail::BufferState *> resources;
  std::vector<const rund::compute::detail::BufferState *> shared_buffers;
  std::vector<const rund::compute::detail::BufferState *> prepared_buffers;
  std::vector<const rund::compute::detail::BufferState *> claims;
  std::vector<const rund::compute::detail::BufferState *> alternate_claims;
  std::vector<const rund::compute::detail::BufferState *> state_banks;
  std::vector<const rund::compute::detail::BufferState *> publications;
  std::vector<const rund::compute::detail::BufferState *> window_counts;

  [[nodiscard]] bool
  operator==(const PipelineBindingIdentity &) const noexcept = default;
};

[[nodiscard]] constexpr JobViewIdentity
BindingView(const rund::compute::detail::JobBufferView view) noexcept {
  return JobViewIdentity{.offset = view.offset,
                         .count = view.count,
                         .stride = view.stride,
                         .element_bytes = view.element_bytes,
                         .alignment = view.alignment};
}

void CaptureBufferOwners(
    const std::vector<std::shared_ptr<rund::compute::detail::BufferState>>
        &source,
    std::vector<const rund::compute::detail::BufferState *> &target,
    bool &valid) {
  target.reserve(source.size());
  for (const auto &owner : source) {
    target.push_back(owner.get());
    valid = valid && owner != nullptr;
  }
}

[[nodiscard]] JobBindingIdentity
CaptureJobBinding(const rund::compute::detail::JobState &job,
                  const rund::compute::Backend backend, bool &valid) {
  using namespace rund::compute::detail;
  JobBindingIdentity result{};
  result.owner = &job;
  result.program = job.program.get();
  result.program_fingerprint = job.program == nullptr
                                   ? rund::compute::graph::Fingerprint{}
                                   : job.program->graph_info.fingerprint;
  result.workspace = job.workspace.get();
  result.workspace_program =
      job.workspace == nullptr ? nullptr : job.workspace->program.get();
  result.arena =
      job.workspace == nullptr ? nullptr : job.workspace->arena.get();
  result.prepared_owner = job.prepared.owner.get();
  result.write_prepared_owner = job.write_prepared.owner.get();
  result.prepared_ok = job.prepared.ok;
  result.write_prepared_ok = job.write_prepared.ok;
  valid = valid && job.program != nullptr &&
          (job.workspace == nullptr || job.workspace->program == job.program);
  if (backend != rund::compute::Backend::Cpu) {
    valid = valid && job.prepared.ok && job.prepared.owner != nullptr;
  }

  CaptureBufferOwners(job.inputs, result.inputs, valid);
  CaptureBufferOwners(job.write_inputs, result.write_inputs, valid);
  CaptureBufferOwners(job.graph_buffers, result.graph_buffers, valid);
  CaptureBufferOwners(job.outputs, result.outputs, valid);
  CaptureBufferOwners(job.cpu_view_buffers, result.cpu_view_buffers, valid);
  const auto graph = job_graph_buffers(job);
  result.effective_graph.reserve(graph.size());
  for (const auto &owner : graph) {
    result.effective_graph.push_back(owner.get());
    valid = valid && owner != nullptr;
  }
  result.input_views.reserve(job.input_views.size());
  for (const JobBufferView view : job.input_views) {
    result.input_views.push_back(BindingView(view));
  }
  result.output_views.reserve(job.output_views.size());
  for (const JobBufferView view : job.output_views) {
    result.output_views.push_back(BindingView(view));
  }
  valid = valid && result.input_views.size() == result.inputs.size() &&
          result.output_views.size() == result.outputs.size();
  result.kernel_views.reserve(job.views.size());
  for (const rund::node::accel::detail::KernelViewSlot view : job.views) {
    result.kernel_views.push_back(KernelViewIdentity{
        .binding = view.binding, .slot = view.slot, .bytes = view.bytes});
  }
  const auto capture_transfers = [&](const std::vector<CpuViewTransfer> &source,
                                     std::vector<CpuTransferIdentity> &target) {
    target.reserve(source.size());
    for (const CpuViewTransfer &transfer : source) {
      target.push_back(CpuTransferIdentity{.external = transfer.external.get(),
                                           .staging = transfer.staging.get(),
                                           .view = BindingView(transfer.view)});
      valid =
          valid && transfer.external != nullptr && transfer.staging != nullptr;
    }
  };
  capture_transfers(job.cpu_view_inputs, result.cpu_inputs);
  capture_transfers(job.cpu_view_outputs, result.cpu_outputs);

  if (job.workspace != nullptr) {
    CaptureBufferOwners(job.workspace->buffers, result.workspace_buffers,
                        valid);
    result.workspace_offsets = job.workspace->offsets;
    valid = valid &&
            result.workspace_buffers.size() == result.workspace_offsets.size();
  }
  if (result.arena != nullptr) {
    const JobArena &arena = *job.workspace->arena;
    result.arena_bound = arena.bound;
    result.arena_binds_heap = arena.binds.heap;
    result.arena_binds_ok = arena.binds.ok;
    CaptureBufferOwners(arena.buffers, result.arena_buffers, valid);
    result.arena_slots.reserve(arena.slots.size());
    for (const JobArenaSlot slot : arena.slots) {
      result.arena_slots.push_back(ArenaSlotIdentity{
          .words = slot.words,
          .owner = slot.owner,
          .offset_words = slot.offset_words,
      });
    }
    result.scratch.reserve(arena.scratch.size());
    for (const rund::node::accel::detail::KernelScratchPage page :
         arena.scratch) {
      result.scratch.push_back(
          ScratchIdentity{.slot = page.slot, .bytes = page.bytes});
    }
    valid = valid && arena.binds.valid();
    const auto *const refs = arena.binds.refs();
    const auto *const handles = arena.binds.handles();
    if (arena.binds.size() != 0u && (refs == nullptr || handles == nullptr)) {
      valid = false;
    } else {
      result.arena_bindings.reserve(
          static_cast<std::size_t>(arena.binds.size()));
      for (std::uint64_t index = 0u; index < arena.binds.size(); ++index) {
        const rund::kernel::ResidentBufferRef &ref = refs[index];
        result.arena_bindings.push_back(ResidentBindingIdentity{
            .handle = handles[index].get(),
            .id = ref.id,
            .bytes = ref.bytes,
            .offset_bytes = ref.offset_bytes,
            .element_bytes = ref.element_bytes,
            .stride_bytes = ref.stride_bytes,
            .count = ref.count,
            .usage = ref.usage,
        });
        valid = valid && handles[index] != nullptr;
      }
    }
  }
  return result;
}

[[nodiscard]] PipelineBindingIdentity
CaptureBindingIdentity(const rund::compute::Pipeline &pipeline,
                       const rund::compute::Backend backend) {
  using namespace rund::compute::detail;
  PipelineBindingIdentity result{};
  const std::shared_ptr<PipelineState> &state =
      PipelineStateAccess::state(pipeline);
  result.owner = state.get();
  result.valid = state != nullptr && state->device != nullptr &&
                 state->device->backend == backend;
  if (state == nullptr) {
    return result;
  }
  result.prepared_owner = state->prepared.owner.get();
  result.alternate_prepared_owner = state->alternate_prepared.owner.get();
  result.prepared_ok = state->prepared.ok;
  result.alternate_prepared_ok = state->alternate_prepared.ok;
  result.transactional = state->transactional;
  if (backend != rund::compute::Backend::Cpu) {
    result.valid =
        result.valid && state->prepared.ok && state->prepared.owner != nullptr;
    if (state->transactional) {
      result.valid = result.valid && state->alternate_prepared.ok &&
                     state->alternate_prepared.owner != nullptr &&
                     state->prepared.owner != state->alternate_prepared.owner;
    } else {
      result.valid = result.valid && !state->alternate_prepared.ok &&
                     state->alternate_prepared.owner == nullptr;
    }
  }

  const auto append_unique = [](const JobState *const owner,
                                std::vector<const JobState *> &owners) {
    if (std::find(owners.begin(), owners.end(), owner) == owners.end()) {
      owners.push_back(owner);
    }
  };
  const auto capture_job = [&](const std::shared_ptr<JobState> &job) {
    if (job == nullptr) {
      result.valid = false;
      return;
    }
    const auto found =
        std::find_if(result.jobs.begin(), result.jobs.end(),
                     [&](const auto &id) { return id.owner == job.get(); });
    if (found == result.jobs.end()) {
      result.jobs.push_back(CaptureJobBinding(*job, backend, result.valid));
    }
  };
  result.steps.reserve(state->steps.size());
  for (const PipelineStep &step : state->steps) {
    result.steps.push_back(StepBindingIdentity{
        .program = step.program.get(),
        .normal = step.job.get(),
        .alternate = step.alternate_job.get(),
        .route = step.route,
        .logical_step = step.logical_step,
        .iteration = step.iteration,
        .window = step.window,
    });
    result.valid = result.valid && step.program != nullptr &&
                   step.job != nullptr && step.job->program == step.program;
    append_unique(step.job.get(), result.normal_jobs);
    capture_job(step.job);
    if (state->transactional) {
      result.valid = result.valid && step.alternate_job != nullptr &&
                     step.alternate_job != step.job &&
                     step.alternate_job->program == step.program &&
                     step.alternate_job->workspace == step.job->workspace;
      append_unique(step.alternate_job.get(), result.alternate_jobs);
      capture_job(step.alternate_job);
    } else if (step.alternate_job != nullptr) {
      result.valid = false;
    }
  }
  if (state->transactional) {
    result.valid = result.valid &&
                   result.normal_jobs.size() == result.alternate_jobs.size();
  }
  CaptureBufferOwners(state->shared_buffers, result.shared_buffers,
                      result.valid);
  CaptureBufferOwners(state->prepared_buffers, result.prepared_buffers,
                      result.valid);
  result.resources.reserve(state->resources.size());
  for (const PipelineResource &resource : state->resources) {
    result.resources.push_back(resource.buffer.get());
    result.valid = result.valid && resource.buffer != nullptr;
  }
  result.claims.reserve(state->claims.size());
  for (const BufferClaim claim : state->claims) {
    result.claims.push_back(claim.buffer);
    result.valid = result.valid && claim.buffer != nullptr;
  }
  result.alternate_claims.reserve(state->alternate_claims.size());
  for (const BufferClaim claim : state->alternate_claims) {
    result.alternate_claims.push_back(claim.buffer);
    result.valid = result.valid && claim.buffer != nullptr;
  }
  result.valid = result.valid && state->publication != nullptr;
  const std::span<const PipelineStatePair> state_pairs =
      state->publication == nullptr
          ? std::span<const PipelineStatePair>{}
          : std::span<const PipelineStatePair>{state->publication->state_pairs};
  result.state_banks.reserve(state_pairs.size() * 2u);
  for (const PipelineStatePair &pair : state_pairs) {
    result.state_banks.push_back(pair.first.get());
    result.state_banks.push_back(pair.second.get());
    result.valid =
        result.valid && pair.first != nullptr && pair.second != nullptr;
  }
  result.publications.reserve(state->publications.size() * 2u);
  for (const PipelinePublish &publication : state->publications) {
    result.publications.push_back(publication.source.get());
    result.publications.push_back(publication.target.get());
    result.valid = result.valid && publication.source != nullptr &&
                   publication.target != nullptr;
  }
  result.window_counts.reserve(state->windows.size());
  for (const PipelineWindow &window : state->windows) {
    result.window_counts.push_back(window.count.get());
    result.valid = result.valid && window.count != nullptr;
  }
  return result;
}

[[nodiscard]] bool RuntimeShape(const rund::compute::Stats &stats,
                                const rund::compute::Backend backend,
                                const std::uint32_t count) noexcept {
  using rund::compute::PipelineStats;
  const std::uint64_t active = CeilDiv(count, kTile);
  const std::uint64_t submits =
      backend == rund::compute::Backend::Cpu ? 0u : 1u;
  const bool physical_shape =
      backend != rund::compute::Backend::Metal ||
      (stats.dispatches == 2u && stats.pipeline.control_command_count == 1u);
  const bool bounded_telemetry =
      backend == rund::compute::Backend::Cpu
          ? stats.control.generated_item_count == 0u &&
                stats.control.generated_capacity == 0u &&
                stats.control.indirect_dispatch_count == 0u &&
                stats.control.indirect_work_item_count == 0u
          : stats.control.generated_item_count == 2u * count &&
                stats.control.generated_capacity == 2u * active * kTile &&
                stats.control.indirect_dispatch_count == 2u * active &&
                stats.control.indirect_work_item_count == 2u * count;
  return physical_shape && bounded_telemetry &&
         stats.control.overflow_ordinal ==
             rund::compute::ControlStats::no_overflow &&
         stats.pipeline.status_entry_count ==
             (backend == rund::compute::Backend::Cpu ? 0u : 3u * kOuter) &&
         stats.command_submits == submits && stats.pipeline.step_count == 1u &&
         stats.pipeline.verified_step_count == 1u &&
         stats.pipeline.failed_step_index == PipelineStats::no_failed_step &&
         stats.pipeline.executed_outer_window_count == active &&
         stats.pipeline.skipped_outer_window_count == kOuter - active &&
         stats.pipeline.executed_inner_iteration_count == active * kInner &&
         stats.pipeline.skipped_inner_iteration_count ==
             (kOuter - active) * kInner &&
         stats.pipeline.failed_outer_window == PipelineStats::no_coordinate &&
         stats.pipeline.failed_inner_iteration ==
             PipelineStats::no_coordinate &&
         stats.pipeline.barrier_count == kTemplates - 1u &&
         stats.pipeline.prepared_template_count == kTemplates &&
         stats.pipeline.prepared_command_count == kCommands &&
         stats.pipeline.rebinding_count == 0u &&
         stats.control.iteration_count == active &&
         stats.control.skipped_iteration_count == kOuter - active &&
         WarmSetupClean(stats);
}

template <class Seed, class Action, class Fold>
[[nodiscard]] int
CheckCount(rund::compute::Device &device, const rund::compute::Backend backend,
           const Seed &seed, const Action &action, const Fold &fold,
           const std::uint32_t count_value) {
  using namespace rund::compute;
  const std::array<std::uint32_t, 1u> initial{kOuterSeed};
  const std::array<std::uint32_t, 1u> count_values{count_value};
  const std::array<std::uint32_t, 1u> output_values{kSentinel};
  auto outer =
      device.upload<std::uint32_t>(std::span<const std::uint32_t>{initial});
  auto queue =
      device.upload<std::uint32_t>(std::span<const std::uint32_t>{kQueue});
  auto domain = device.upload<std::uint32_t>(
      std::span<const std::uint32_t>{kDomainValues});
  auto count = device.upload<std::uint32_t>(
      std::span<const std::uint32_t>{count_values});
  auto output = device.upload<std::uint32_t>(
      std::span<const std::uint32_t>{output_values});
  if (!outer || !queue || !domain || !count || !output) {
    return 1;
  }

  const auto body = tile_repeat<kInner>(seed, action, fold);
  auto builder = pipeline(device);
  builder.windows<kMaximum, kTile>(body, rund::compute::window(*count),
                                   read(*outer, *queue, *domain),
                                   write_final(*output));
  const auto plan = builder.plan();
  if (!plan || !PlanShape(*plan, seed, action, fold)) {
    if (plan) {
      std::fprintf(
          stderr,
          "nested plan backend=%u count=%u outer=%llu tile=%llu inner=%llu "
          "templates=%llu commands=%llu barriers=%llu resources=%llu "
          "state=%llu "
          "logical/live/physical=%llu/%llu/%llu\n",
          static_cast<unsigned>(backend), count_value,
          static_cast<unsigned long long>(plan->outer_window_count),
          static_cast<unsigned long long>(plan->tile_capacity),
          static_cast<unsigned long long>(plan->inner_iteration_count),
          static_cast<unsigned long long>(plan->prepared_template_count),
          static_cast<unsigned long long>(plan->prepared_command_count),
          static_cast<unsigned long long>(plan->barrier_count),
          static_cast<unsigned long long>(plan->resource_count),
          static_cast<unsigned long long>(plan->state_bytes),
          static_cast<unsigned long long>(plan->logical_bytes),
          static_cast<unsigned long long>(plan->live_bytes),
          static_cast<unsigned long long>(plan->physical_bytes));
    } else {
      std::fprintf(stderr,
                   "nested plan rejected backend=%u count=%u reason=%u\n",
                   static_cast<unsigned>(backend), count_value,
                   static_cast<unsigned>(plan.reason()));
    }
    return 2;
  }

  auto prepared = std::move(builder)
                      .budget(MemoryBudget{.bytes = plan->peak_bytes})
                      .prepare();
  if (!prepared || prepared->plan() != *plan || !PreparedShape(*prepared)) {
    std::fprintf(stderr,
                 "nested prepare backend=%u count=%u status=%u reason=%u\n",
                 static_cast<unsigned>(backend), count_value,
                 static_cast<unsigned>(prepared.ok()),
                 static_cast<unsigned>(prepared.reason()));
    return 3;
  }
  const PipelineBindingIdentity frozen_bindings =
      CaptureBindingIdentity(*prepared, backend);
  if (!frozen_bindings.valid) {
    std::fprintf(
        stderr,
        "nested frozen bindings backend=%u count=%u valid=0 "
        "transactional=%u prepared=%u:%p alternate=%u:%p jobs=%llu/%llu/%llu\n",
        static_cast<unsigned>(backend), count_value,
        static_cast<unsigned>(frozen_bindings.transactional),
        static_cast<unsigned>(frozen_bindings.prepared_ok),
        frozen_bindings.prepared_owner,
        static_cast<unsigned>(frozen_bindings.alternate_prepared_ok),
        frozen_bindings.alternate_prepared_owner,
        static_cast<unsigned long long>(frozen_bindings.normal_jobs.size()),
        static_cast<unsigned long long>(frozen_bindings.alternate_jobs.size()),
        static_cast<unsigned long long>(frozen_bindings.jobs.size()));
    return 3;
  }

  if (count_value > kMaximum) {
    const Status failed = prepared->run();
    const Stats stats = prepared->stats();
    const bool bindings_unchanged =
        CaptureBindingIdentity(*prepared, backend) == frozen_bindings;
    const std::uint64_t submits = backend == Backend::Cpu ? 0u : 1u;
    if (failed || failed.reason() != Reason::BoundedCountInvalid ||
        prepared->generation() != 0u || stats.command_submits != submits ||
        stats.pipeline.verified_step_count != 0u ||
        stats.pipeline.failed_step_index != 0u ||
        stats.pipeline.executed_outer_window_count != 0u ||
        stats.pipeline.executed_inner_iteration_count != 0u ||
        stats.pipeline.rebinding_count != 0u ||
        stats.control.overflow_ordinal != kMaximum || !bindings_unchanged) {
      std::fprintf(
          stderr,
          "nested overflow backend=%u status=%u reason=%u generation=%llu "
          "submits=%llu verified=%llu failed=%llu outer=%llu inner=%llu "
          "ordinal=%llu mutation=%llu bindings=%u\n",
          static_cast<unsigned>(backend), static_cast<unsigned>(failed.ok()),
          static_cast<unsigned>(failed.reason()),
          static_cast<unsigned long long>(prepared->generation()),
          static_cast<unsigned long long>(stats.command_submits),
          static_cast<unsigned long long>(stats.pipeline.verified_step_count),
          static_cast<unsigned long long>(stats.pipeline.failed_step_index),
          static_cast<unsigned long long>(
              stats.pipeline.executed_outer_window_count),
          static_cast<unsigned long long>(
              stats.pipeline.executed_inner_iteration_count),
          static_cast<unsigned long long>(stats.control.overflow_ordinal),
          static_cast<unsigned long long>(stats.pipeline.rebinding_count),
          static_cast<unsigned>(bindings_unchanged));
      return 4;
    }
    auto observer = on(device)
                        .map<std::uint32_t>("nested-window-observe", 1u,
                                            [](auto value) { return value; })
                        .compile();
    auto scratch = device.buffer<std::uint32_t>(1u);
    std::array<std::uint32_t, 1u> actual{};
    if (!observer || !scratch ||
        !Observe(*observer, *output, *scratch, actual) ||
        actual != output_values) {
      std::fprintf(stderr,
                   "nested overflow publication backend=%u actual=%u "
                   "expected=%u\n",
                   static_cast<unsigned>(backend), actual[0], kSentinel);
      return 5;
    }
    return 0;
  }

  const Status first = prepared->run();
  std::array<std::uint32_t, 1u> first_output{};
  const Status first_read = prepared->read(*output, first_output);
  const bool first_bindings_unchanged =
      CaptureBindingIdentity(*prepared, backend) == frozen_bindings;
  if (!first || !first_read || !first_bindings_unchanged ||
      first_output[0] != SerialOracle(count_value)) {
    std::fprintf(stderr,
                 "nested first backend=%u count=%u status=%u reason=%u "
                 "actual=%u expected=%u bindings=%u\n",
                 static_cast<unsigned>(backend), count_value,
                 static_cast<unsigned>(first.ok()),
                 static_cast<unsigned>(first.reason()), first_output[0],
                 SerialOracle(count_value),
                 static_cast<unsigned>(first_bindings_unchanged));
    return 6;
  }

  const MemoryStats before = prepared->memory();
  const Status warm = prepared->run();
  const Stats warm_stats = prepared->stats();
  const MemoryStats after = prepared->memory();
  std::array<std::uint32_t, 1u> warm_output{};
  const Status warm_read = prepared->read(*output, warm_output);
  const bool warm_bindings_unchanged =
      CaptureBindingIdentity(*prepared, backend) == frozen_bindings;
  if (!warm || !RuntimeShape(warm_stats, backend, count_value) ||
      !rund_node_test_pipeline::SameMemory(before, after) ||
      prepared->generation() != 2u || !warm_read || !warm_bindings_unchanged ||
      warm_output[0] != SerialOracle(count_value)) {
    std::fprintf(
        stderr,
        "nested warm backend=%u count=%u status=%u reason=%u output=%u/%u "
        "generation=%llu outer=%llu/%llu inner=%llu/%llu "
        "templates=%llu commands=%llu dispatches=%llu control=%llu rebind=%llu "
        "iterations=%llu/%llu alloc=%llu compile=%llu bindings=%u\n",
        static_cast<unsigned>(backend), count_value,
        static_cast<unsigned>(warm.ok()), static_cast<unsigned>(warm.reason()),
        warm_output[0], SerialOracle(count_value),
        static_cast<unsigned long long>(prepared->generation()),
        static_cast<unsigned long long>(
            warm_stats.pipeline.executed_outer_window_count),
        static_cast<unsigned long long>(
            warm_stats.pipeline.skipped_outer_window_count),
        static_cast<unsigned long long>(
            warm_stats.pipeline.executed_inner_iteration_count),
        static_cast<unsigned long long>(
            warm_stats.pipeline.skipped_inner_iteration_count),
        static_cast<unsigned long long>(
            warm_stats.pipeline.prepared_template_count),
        static_cast<unsigned long long>(
            warm_stats.pipeline.prepared_command_count),
        static_cast<unsigned long long>(warm_stats.dispatches),
        static_cast<unsigned long long>(
            warm_stats.pipeline.control_command_count),
        static_cast<unsigned long long>(warm_stats.pipeline.rebinding_count),
        static_cast<unsigned long long>(warm_stats.control.iteration_count),
        static_cast<unsigned long long>(
            warm_stats.control.skipped_iteration_count),
        static_cast<unsigned long long>(warm_stats.buffer_allocations),
        static_cast<unsigned long long>(warm_stats.pipeline_compiles),
        static_cast<unsigned>(warm_bindings_unchanged));
    return 7;
  }
  return 0;
}

[[nodiscard]] int
CheckTransactionalBindingIdentity(rund::compute::Device &device,
                                  const rund::compute::Backend backend) {
  using namespace rund::compute;
  constexpr std::array<std::uint32_t, 1u> initial{kOuterSeed};
  constexpr std::array<std::uint32_t, 1u> pending{kSentinel};
  auto increment =
      on(device)
          .map<std::uint32_t>("nested-binding-identity-increment", 1u,
                              [](auto value) { return value + 1u; })
          .compile();
  auto first_bank = device.upload<std::uint32_t>(initial);
  auto second_bank = device.upload<std::uint32_t>(pending);
  if (!increment || !first_bank || !second_bank) {
    return 1;
  }

  auto builder = pipeline(device);
  builder.state(*first_bank, *second_bank)
      .repeat<kInner>(*increment, read(*first_bank), write_final(*second_bank))
      .commit();
  const auto plan = builder.plan();
  if (!plan) {
    return 2;
  }
  auto prepared = std::move(builder)
                      .budget(MemoryBudget{.bytes = plan->peak_bytes})
                      .prepare();
  if (!prepared) {
    std::fprintf(stderr, "nested transactional prepare backend=%u reason=%u\n",
                 static_cast<unsigned>(backend),
                 static_cast<unsigned>(prepared.reason()));
    return 3;
  }
  const PipelineBindingIdentity frozen =
      CaptureBindingIdentity(*prepared, backend);
  if (!frozen.valid || !frozen.transactional || frozen.normal_jobs.empty() ||
      frozen.normal_jobs.size() != frozen.alternate_jobs.size()) {
    std::fprintf(stderr,
                 "nested transactional identity backend=%u valid=%u tx=%u "
                 "jobs=%llu/%llu\n",
                 static_cast<unsigned>(backend),
                 static_cast<unsigned>(frozen.valid),
                 static_cast<unsigned>(frozen.transactional),
                 static_cast<unsigned long long>(frozen.normal_jobs.size()),
                 static_cast<unsigned long long>(frozen.alternate_jobs.size()));
    return 4;
  }

  std::array<std::uint32_t, 1u> first_value{};
  const Status first_run = prepared->run();
  const Status first_read = prepared->read(*second_bank, first_value);
  const bool first_identity =
      CaptureBindingIdentity(*prepared, backend) == frozen;
  constexpr std::uint32_t first_expected = kOuterSeed + kInner;
  if (!first_run || !first_read || !first_identity ||
      first_value[0u] != first_expected) {
    std::fprintf(stderr,
                 "nested transactional first backend=%u status=%u/%u "
                 "value=%u/%u identity=%u\n",
                 static_cast<unsigned>(backend),
                 static_cast<unsigned>(first_run.ok()),
                 static_cast<unsigned>(first_run.reason()), first_value[0u],
                 first_expected, static_cast<unsigned>(first_identity));
    return 5;
  }

  std::array<std::uint32_t, 1u> second_value{};
  const Status second_run = prepared->run();
  const Stats stats = prepared->stats();
  const Status second_read = prepared->read(*second_bank, second_value);
  const bool second_identity =
      CaptureBindingIdentity(*prepared, backend) == frozen;
  constexpr std::uint32_t second_expected = kOuterSeed + 2u * kInner;
  if (!second_run || !second_read || !second_identity ||
      prepared->generation() != 2u || second_value[0u] != second_expected ||
      stats.pipeline.rebinding_count != 0u || !WarmSetupClean(stats)) {
    std::fprintf(
        stderr,
        "nested transactional warm backend=%u status=%u/%u generation=%llu "
        "value=%u/%u mutation=%llu identity=%u\n",
        static_cast<unsigned>(backend), static_cast<unsigned>(second_run.ok()),
        static_cast<unsigned>(second_run.reason()),
        static_cast<unsigned long long>(prepared->generation()),
        second_value[0u], second_expected,
        static_cast<unsigned long long>(stats.pipeline.rebinding_count),
        static_cast<unsigned>(second_identity));
    return 6;
  }
  return 0;
}

[[nodiscard]] int CheckNestedTerminal(rund::compute::Device &device,
                                      const rund::compute::Backend backend) {
  using namespace rund::compute;
  auto seed = TerminalSeedProgram(device);
  auto action = on(device)
                    .input<std::uint32_t>(1u)
                    .branch([](auto value) {
                      auto index =
                          value.map("nested-window-terminal-action-index",
                                    [](auto current) { return current * 0u; });
                      return value.gather(index).scalar().map(
                          "nested-window-terminal-action",
                          [](auto current) { return current + 1u; });
                    })
                    .compile();
  auto fold = TerminalFoldProgram(device);
  if (!seed || !action || !fold) {
    return 1;
  }

  const auto run_case = [&]<std::size_t Maximum, std::size_t Tile>(
                            const std::uint32_t initial_terminal,
                            const std::uint32_t count_value,
                            const std::uint64_t active, Stats *const observed) {
    constexpr std::array<std::uint32_t, 1u> outer_values{kOuterSeed};
    const std::array<std::uint32_t, 1u> count_values{count_value};
    const std::array<std::uint32_t, 1u> terminal_values{initial_terminal};
    auto outer = device.upload<std::uint32_t>(
        std::span<const std::uint32_t>{outer_values});
    auto terminal = device.upload<std::uint32_t>(
        std::span<const std::uint32_t>{terminal_values});
    auto count = device.upload<std::uint32_t>(
        std::span<const std::uint32_t>{count_values});
    auto output = device.buffer<std::uint32_t>(1u);
    auto stopped = device.buffer<std::uint32_t>(1u);
    if (!outer || !terminal || !count || !output || !stopped) {
      return 1;
    }
    const auto body = tile_repeat<kInner>(*seed, *action, *fold);
    auto builder = pipeline(device);
    builder.windows<Maximum, Tile>(
        body, rund::compute::window(*count).until<1u>(7u),
        read(*outer, *terminal), write_final(*output, *stopped));
    const auto plan = builder.plan();
    if (!plan) {
      return 2;
    }
    auto prepared = std::move(builder)
                        .budget(MemoryBudget{.bytes = plan->peak_bytes})
                        .prepare();
    std::array<std::uint32_t, 1u> output_actual{};
    std::array<std::uint32_t, 1u> terminal_actual{};
    const Status ran =
        prepared ? prepared->run() : Status::fail(prepared.reason());
    const Stats stats = prepared ? prepared->stats() : Stats{};
    constexpr std::uint64_t outer_count = CeilDiv(Maximum, Tile);
    const std::uint32_t expected_output =
        active == 0u ? kOuterSeed : kOuterSeed + 1u + kInner;
    if (!prepared || !ran || prepared->generation() != 1u ||
        !prepared->read(*output, output_actual) ||
        !prepared->read(*stopped, terminal_actual) ||
        output_actual[0] != expected_output || terminal_actual[0] != 7u ||
        stats.command_submits != (backend == Backend::Cpu ? 0u : 1u) ||
        stats.pipeline.verified_step_count != 1u ||
        stats.pipeline.failed_step_index != PipelineStats::no_failed_step ||
        stats.pipeline.executed_outer_window_count != active ||
        stats.pipeline.skipped_outer_window_count != outer_count - active ||
        stats.pipeline.executed_inner_iteration_count != active * kInner ||
        stats.pipeline.skipped_inner_iteration_count !=
            (outer_count - active) * kInner ||
        stats.control.iteration_count != active ||
        stats.control.skipped_iteration_count != outer_count - active) {
      std::fprintf(
          stderr,
          "nested terminal backend=%u initial=%u prepared=%u status=%u "
          "reason=%u generation=%llu output=%u/%u terminal=%u "
          "outer=%llu/%llu inner=%llu/%llu control=%llu/%llu "
          "verified=%llu failed=%llu submits=%llu\n",
          static_cast<unsigned>(backend), initial_terminal,
          static_cast<unsigned>(prepared.ok()), static_cast<unsigned>(ran.ok()),
          static_cast<unsigned>(ran.reason()),
          static_cast<unsigned long long>(prepared ? prepared->generation()
                                                   : 0u),
          output_actual[0], expected_output, terminal_actual[0],
          static_cast<unsigned long long>(
              stats.pipeline.executed_outer_window_count),
          static_cast<unsigned long long>(
              stats.pipeline.skipped_outer_window_count),
          static_cast<unsigned long long>(
              stats.pipeline.executed_inner_iteration_count),
          static_cast<unsigned long long>(
              stats.pipeline.skipped_inner_iteration_count),
          static_cast<unsigned long long>(stats.control.iteration_count),
          static_cast<unsigned long long>(
              stats.control.skipped_iteration_count),
          static_cast<unsigned long long>(stats.pipeline.verified_step_count),
          static_cast<unsigned long long>(stats.pipeline.failed_step_index),
          static_cast<unsigned long long>(stats.command_submits));
      return 3;
    }
    if (observed != nullptr) {
      *observed = stats;
    }
    return 0;
  };

  const int initial =
      run_case.template operator()<kMaximum, kTile>(7u, kMaximum, 0u, nullptr);
  if (initial != 0) {
    return 10 + initial;
  }
  Stats produced_stats{};
  const int produced = run_case.template operator()<kMaximum, kTile>(
      0u, kMaximum, 1u, &produced_stats);
  if (produced != 0) {
    return 20 + produced;
  }
  Stats single_stats{};
  const int single =
      run_case.template operator()<kTile, kTile>(0u, kTile, 1u, &single_stats);
  if (single != 0) {
    return 30 + single;
  }
  const ControlStats &produced_control = produced_stats.control;
  const ControlStats &single_control = single_stats.control;
  if (produced_control.generated_item_count !=
          single_control.generated_item_count ||
      produced_control.generated_capacity !=
          single_control.generated_capacity ||
      produced_control.indirect_dispatch_count !=
          single_control.indirect_dispatch_count ||
      produced_control.indirect_work_item_count !=
          single_control.indirect_work_item_count ||
      produced_control.conflict_count != single_control.conflict_count ||
      produced_control.overflow_ordinal != single_control.overflow_ordinal) {
    std::fprintf(
        stderr,
        "nested terminal payload backend=%u generated=%llu/%llu "
        "capacity=%llu/%llu indirect=%llu/%llu work=%llu/%llu "
        "conflict=%llu/%llu overflow=%llu/%llu\n",
        static_cast<unsigned>(backend),
        static_cast<unsigned long long>(produced_control.generated_item_count),
        static_cast<unsigned long long>(single_control.generated_item_count),
        static_cast<unsigned long long>(produced_control.generated_capacity),
        static_cast<unsigned long long>(single_control.generated_capacity),
        static_cast<unsigned long long>(
            produced_control.indirect_dispatch_count),
        static_cast<unsigned long long>(single_control.indirect_dispatch_count),
        static_cast<unsigned long long>(
            produced_control.indirect_work_item_count),
        static_cast<unsigned long long>(
            single_control.indirect_work_item_count),
        static_cast<unsigned long long>(produced_control.conflict_count),
        static_cast<unsigned long long>(single_control.conflict_count),
        static_cast<unsigned long long>(produced_control.overflow_ordinal),
        static_cast<unsigned long long>(single_control.overflow_ordinal));
    return 40;
  }
  return 0;
}

template <class Seed, class Action, class Fold>
[[nodiscard]] int CheckNestedFailureCase(
    rund::compute::Device &device, const rund::compute::Backend backend,
    const Seed &seed, const Action &action, const Fold &fold,
    const rund::compute::PipelineNestedPhase phase, const std::uint64_t inner,
    const std::uint64_t executed_inner) {
  using namespace rund::compute;
  constexpr std::array<std::uint32_t, 1u> initial{kOuterSeed};
  constexpr std::array<std::uint32_t, 1u> count_values{kMaximum};
  constexpr std::array<std::uint32_t, 1u> output_values{kSentinel};
  auto outer =
      device.upload<std::uint32_t>(std::span<const std::uint32_t>{initial});
  auto count = device.upload<std::uint32_t>(
      std::span<const std::uint32_t>{count_values});
  auto output = device.upload<std::uint32_t>(
      std::span<const std::uint32_t>{output_values});
  auto observer = on(device)
                      .map<std::uint32_t>("nested-window-failure-observe", 1u,
                                          [](auto value) { return value; })
                      .compile();
  auto scratch = device.buffer<std::uint32_t>(1u);
  if (!outer || !count || !output || !observer || !scratch) {
    return 1;
  }
  const auto body = tile_repeat<kInner>(seed, action, fold);
  auto builder = pipeline(device);
  builder.windows<kMaximum, kTile>(body, rund::compute::window(*count),
                                   read(*outer), write_final(*output));
  const auto plan = builder.plan();
  if (!plan) {
    return 2;
  }
  auto prepared = std::move(builder)
                      .budget(MemoryBudget{.bytes = plan->peak_bytes})
                      .prepare();
  std::array<std::uint32_t, 1u> actual{};
  const Status failed =
      prepared ? prepared->run() : Status::fail(prepared.reason());
  const Stats stats = prepared ? prepared->stats() : Stats{};
  if (!prepared || failed || failed.reason() != Reason::GatherIndexOutOfRange ||
      prepared->generation() != 0u || prepared->poisoned() ||
      !Observe(*observer, *output, *scratch, actual) ||
      actual != output_values ||
      stats.command_submits != (backend == Backend::Cpu ? 0u : 1u) ||
      stats.pipeline.verified_step_count != 0u ||
      stats.pipeline.failed_step_index != 0u ||
      stats.pipeline.failed_outer_window != 1u ||
      stats.pipeline.failed_inner_iteration != inner ||
      stats.pipeline.failed_nested_phase != phase ||
      stats.pipeline.executed_outer_window_count != 1u ||
      stats.pipeline.skipped_outer_window_count != 0u ||
      stats.pipeline.executed_inner_iteration_count != executed_inner ||
      stats.pipeline.skipped_inner_iteration_count != 0u ||
      stats.control.iteration_count != 1u ||
      stats.publication.discard_count != 1u) {
    std::fprintf(
        stderr,
        "nested failure backend=%u phase=%u prepared=%u status=%u reason=%u "
        "generation=%llu poison=%u output=%u/%u verified=%llu failed=%llu "
        "coords=%llu/%llu expected_inner=%llu outer=%llu/%llu "
        "inner_work=%llu/%llu control=%llu discard=%llu submits=%llu\n",
        static_cast<unsigned>(backend), static_cast<unsigned>(phase),
        static_cast<unsigned>(prepared.ok()),
        static_cast<unsigned>(failed.ok()),
        static_cast<unsigned>(failed.reason()),
        static_cast<unsigned long long>(prepared ? prepared->generation() : 0u),
        static_cast<unsigned>(prepared ? prepared->poisoned() : false),
        actual[0], kSentinel,
        static_cast<unsigned long long>(stats.pipeline.verified_step_count),
        static_cast<unsigned long long>(stats.pipeline.failed_step_index),
        static_cast<unsigned long long>(stats.pipeline.failed_outer_window),
        static_cast<unsigned long long>(stats.pipeline.failed_inner_iteration),
        static_cast<unsigned long long>(inner),
        static_cast<unsigned long long>(
            stats.pipeline.executed_outer_window_count),
        static_cast<unsigned long long>(
            stats.pipeline.skipped_outer_window_count),
        static_cast<unsigned long long>(
            stats.pipeline.executed_inner_iteration_count),
        static_cast<unsigned long long>(executed_inner),
        static_cast<unsigned long long>(stats.control.iteration_count),
        static_cast<unsigned long long>(stats.publication.discard_count),
        static_cast<unsigned long long>(stats.command_submits));
    return 3;
  }
  return 0;
}

[[nodiscard]] int CheckNestedFailures(rund::compute::Device &device,
                                      const rund::compute::Backend backend) {
  using namespace rund::compute;
  auto seed = FailureSeedProgram<false>(device);
  auto seed_fault = FailureSeedProgram<true>(device);
  auto action = on(device)
                    .map<std::uint32_t>("nested-window-failure-action", 1u,
                                        [](auto value) { return value + 1u; })
                    .compile();
  auto action_first = FailureActionProgram<16u>(device);
  auto action_middle = FailureActionProgram<17u>(device);
  auto action_last = FailureActionProgram<18u>(device);
  auto fold = FailureFoldProgram<false>(device);
  auto fold_fault = FailureFoldProgram<true>(device);
  if (!seed || !seed_fault || !action || !action_first || !action_middle ||
      !action_last || !fold || !fold_fault) {
    return 1;
  }
  constexpr std::uint64_t none = PipelineStats::no_coordinate;
  if (const int result =
          CheckNestedFailureCase(device, backend, *seed_fault, *action, *fold,
                                 PipelineNestedPhase::Seed, none, kInner);
      result != 0) {
    return 10 + result;
  }
  if (const int result =
          CheckNestedFailureCase(device, backend, *seed, *action_first, *fold,
                                 PipelineNestedPhase::Action, 0u, kInner);
      result != 0) {
    return 20 + result;
  }
  if (const int result =
          CheckNestedFailureCase(device, backend, *seed, *action_middle, *fold,
                                 PipelineNestedPhase::Action, 1u, kInner + 1u);
      result != 0) {
    return 30 + result;
  }
  if (const int result =
          CheckNestedFailureCase(device, backend, *seed, *action_last, *fold,
                                 PipelineNestedPhase::Action, 2u, kInner + 2u);
      result != 0) {
    return 40 + result;
  }
  const int fold_result =
      CheckNestedFailureCase(device, backend, *seed, *action, *fold_fault,
                             PipelineNestedPhase::Fold, none, kInner * 2u);
  return fold_result == 0 ? 0 : 50 + fold_result;
}

template <class Seed, class Action, class Fold>
[[nodiscard]] int CheckNestedProfile(rund::compute::Device &device,
                                     const rund::compute::Backend backend,
                                     const Seed &seed, const Action &action,
                                     const Fold &fold) {
  using namespace rund::compute;
  constexpr std::uint32_t active_count = kTile + 1u;
  constexpr std::array<std::uint32_t, 1u> initial{kOuterSeed};
  constexpr std::array<std::uint32_t, 1u> count_values{active_count};
  auto outer =
      device.upload<std::uint32_t>(std::span<const std::uint32_t>{initial});
  auto queue =
      device.upload<std::uint32_t>(std::span<const std::uint32_t>{kQueue});
  auto domain = device.upload<std::uint32_t>(
      std::span<const std::uint32_t>{kDomainValues});
  auto count = device.upload<std::uint32_t>(
      std::span<const std::uint32_t>{count_values});
  auto output = device.buffer<std::uint32_t>(1u);
  if (!outer || !queue || !domain || !count || !output) {
    return 1;
  }

  const auto body = tile_repeat<kInner>(seed, action, fold);
  auto builder = pipeline(device);
  builder.profile(PipelineProfile::Steps)
      .windows<kMaximum, kTile>(body, rund::compute::window(*count),
                                read(*outer, *queue, *domain),
                                write_final(*output));
  const auto plan = builder.plan();
  if (!plan) {
    return 2;
  }
  auto prepared = std::move(builder)
                      .budget(MemoryBudget{.bytes = plan->peak_bytes})
                      .prepare();
  std::array<std::uint32_t, 1u> actual{};
  const Status ran =
      prepared ? prepared->run() : Status::fail(prepared.reason());
  std::array<PipelineStepProfile, kTemplates> rows{};
  const auto profile =
      prepared ? prepared->profile(rows)
               : Result<PipelineProfileSnapshot>::fail(prepared.reason());
  if (!prepared || !ran || !profile || profile->written != kTemplates ||
      profile->total != kTemplates || !prepared->read(*output, actual) ||
      actual[0] != SerialOracle(active_count) ||
      (backend == Backend::Metal &&
       (profile->execution.dispatches != 2u ||
        profile->execution.pipeline.control_command_count != 1u)) ||
      profile->execution.pipeline.executed_outer_window_count != 2u ||
      profile->execution.pipeline.executed_inner_iteration_count !=
          2u * kInner ||
      profile->execution.command_submits !=
          (backend == Backend::Cpu ? 0u : 1u)) {
    std::fprintf(
        stderr,
        "nested profile backend=%u prepared=%u run=%u/%u profile=%u/%u "
        "rows=%llu/%llu output=%u/%u outer=%llu inner=%llu submits=%llu "
        "dispatches=%llu control=%llu\n",
        static_cast<unsigned>(backend), static_cast<unsigned>(prepared.ok()),
        static_cast<unsigned>(ran.ok()), static_cast<unsigned>(ran.reason()),
        static_cast<unsigned>(profile.ok()),
        static_cast<unsigned>(profile.reason()),
        static_cast<unsigned long long>(profile ? profile->written : 0u),
        static_cast<unsigned long long>(profile ? profile->total : 0u),
        actual[0], SerialOracle(active_count),
        static_cast<unsigned long long>(
            profile ? profile->execution.pipeline.executed_outer_window_count
                    : 0u),
        static_cast<unsigned long long>(
            profile ? profile->execution.pipeline.executed_inner_iteration_count
                    : 0u),
        static_cast<unsigned long long>(
            profile ? profile->execution.command_submits : 0u),
        static_cast<unsigned long long>(profile ? profile->execution.dispatches
                                                : 0u),
        static_cast<unsigned long long>(
            profile ? profile->execution.pipeline.control_command_count : 0u));
    return 3;
  }

  if (backend == Backend::Vulkan) {
    for (std::size_t inner = 0u; inner < kInner; ++inner) {
      const PipelineStepStats &execution = rows[kOuter + inner].execution;
      const std::uint64_t expected_physical = inner == 0u ? kOuter : 0u;
      if (execution.sample_count != 1u ||
          execution.original_dispatches != kOuter ||
          execution.final_dispatches != expected_physical ||
          execution.workgroup_count != expected_physical ||
          execution.work_item_count != expected_physical) {
        std::fprintf(
            stderr,
            "nested Vulkan transducer row=%llu samples=%llu authored=%llu "
            "physical=%llu/%llu groups=%llu items=%llu\n",
            static_cast<unsigned long long>(inner),
            static_cast<unsigned long long>(execution.sample_count),
            static_cast<unsigned long long>(execution.original_dispatches),
            static_cast<unsigned long long>(execution.final_dispatches),
            static_cast<unsigned long long>(expected_physical),
            static_cast<unsigned long long>(execution.workgroup_count),
            static_cast<unsigned long long>(execution.work_item_count));
        return 4;
      }
    }
  }

  if (backend == Backend::Metal) {
    for (std::size_t index = 0u; index < rows.size(); ++index) {
      const PipelineStepStats &execution = rows[index].execution;
      const bool physical_owner = index == 0u;
      const std::uint64_t expected_dispatches = physical_owner ? 2u : 0u;
      const std::uint64_t expected_workgroups =
          physical_owner ? kOuter + 1u : 0u;
      const std::uint32_t seed_live =
          index == 0u ? kTile : (index == 1u ? active_count - kTile : 0u);
      const bool seed_active = index < kOuter && seed_live != 0u;
      const std::uint64_t generated = 2u * seed_live;
      const std::uint64_t capacity = seed_active ? 2u * kTile : 0u;
      const std::uint64_t indirect = seed_active ? 2u : 0u;
      if (execution.sample_count != 1u || execution.original_dispatches == 0u ||
          execution.final_dispatches != expected_dispatches ||
          execution.workgroup_count != expected_workgroups ||
          (physical_owner ? execution.work_item_count == 0u
                          : execution.work_item_count != 0u) ||
          execution.control.generated_item_count != generated ||
          execution.control.generated_capacity != capacity ||
          execution.control.indirect_dispatch_count != indirect ||
          execution.control.indirect_work_item_count != generated ||
          execution.control.overflow_ordinal != ControlStats::no_overflow) {
        std::fprintf(
            stderr,
            "nested Metal aggregate profile row=%llu samples=%llu "
            "original=%llu physical=%llu groups=%llu items=%llu "
            "generated=%llu/%llu capacity=%llu/%llu indirect=%llu/%llu "
            "work=%llu/%llu overflow=%llu\n",
            static_cast<unsigned long long>(index),
            static_cast<unsigned long long>(execution.sample_count),
            static_cast<unsigned long long>(execution.original_dispatches),
            static_cast<unsigned long long>(execution.final_dispatches),
            static_cast<unsigned long long>(execution.workgroup_count),
            static_cast<unsigned long long>(execution.work_item_count),
            static_cast<unsigned long long>(
                execution.control.generated_item_count),
            static_cast<unsigned long long>(generated),
            static_cast<unsigned long long>(
                execution.control.generated_capacity),
            static_cast<unsigned long long>(capacity),
            static_cast<unsigned long long>(
                execution.control.indirect_dispatch_count),
            static_cast<unsigned long long>(indirect),
            static_cast<unsigned long long>(
                execution.control.indirect_work_item_count),
            static_cast<unsigned long long>(generated),
            static_cast<unsigned long long>(
                execution.control.overflow_ordinal));
        return 4;
      }
    }
  }

  const graph::Fingerprint seed_fingerprint = seed.fingerprint();
  const graph::Fingerprint action_fingerprint = action.fingerprint();
  const graph::Fingerprint fold_fingerprint = fold.fingerprint();
  for (std::size_t index = 0u; index < rows.size(); ++index) {
    const PipelineStepProfile &row = rows[index];
    const bool seed_row = index < kOuter;
    const bool action_row = index >= kOuter && index < kOuter + kInner;
    const std::uint32_t expected_iteration =
        seed_row     ? static_cast<std::uint32_t>(index)
        : action_row ? static_cast<std::uint32_t>(index - kOuter)
                     : static_cast<std::uint32_t>(index - kOuter - kInner);
    const PipelineNestedPhase expected_phase =
        seed_row     ? PipelineNestedPhase::Seed
        : action_row ? PipelineNestedPhase::Action
                     : PipelineNestedPhase::Fold;
    const graph::Fingerprint expected_program = seed_row ? seed_fingerprint
                                                : action_row
                                                    ? action_fingerprint
                                                    : fold_fingerprint;
    if (row.index != 0u || row.iteration != expected_iteration ||
        row.outer_window_bound != kOuter ||
        row.inner_iteration_bound != kInner ||
        row.nested_phase != expected_phase || row.program != expected_program ||
        row.outer_window != (seed_row ? expected_iteration
                                      : PipelineStepProfile::no_coordinate) ||
        row.inner_iteration != (action_row
                                    ? expected_iteration
                                    : PipelineStepProfile::no_coordinate)) {
      std::fprintf(
          stderr,
          "nested profile row backend=%u row=%llu index=%u iteration=%u/%u "
          "bounds=%u/%u phase=%u/%u coords=%u/%u program=%016llx:%016llx/"
          "%016llx:%016llx\n",
          static_cast<unsigned>(backend),
          static_cast<unsigned long long>(index), row.index, row.iteration,
          expected_iteration, row.outer_window_bound, row.inner_iteration_bound,
          static_cast<unsigned>(row.nested_phase),
          static_cast<unsigned>(expected_phase), row.outer_window,
          row.inner_iteration, static_cast<unsigned long long>(row.program.hi),
          static_cast<unsigned long long>(row.program.lo),
          static_cast<unsigned long long>(expected_program.hi),
          static_cast<unsigned long long>(expected_program.lo));
      return 5;
    }
  }
  return 0;
}

template <class Seed, class Fold>
[[nodiscard]] int CheckRetainedReuse(rund::compute::Device &device,
                                     const Seed &seed, const Fold &fold) {
  using namespace rund::compute;
  constexpr std::array<std::uint32_t, 1u> initial{kOuterSeed};
  constexpr std::array<std::uint32_t, 1u> count_values{kMaximum};
  auto action = MemoryActionProgram(device);
  auto outer =
      device.upload<std::uint32_t>(std::span<const std::uint32_t>{initial});
  auto queue =
      device.upload<std::uint32_t>(std::span<const std::uint32_t>{kQueue});
  auto domain = device.upload<std::uint32_t>(
      std::span<const std::uint32_t>{kDomainValues});
  auto count = device.upload<std::uint32_t>(
      std::span<const std::uint32_t>{count_values});
  auto output_one = device.buffer<std::uint32_t>(1u);
  auto output_many = device.buffer<std::uint32_t>(1u);
  auto output_short = device.buffer<std::uint32_t>(1u);
  if (!action || !outer || !queue || !domain || !count || !output_one ||
      !output_many || !output_short) {
    return 1;
  }

  const auto make_builder =
      [&]<std::size_t Inner>(rund::compute::Buffer<std::uint32_t> &output) {
        const auto body = tile_repeat<Inner>(seed, *action, fold);
        auto builder = pipeline(device);
        builder.windows<kMaximum, kTile>(body, rund::compute::window(*count),
                                         read(*outer, *queue, *domain),
                                         write_final(output));
        return builder;
      };

  auto one_builder = make_builder.template operator()<1u>(*output_one);
  auto many_builder = make_builder.template operator()<64u>(*output_many);
  const auto one_plan = one_builder.plan();
  const auto many_plan = many_builder.plan();
  const std::uint64_t action_delta =
      kOuter * 63u * action->graph().memory.logical_bytes;
  const std::uint64_t phase_live = std::max({seed.graph().memory.live_bytes,
                                             action->graph().memory.live_bytes,
                                             fold.graph().memory.live_bytes});
  const auto expected_logical = [&](const PipelinePlan &plan,
                                    const std::uint64_t inner) {
    const std::uint64_t infrastructure = plan.state_bytes + plan.prepared_bytes;
    return infrastructure +
           kOuter * (seed.graph().memory.logical_bytes +
                     inner * action->graph().memory.logical_bytes +
                     fold.graph().memory.logical_bytes);
  };
  if (!one_plan || !many_plan || action->graph().memory.logical_bytes == 0u ||
      one_plan->state_bytes != many_plan->state_bytes ||
      one_plan->transient_bytes != many_plan->transient_bytes ||
      one_plan->prepared_bytes != many_plan->prepared_bytes ||
      one_plan->scratch_bytes != many_plan->scratch_bytes ||
      one_plan->scratch_count != many_plan->scratch_count ||
      one_plan->peak_bytes != many_plan->peak_bytes ||
      one_plan->live_bytes != many_plan->live_bytes ||
      one_plan->physical_bytes != many_plan->physical_bytes ||
      one_plan->logical_bytes != expected_logical(*one_plan, 1u) ||
      many_plan->logical_bytes != expected_logical(*many_plan, 64u) ||
      one_plan->logical_bytes + action_delta != many_plan->logical_bytes ||
      one_plan->live_bytes !=
          one_plan->state_bytes + one_plan->prepared_bytes + phase_live ||
      one_plan->physical_bytes != one_plan->state_bytes +
                                      one_plan->prepared_bytes +
                                      one_plan->transient_bytes ||
      one_plan->allocation_count != many_plan->allocation_count ||
      one_plan->resource_count != many_plan->resource_count ||
      one_plan->prepared_template_count != kOuter + 1u + 3u ||
      many_plan->prepared_template_count != kOuter + 64u + 3u ||
      one_plan->prepared_command_count != kOuter * (1u + 2u) ||
      many_plan->prepared_command_count != kOuter * (64u + 2u)) {
    if (one_plan && many_plan) {
      std::fprintf(
          stderr,
          "nested reuse plan state=%llu/%llu transient=%llu/%llu "
          "prepared=%llu/%llu scratch=%llu:%llu/%llu:%llu "
          "logical=%llu/%llu action=%llu delta=%llu live=%llu/%llu "
          "peak=%llu/%llu physical=%llu/%llu allocations=%llu/%llu "
          "templates=%llu/%llu commands=%llu/%llu\n",
          static_cast<unsigned long long>(one_plan->state_bytes),
          static_cast<unsigned long long>(many_plan->state_bytes),
          static_cast<unsigned long long>(one_plan->transient_bytes),
          static_cast<unsigned long long>(many_plan->transient_bytes),
          static_cast<unsigned long long>(one_plan->prepared_bytes),
          static_cast<unsigned long long>(many_plan->prepared_bytes),
          static_cast<unsigned long long>(one_plan->scratch_bytes),
          static_cast<unsigned long long>(one_plan->scratch_count),
          static_cast<unsigned long long>(many_plan->scratch_bytes),
          static_cast<unsigned long long>(many_plan->scratch_count),
          static_cast<unsigned long long>(one_plan->logical_bytes),
          static_cast<unsigned long long>(many_plan->logical_bytes),
          static_cast<unsigned long long>(action->graph().memory.logical_bytes),
          static_cast<unsigned long long>(action_delta),
          static_cast<unsigned long long>(one_plan->live_bytes),
          static_cast<unsigned long long>(many_plan->live_bytes),
          static_cast<unsigned long long>(one_plan->peak_bytes),
          static_cast<unsigned long long>(many_plan->peak_bytes),
          static_cast<unsigned long long>(one_plan->physical_bytes),
          static_cast<unsigned long long>(many_plan->physical_bytes),
          static_cast<unsigned long long>(one_plan->allocation_count),
          static_cast<unsigned long long>(many_plan->allocation_count),
          static_cast<unsigned long long>(one_plan->prepared_template_count),
          static_cast<unsigned long long>(many_plan->prepared_template_count),
          static_cast<unsigned long long>(one_plan->prepared_command_count),
          static_cast<unsigned long long>(many_plan->prepared_command_count));
    }
    return 2;
  }

  auto one = std::move(one_builder)
                 .budget(MemoryBudget{.bytes = one_plan->peak_bytes})
                 .prepare();
  auto many = std::move(many_builder)
                  .budget(MemoryBudget{.bytes = many_plan->peak_bytes})
                  .prepare();
  const graph::Fingerprint action_fingerprint = action->fingerprint();
  if (!one || !many || !action_fingerprint ||
      one->fingerprint() == many->fingerprint() ||
      ActionOwnerCount(*one, action_fingerprint) != 1u ||
      ActionOwnerCount(*many, action_fingerprint) != 2u) {
    std::fprintf(
        stderr,
        "nested reuse prepare one=%u/%u many=%u/%u fingerprints=%u/%u "
        "owners=%llu/%llu\n",
        static_cast<unsigned>(one.ok()), static_cast<unsigned>(one.reason()),
        static_cast<unsigned>(many.ok()), static_cast<unsigned>(many.reason()),
        static_cast<unsigned>(one ? static_cast<bool>(one->fingerprint())
                                  : false),
        static_cast<unsigned>(many ? static_cast<bool>(many->fingerprint())
                                   : false),
        static_cast<unsigned long long>(
            one ? ActionOwnerCount(*one, action_fingerprint) : 0u),
        static_cast<unsigned long long>(
            many ? ActionOwnerCount(*many, action_fingerprint) : 0u));
    return 3;
  }

  auto short_builder = make_builder.template operator()<64u>(*output_short);
  auto rejected = std::move(short_builder)
                      .budget(MemoryBudget{.bytes = many_plan->peak_bytes - 1u})
                      .prepare();
  if (rejected || rejected.reason() != Reason::PipelineMemoryBudget) {
    std::fprintf(stderr, "nested short budget status=%u reason=%u peak=%llu\n",
                 static_cast<unsigned>(rejected.ok()),
                 static_cast<unsigned>(rejected.reason()),
                 static_cast<unsigned long long>(many_plan->peak_bytes));
    return 4;
  }
  return 0;
}

template <class Action, class Fold>
[[nodiscard]] int CheckMaximumPlan(rund::compute::Device &device,
                                   const Action &action, const Fold &fold) {
  using namespace rund::compute;
  constexpr std::size_t tile = 1024u;
  constexpr std::size_t inner = 64u;
  constexpr std::size_t small_maximum = tile;
  constexpr std::size_t maximum = 516096u;
  constexpr std::size_t small_outer = CeilDiv(small_maximum, tile);
  constexpr std::size_t outer = CeilDiv(maximum, tile);
  static_assert(outer == 504u);
  static_assert(outer * inner > PipelineIterationCapacity);

  const auto planned = [&]<std::size_t Maximum>() -> Result<PipelinePlan> {
    auto seed = SeedProgram<Maximum, tile>(device);
    auto outer_state = device.buffer<std::uint32_t>(1u);
    auto queue = device.buffer<std::uint32_t>(Maximum);
    auto domain = device.buffer<std::uint32_t>(kDomain);
    auto count = device.buffer<std::uint32_t>(1u);
    auto output = device.buffer<std::uint32_t>(1u);
    if (!seed || !outer_state || !queue || !domain || !count || !output) {
      return Result<PipelinePlan>::fail(Reason::PipelineInvalid);
    }
    const auto body = tile_repeat<inner>(*seed, action, fold);
    auto builder = pipeline(device);
    builder.windows<Maximum, tile>(body, rund::compute::window(*count),
                                   read(*outer_state, *queue, *domain),
                                   write_final(*output));
    return builder.plan();
  };

  const auto small = planned.template operator()<small_maximum>();
  const auto large = planned.template operator()<maximum>();
  constexpr std::uint64_t small_schedule_bytes =
      small_outer * sizeof(std::uint32_t);
  constexpr std::uint64_t large_schedule_bytes = outer * sizeof(std::uint32_t);
  constexpr std::uint64_t state_bytes = (outer + 5u) * sizeof(std::uint32_t);
  if (!small || !large || small->state_bytes < small_schedule_bytes ||
      large->state_bytes < large_schedule_bytes ||
      small->state_bytes - small_schedule_bytes !=
          large->state_bytes - large_schedule_bytes ||
      small->transient_bytes != large->transient_bytes ||
      small->prepared_bytes != large->prepared_bytes ||
      small->scratch_bytes != large->scratch_bytes ||
      small->scratch_count != large->scratch_count ||
      small->allocation_count != large->allocation_count ||
      small->node_count != large->node_count ||
      small->resource_count != large->resource_count ||
      large->state_bytes != state_bytes || large->outer_window_count != outer ||
      large->tile_capacity != tile || large->inner_iteration_count != inner ||
      large->prepared_template_count != outer + inner + 3u ||
      large->prepared_command_count != outer * (inner + 2u) ||
      large->logical_bytes <= large->physical_bytes) {
    if (small && large) {
      std::fprintf(
          stderr,
          "nested maximum plan outer=%llu tile=%llu inner=%llu "
          "state=%llu/%llu normalized=%llu/%llu "
          "transient=%llu/%llu prepared=%llu/%llu scratch=%llu:%llu/"
          "%llu:%llu allocations=%llu/%llu nodes=%llu/%llu "
          "resources=%llu/%llu templates=%llu commands=%llu "
          "logical/physical=%llu/%llu\n",
          static_cast<unsigned long long>(large->outer_window_count),
          static_cast<unsigned long long>(large->tile_capacity),
          static_cast<unsigned long long>(large->inner_iteration_count),
          static_cast<unsigned long long>(small->state_bytes),
          static_cast<unsigned long long>(large->state_bytes),
          static_cast<unsigned long long>(small->state_bytes -
                                          small_schedule_bytes),
          static_cast<unsigned long long>(large->state_bytes -
                                          large_schedule_bytes),
          static_cast<unsigned long long>(small->transient_bytes),
          static_cast<unsigned long long>(large->transient_bytes),
          static_cast<unsigned long long>(small->prepared_bytes),
          static_cast<unsigned long long>(large->prepared_bytes),
          static_cast<unsigned long long>(small->scratch_bytes),
          static_cast<unsigned long long>(small->scratch_count),
          static_cast<unsigned long long>(large->scratch_bytes),
          static_cast<unsigned long long>(large->scratch_count),
          static_cast<unsigned long long>(small->allocation_count),
          static_cast<unsigned long long>(large->allocation_count),
          static_cast<unsigned long long>(small->node_count),
          static_cast<unsigned long long>(large->node_count),
          static_cast<unsigned long long>(small->resource_count),
          static_cast<unsigned long long>(large->resource_count),
          static_cast<unsigned long long>(large->prepared_template_count),
          static_cast<unsigned long long>(large->prepared_command_count),
          static_cast<unsigned long long>(large->logical_bytes),
          static_cast<unsigned long long>(large->physical_bytes));
    }
    return 1;
  }
  return 0;
}

[[nodiscard]] int
CheckNestedAggregateStats(rund::compute::Device &device,
                          const rund::compute::Backend backend) {
  using namespace rund::compute;
  constexpr std::size_t first_maximum = 5u;
  constexpr std::size_t first_tile = 3u;
  constexpr std::size_t first_inner = 2u;
  constexpr std::size_t first_outer = CeilDiv(first_maximum, first_tile);
  constexpr std::size_t second_maximum = 8u;
  constexpr std::size_t second_tile = 3u;
  constexpr std::size_t second_inner = 4u;
  constexpr std::size_t second_outer = CeilDiv(second_maximum, second_tile);
  constexpr std::array<std::uint32_t, 1u> first_initial{100u};
  constexpr std::array<std::uint32_t, 1u> second_initial{200u};
  constexpr std::array<std::uint32_t, 1u> first_count_values{0u};
  constexpr std::array<std::uint32_t, 1u> second_count_values{4u};
  constexpr std::uint64_t executed_outer = 2u;
  constexpr std::uint64_t skipped_outer =
      first_outer + (second_outer - executed_outer);
  constexpr std::uint64_t executed_inner = executed_outer * second_inner;
  constexpr std::uint64_t skipped_inner =
      first_outer * first_inner +
      (second_outer - executed_outer) * second_inner;
  constexpr std::uint32_t second_expected =
      second_initial[0u] + (1u + second_inner) + (2u + second_inner);

  auto seed = TerminalSeedProgram(device);
  auto action = on(device)
                    .map<std::uint32_t>("nested-window-aggregate-action", 1u,
                                        [](auto value) { return value + 1u; })
                    .compile();
  auto fold = FailureFoldProgram<false>(device);
  auto first_outer_seed = device.upload<std::uint32_t>(first_initial);
  auto second_outer_seed = device.upload<std::uint32_t>(second_initial);
  auto first_count = device.upload<std::uint32_t>(first_count_values);
  auto second_count = device.upload<std::uint32_t>(second_count_values);
  auto first_output = device.buffer<std::uint32_t>(1u);
  auto second_output = device.buffer<std::uint32_t>(1u);
  if (!seed || !action || !fold || !first_outer_seed || !second_outer_seed ||
      !first_count || !second_count || !first_output || !second_output) {
    return 1;
  }

  const auto first_body = tile_repeat<first_inner>(*seed, *action, *fold);
  const auto second_body = tile_repeat<second_inner>(*seed, *action, *fold);
  auto builder = pipeline(device);
  builder
      .windows<first_maximum, first_tile>(
          first_body, rund::compute::window(*first_count),
          read(*first_outer_seed), write_final(*first_output))
      .windows<second_maximum, second_tile>(
          second_body, rund::compute::window(*second_count),
          read(*second_outer_seed), write_final(*second_output));
  auto prepared = std::move(builder).prepare();
  std::array<std::uint32_t, 1u> first_actual{};
  std::array<std::uint32_t, 1u> second_actual{};
  const Status ran =
      prepared ? prepared->run() : Status::fail(prepared.reason());
  const Stats stats = prepared ? prepared->stats() : Stats{};
  if (!prepared || !ran || !prepared->read(*first_output, first_actual) ||
      !prepared->read(*second_output, second_actual) ||
      first_actual != first_initial || second_actual[0u] != second_expected ||
      stats.pipeline.step_count != 2u ||
      stats.pipeline.verified_step_count != 2u ||
      stats.pipeline.executed_outer_window_count != executed_outer ||
      stats.pipeline.skipped_outer_window_count != skipped_outer ||
      stats.pipeline.executed_inner_iteration_count != executed_inner ||
      stats.pipeline.skipped_inner_iteration_count != skipped_inner ||
      stats.control.iteration_count != executed_outer ||
      stats.control.skipped_iteration_count != skipped_outer ||
      stats.command_submits != (backend == Backend::Cpu ? 0u : 1u)) {
    std::fprintf(
        stderr,
        "nested aggregate backend=%u prepared=%u status=%u reason=%u "
        "outputs=%u/%u:%u/%u outer=%llu/%llu:%llu/%llu "
        "inner=%llu/%llu:%llu/%llu control=%llu/%llu submits=%llu\n",
        static_cast<unsigned>(backend), static_cast<unsigned>(prepared.ok()),
        static_cast<unsigned>(ran.ok()), static_cast<unsigned>(ran.reason()),
        first_actual[0u], first_initial[0u], second_actual[0u], second_expected,
        static_cast<unsigned long long>(
            stats.pipeline.executed_outer_window_count),
        static_cast<unsigned long long>(executed_outer),
        static_cast<unsigned long long>(
            stats.pipeline.skipped_outer_window_count),
        static_cast<unsigned long long>(skipped_outer),
        static_cast<unsigned long long>(
            stats.pipeline.executed_inner_iteration_count),
        static_cast<unsigned long long>(executed_inner),
        static_cast<unsigned long long>(
            stats.pipeline.skipped_inner_iteration_count),
        static_cast<unsigned long long>(skipped_inner),
        static_cast<unsigned long long>(stats.control.iteration_count),
        static_cast<unsigned long long>(stats.control.skipped_iteration_count),
        static_cast<unsigned long long>(stats.command_submits));
    return 2;
  }
  return 0;
}

template <class Seed, class Action, class Fold>
[[nodiscard]] int CheckAggregateSeedFailures(
    rund::compute::Device &device, const rund::compute::Backend backend,
    const Seed &seed, const Action &action, const Fold &fold) {
  using namespace rund::compute;
  const auto same_execution = [](const PipelineStepStats &left,
                                 const PipelineStepStats &right) {
    return left.sample_count == right.sample_count &&
           left.original_dispatches == right.original_dispatches &&
           left.final_dispatches == right.final_dispatches &&
           left.barrier_count == right.barrier_count &&
           left.worker_count == right.worker_count &&
           left.participating_workers == right.participating_workers &&
           left.tile_count == right.tile_count &&
           left.tile_size == right.tile_size &&
           left.vector_chunks == right.vector_chunks &&
           left.tail_chunks == right.tail_chunks &&
           left.workgroup_count == right.workgroup_count &&
           left.work_item_count == right.work_item_count &&
           rund_node_test_pipeline::SameControlStats(left.control,
                                                     right.control);
  };
  const auto run_case = [&](const bool reduce_overflow) {
    constexpr std::array<std::uint32_t, 1u> initial{kOuterSeed};
    constexpr std::array<std::uint32_t, 1u> output_values{kSentinel};
    const std::array<std::uint32_t, 1u> count_values{reduce_overflow ? 6u : 5u};
    std::array<std::uint32_t, kMaximum> queue_values{kQueue};
    std::array<std::uint32_t, kDomain> domain_values{kDomainValues};
    if (reduce_overflow) {
      domain_values[kQueue[4u]] = std::numeric_limits<std::uint32_t>::max();
      domain_values[kQueue[5u]] = 1u;
    } else {
      queue_values[4u] = static_cast<std::uint32_t>(kDomain);
    }

    auto outer = device.upload<std::uint32_t>(initial);
    auto queue = device.upload<std::uint32_t>(queue_values);
    auto domain = device.upload<std::uint32_t>(domain_values);
    auto count = device.upload<std::uint32_t>(count_values);
    auto output = device.upload<std::uint32_t>(output_values);
    auto observer =
        on(device)
            .map<std::uint32_t>("nested-aggregate-failure-observe", 1u,
                                [](auto value) { return value; })
            .compile();
    auto scratch = device.buffer<std::uint32_t>(1u);
    if (!outer || !queue || !domain || !count || !output || !observer ||
        !scratch) {
      return 1;
    }

    const auto body = tile_repeat<kInner>(seed, action, fold);
    auto builder = pipeline(device);
    builder.profile(PipelineProfile::Steps)
        .windows<kMaximum, kTile>(body, rund::compute::window(*count),
                                  read(*outer, *queue, *domain),
                                  write_final(*output));
    auto prepared = std::move(builder).prepare();
    std::array<std::uint32_t, 1u> actual{};
    const Status first_failed =
        prepared ? prepared->run() : Status::fail(prepared.reason());
    const Stats first_stats = prepared ? prepared->stats() : Stats{};
    std::array<PipelineStepProfile, kTemplates> first_rows{};
    const auto first_profile =
        prepared ? prepared->profile(first_rows)
                 : Result<PipelineProfileSnapshot>::fail(prepared.reason());
    const Reason expected_reason = reduce_overflow
                                       ? Reason::ReduceSumOverflow
                                       : Reason::GatherIndexOutOfRange;
    const std::uint64_t expected_ordinal =
        reduce_overflow ? std::numeric_limits<std::uint64_t>::max() : 0u;
    const std::uint64_t expected_generated =
        backend == Backend::Cpu ? 0u : (reduce_overflow ? 12u : 9u);
    const std::uint64_t expected_capacity =
        backend == Backend::Cpu ? 0u : (reduce_overflow ? 16u : 12u);
    const std::uint64_t expected_indirect =
        backend == Backend::Cpu ? 0u : (reduce_overflow ? 4u : 3u);
    const bool direct_shape = backend != Backend::Metal ||
                              first_stats.pipeline.control_command_count == 1u;
    if (!prepared || first_failed || first_failed.reason() != expected_reason ||
        !first_profile || first_profile->written != kTemplates ||
        first_profile->total != kTemplates || first_profile->truncated() ||
        prepared->generation() != 0u || prepared->poisoned() ||
        !Observe(*observer, *output, *scratch, actual) ||
        actual != output_values || !direct_shape ||
        first_stats.command_submits != (backend == Backend::Cpu ? 0u : 1u) ||
        first_stats.pipeline.verified_step_count != 0u ||
        first_stats.pipeline.failed_step_index != 0u ||
        first_stats.pipeline.failed_outer_window != 1u ||
        first_stats.pipeline.failed_inner_iteration !=
            PipelineStats::no_coordinate ||
        first_stats.pipeline.failed_nested_phase != PipelineNestedPhase::Seed ||
        first_stats.pipeline.executed_outer_window_count != 1u ||
        first_stats.pipeline.skipped_outer_window_count != 0u ||
        first_stats.pipeline.executed_inner_iteration_count != kInner ||
        first_stats.pipeline.skipped_inner_iteration_count != 0u ||
        first_stats.control.iteration_count != 1u ||
        first_stats.control.skipped_iteration_count != 0u ||
        first_stats.control.generated_item_count != expected_generated ||
        first_stats.control.generated_capacity != expected_capacity ||
        first_stats.control.indirect_dispatch_count != expected_indirect ||
        first_stats.control.indirect_work_item_count != expected_generated ||
        first_stats.control.overflow_ordinal != expected_ordinal ||
        first_stats.publication.discard_count != 1u ||
        first_profile->execution.pipeline.verified_step_count != 0u ||
        first_profile->execution.pipeline.failed_step_index != 0u) {
      std::fprintf(
          stderr,
          "nested aggregate seed failure backend=%u reduce=%u prepared=%u "
          "status=%u reason=%u/%u profile=%u/%u rows=%llu/%llu "
          "generation=%llu poison=%u output=%u/%u "
          "verified=%llu failed=%llu coords=%llu/%llu/%u outer=%llu/%llu "
          "inner=%llu/%llu control=%llu/%llu ordinal=%llu dispatch=%llu "
          "commands=%llu discard=%llu submits=%llu\n",
          static_cast<unsigned>(backend),
          static_cast<unsigned>(reduce_overflow),
          static_cast<unsigned>(prepared.ok()),
          static_cast<unsigned>(first_failed.ok()),
          static_cast<unsigned>(first_failed.reason()),
          static_cast<unsigned>(expected_reason),
          static_cast<unsigned>(first_profile.ok()),
          static_cast<unsigned>(first_profile.reason()),
          static_cast<unsigned long long>(first_profile ? first_profile->written
                                                        : 0u),
          static_cast<unsigned long long>(first_profile ? first_profile->total
                                                        : 0u),
          static_cast<unsigned long long>(prepared ? prepared->generation()
                                                   : 0u),
          static_cast<unsigned>(prepared ? prepared->poisoned() : false),
          actual[0u], kSentinel,
          static_cast<unsigned long long>(
              first_stats.pipeline.verified_step_count),
          static_cast<unsigned long long>(
              first_stats.pipeline.failed_step_index),
          static_cast<unsigned long long>(
              first_stats.pipeline.failed_outer_window),
          static_cast<unsigned long long>(
              first_stats.pipeline.failed_inner_iteration),
          static_cast<unsigned>(first_stats.pipeline.failed_nested_phase),
          static_cast<unsigned long long>(
              first_stats.pipeline.executed_outer_window_count),
          static_cast<unsigned long long>(
              first_stats.pipeline.skipped_outer_window_count),
          static_cast<unsigned long long>(
              first_stats.pipeline.executed_inner_iteration_count),
          static_cast<unsigned long long>(
              first_stats.pipeline.skipped_inner_iteration_count),
          static_cast<unsigned long long>(first_stats.control.iteration_count),
          static_cast<unsigned long long>(
              first_stats.control.skipped_iteration_count),
          static_cast<unsigned long long>(first_stats.control.overflow_ordinal),
          static_cast<unsigned long long>(first_stats.dispatches),
          static_cast<unsigned long long>(
              first_stats.pipeline.control_command_count),
          static_cast<unsigned long long>(
              first_stats.publication.discard_count),
          static_cast<unsigned long long>(first_stats.command_submits));
      return 2;
    }

    if (backend == Backend::Metal) {
      const PipelineStepStats &prefix = first_rows[0u].execution;
      const PipelineStepStats &failed = first_rows[1u].execution;
      const std::uint64_t failed_generated = reduce_overflow ? 4u : 1u;
      const std::uint64_t failed_capacity = reduce_overflow ? 8u : 4u;
      const std::uint64_t failed_indirect = reduce_overflow ? 2u : 1u;
      if (first_rows[0u].outer_window != 0u ||
          first_rows[1u].outer_window != 1u ||
          first_rows[0u].nested_phase != PipelineNestedPhase::Seed ||
          first_rows[1u].nested_phase != PipelineNestedPhase::Seed ||
          prefix.sample_count != 1u || prefix.final_dispatches != 2u ||
          prefix.workgroup_count != kOuter + 1u ||
          prefix.work_item_count == 0u ||
          prefix.control.generated_item_count != 2u * kTile ||
          prefix.control.generated_capacity != 2u * kTile ||
          prefix.control.indirect_dispatch_count != 2u ||
          prefix.control.indirect_work_item_count != 2u * kTile ||
          prefix.control.overflow_ordinal != ControlStats::no_overflow ||
          failed.sample_count != 1u || failed.original_dispatches == 0u ||
          failed.final_dispatches != 0u || failed.workgroup_count != 0u ||
          failed.work_item_count != 0u ||
          failed.control.generated_item_count != failed_generated ||
          failed.control.generated_capacity != failed_capacity ||
          failed.control.indirect_dispatch_count != failed_indirect ||
          failed.control.indirect_work_item_count != failed_generated ||
          failed.control.overflow_ordinal != expected_ordinal ||
          !rund_node_test_pipeline::TimingUnavailable(first_rows[0u].timing) ||
          !rund_node_test_pipeline::TimingUnavailable(first_rows[1u].timing)) {
        std::fprintf(
            stderr,
            "nested aggregate failure profile backend=%u reduce=%u "
            "prefix=%llu/%llu/%llu/%llu/%llu failed=%llu/%llu/%llu/%llu/"
            "%llu rows=%u/%u phases=%u/%u\n",
            static_cast<unsigned>(backend),
            static_cast<unsigned>(reduce_overflow),
            static_cast<unsigned long long>(prefix.sample_count),
            static_cast<unsigned long long>(prefix.final_dispatches),
            static_cast<unsigned long long>(prefix.workgroup_count),
            static_cast<unsigned long long>(
                prefix.control.generated_item_count),
            static_cast<unsigned long long>(prefix.control.overflow_ordinal),
            static_cast<unsigned long long>(failed.sample_count),
            static_cast<unsigned long long>(failed.final_dispatches),
            static_cast<unsigned long long>(
                failed.control.generated_item_count),
            static_cast<unsigned long long>(
                failed.control.indirect_dispatch_count),
            static_cast<unsigned long long>(failed.control.overflow_ordinal),
            first_rows[0u].outer_window, first_rows[1u].outer_window,
            static_cast<unsigned>(first_rows[0u].nested_phase),
            static_cast<unsigned>(first_rows[1u].nested_phase));
        return 3;
      }
      for (std::size_t index = 2u; index < first_rows.size(); ++index) {
        if (first_rows[index].execution.available() ||
            !rund_node_test_pipeline::TimingUnavailable(
                first_rows[index].timing)) {
          std::fprintf(stderr,
                       "nested aggregate failure suffix backend=%u reduce=%u "
                       "row=%llu work=%llu timing=%llu\n",
                       static_cast<unsigned>(backend),
                       static_cast<unsigned>(reduce_overflow),
                       static_cast<unsigned long long>(index),
                       static_cast<unsigned long long>(
                           first_rows[index].execution.sample_count),
                       static_cast<unsigned long long>(
                           first_rows[index].timing.sample_count));
          return 4;
        }
      }
    }

    const Status second_failed = prepared->run();
    const Stats second_stats = prepared->stats();
    std::array<PipelineStepProfile, kTemplates> second_rows{};
    const auto second_profile = prepared->profile(second_rows);
    if (second_failed || second_failed.reason() != expected_reason ||
        !second_profile || second_profile->written != kTemplates ||
        second_profile->total != kTemplates || second_profile->truncated() ||
        prepared->generation() != 0u || prepared->poisoned() ||
        second_stats.pipeline.verified_step_count != 0u ||
        second_stats.pipeline.failed_step_index != 0u ||
        second_stats.pipeline.failed_outer_window != 1u ||
        second_stats.pipeline.failed_nested_phase !=
            PipelineNestedPhase::Seed ||
        second_stats.publication.discard_count != 2u ||
        second_profile->execution.pipeline.verified_step_count != 0u ||
        second_profile->execution.pipeline.failed_step_index != 0u) {
      std::fprintf(
          stderr,
          "nested aggregate failure repeat backend=%u reduce=%u status=%u "
          "reason=%u/%u profile=%u/%u rows=%llu/%llu generation=%llu "
          "poison=%u verified=%llu failed=%llu outer=%llu phase=%u "
          "discard=%llu\n",
          static_cast<unsigned>(backend),
          static_cast<unsigned>(reduce_overflow),
          static_cast<unsigned>(second_failed.ok()),
          static_cast<unsigned>(second_failed.reason()),
          static_cast<unsigned>(expected_reason),
          static_cast<unsigned>(second_profile.ok()),
          static_cast<unsigned>(second_profile.reason()),
          static_cast<unsigned long long>(
              second_profile ? second_profile->written : 0u),
          static_cast<unsigned long long>(second_profile ? second_profile->total
                                                         : 0u),
          static_cast<unsigned long long>(prepared->generation()),
          static_cast<unsigned>(prepared->poisoned()),
          static_cast<unsigned long long>(
              second_stats.pipeline.verified_step_count),
          static_cast<unsigned long long>(
              second_stats.pipeline.failed_step_index),
          static_cast<unsigned long long>(
              second_stats.pipeline.failed_outer_window),
          static_cast<unsigned>(second_stats.pipeline.failed_nested_phase),
          static_cast<unsigned long long>(
              second_stats.publication.discard_count));
      return 5;
    }
    for (std::size_t index = 0u; index < first_rows.size(); ++index) {
      if (!same_execution(first_rows[index].execution,
                          second_rows[index].execution) ||
          (backend == Backend::Metal &&
           !rund_node_test_pipeline::TimingUnavailable(
               second_rows[index].timing))) {
        std::fprintf(
            stderr,
            "nested aggregate failure profile reset backend=%u reduce=%u "
            "row=%llu samples=%llu/%llu timing=%llu generated=%llu/%llu "
            "overflow=%llu/%llu\n",
            static_cast<unsigned>(backend),
            static_cast<unsigned>(reduce_overflow),
            static_cast<unsigned long long>(index),
            static_cast<unsigned long long>(
                first_rows[index].execution.sample_count),
            static_cast<unsigned long long>(
                second_rows[index].execution.sample_count),
            static_cast<unsigned long long>(
                second_rows[index].timing.sample_count),
            static_cast<unsigned long long>(
                first_rows[index].execution.control.generated_item_count),
            static_cast<unsigned long long>(
                second_rows[index].execution.control.generated_item_count),
            static_cast<unsigned long long>(
                first_rows[index].execution.control.overflow_ordinal),
            static_cast<unsigned long long>(
                second_rows[index].execution.control.overflow_ordinal));
        return 6;
      }
    }
    return 0;
  };

  const int gather = run_case(false);
  if (gather != 0) {
    return gather;
  }
  const int reduce = run_case(true);
  return reduce == 0 ? 0 : 10 + reduce;
}

template <class Seed, class Action, class Fold>
[[nodiscard]] int CheckComposition(rund::compute::Device &device,
                                   const rund::compute::Backend backend,
                                   const Seed &seed, const Action &action,
                                   const Fold &fold) {
  using namespace rund::compute;
  constexpr std::array<std::uint32_t, 1u> initial{kOuterSeed};
  constexpr std::array<std::uint32_t, 1u> count_values{kMaximum};
  auto increment =
      on(device)
          .map<std::uint32_t>("nested-window-downstream-repeat", 1u,
                              [](auto value) { return value + 1u; })
          .compile();
  auto outer =
      device.upload<std::uint32_t>(std::span<const std::uint32_t>{initial});
  auto queue =
      device.upload<std::uint32_t>(std::span<const std::uint32_t>{kQueue});
  auto domain = device.upload<std::uint32_t>(
      std::span<const std::uint32_t>{kDomainValues});
  auto count = device.upload<std::uint32_t>(
      std::span<const std::uint32_t>{count_values});
  auto nested_output = device.buffer<std::uint32_t>(1u);
  auto derived = device.buffer<std::uint32_t>(1u);
  if (!increment || !outer || !queue || !domain || !count || !nested_output ||
      !derived) {
    return 1;
  }

  const auto body = tile_repeat<kInner>(seed, action, fold);
  auto builder = pipeline(device);
  builder
      .windows<kMaximum, kTile>(body, rund::compute::window(*count),
                                read(*outer, *queue, *domain),
                                write_final(*nested_output))
      .template repeat<2u>(*increment, read(*nested_output),
                           write_final(*derived));
  const auto plan = builder.plan();
  const std::uint64_t logical_workspace =
      kOuter * (seed.graph().memory.logical_bytes +
                kInner * action.graph().memory.logical_bytes +
                fold.graph().memory.logical_bytes) +
      2u * increment->graph().memory.logical_bytes;
  const std::uint64_t live_workspace = std::max(
      {seed.graph().memory.live_bytes, action.graph().memory.live_bytes,
       fold.graph().memory.live_bytes, increment->graph().memory.live_bytes});
  if (!plan || plan->prepared_template_count != kTemplates + 2u ||
      plan->prepared_command_count != kCommands + 2u ||
      plan->barrier_count != kTemplates + 1u ||
      plan->logical_bytes !=
          plan->state_bytes + plan->prepared_bytes + logical_workspace ||
      plan->live_bytes !=
          plan->state_bytes + plan->prepared_bytes + live_workspace ||
      plan->physical_bytes !=
          plan->state_bytes + plan->prepared_bytes + plan->transient_bytes ||
      plan->physical_bytes != plan->peak_bytes) {
    std::fprintf(
        stderr,
        "nested composition plan backend=%u status=%u reason=%u "
        "templates=%llu commands=%llu barriers=%llu "
        "logical/live/physical=%llu/%llu/%llu workspace=%llu/%llu\n",
        static_cast<unsigned>(backend), static_cast<unsigned>(plan.ok()),
        static_cast<unsigned>(plan.reason()),
        static_cast<unsigned long long>(plan ? plan->prepared_template_count
                                             : 0u),
        static_cast<unsigned long long>(plan ? plan->prepared_command_count
                                             : 0u),
        static_cast<unsigned long long>(plan ? plan->barrier_count : 0u),
        static_cast<unsigned long long>(plan ? plan->logical_bytes : 0u),
        static_cast<unsigned long long>(plan ? plan->live_bytes : 0u),
        static_cast<unsigned long long>(plan ? plan->physical_bytes : 0u),
        static_cast<unsigned long long>(logical_workspace),
        static_cast<unsigned long long>(live_workspace));
    return 2;
  }
  auto prepared = std::move(builder)
                      .budget(MemoryBudget{.bytes = plan->peak_bytes})
                      .prepare();
  std::array<std::uint32_t, 1u> nested_actual{};
  std::array<std::uint32_t, 1u> derived_actual{};
  const Status ran =
      prepared ? prepared->run() : Status::fail(prepared.reason());
  const Stats stats = prepared ? prepared->stats() : Stats{};
  if (!prepared || !ran || !prepared->read(*nested_output, nested_actual) ||
      !prepared->read(*derived, derived_actual) ||
      nested_actual[0] != SerialOracle(kMaximum) ||
      derived_actual[0] != SerialOracle(kMaximum) + 2u ||
      stats.pipeline.step_count != 2u ||
      stats.pipeline.verified_step_count != 2u ||
      stats.pipeline.barrier_count != plan->barrier_count ||
      stats.pipeline.executed_outer_window_count != kOuter ||
      stats.pipeline.executed_inner_iteration_count != kOuter * kInner ||
      stats.command_submits != (backend == Backend::Cpu ? 0u : 1u)) {
    std::fprintf(
        stderr,
        "nested composition run backend=%u prepared=%u status=%u reason=%u "
        "nested=%u/%u derived=%u/%u steps=%llu/%llu outer=%llu inner=%llu "
        "submits=%llu\n",
        static_cast<unsigned>(backend), static_cast<unsigned>(prepared.ok()),
        static_cast<unsigned>(ran.ok()), static_cast<unsigned>(ran.reason()),
        nested_actual[0], SerialOracle(kMaximum), derived_actual[0],
        SerialOracle(kMaximum) + 2u,
        static_cast<unsigned long long>(stats.pipeline.step_count),
        static_cast<unsigned long long>(stats.pipeline.verified_step_count),
        static_cast<unsigned long long>(
            stats.pipeline.executed_outer_window_count),
        static_cast<unsigned long long>(
            stats.pipeline.executed_inner_iteration_count),
        static_cast<unsigned long long>(stats.command_submits));
    return 3;
  }
  return 0;
}

[[nodiscard]] int CheckProductPlan(rund::compute::Device &device,
                                   const rund::compute::Backend backend) {
  using namespace rund::compute;
  constexpr std::size_t maximum = 33u;
  constexpr std::size_t tile = 1u;
  constexpr std::size_t inner = 33u;
  constexpr std::size_t outer = CeilDiv(maximum, tile);
  constexpr std::size_t templates = outer + inner + 3u;
  constexpr std::size_t commands = outer * (inner + 2u);
  static_assert(outer * inner > PipelineIterationCapacity);
  static_assert(templates < PipelineIterationCapacity);

  auto seed = SeedProgram<maximum, tile>(device);
  auto action = ActionProgram(device);
  auto fold = FoldProgram(device);
  std::array<std::uint32_t, maximum> queue_values{};
  for (std::size_t index = 0u; index < queue_values.size(); ++index) {
    queue_values[index] = static_cast<std::uint32_t>(index + 1u);
  }
  constexpr std::array<std::uint32_t, 1u> initial{kOuterSeed};
  constexpr std::array<std::uint32_t, 1u> count_values{maximum};
  auto outer_buffer =
      device.upload<std::uint32_t>(std::span<const std::uint32_t>{initial});
  auto queue = device.upload<std::uint32_t>(
      std::span<const std::uint32_t>{queue_values});
  auto domain = device.upload<std::uint32_t>(
      std::span<const std::uint32_t>{kDomainValues});
  auto count = device.upload<std::uint32_t>(
      std::span<const std::uint32_t>{count_values});
  auto output = device.buffer<std::uint32_t>(1u);
  if (!seed || !action || !fold || !outer_buffer || !queue || !domain ||
      !count || !output) {
    return 1;
  }

  const auto body = tile_repeat<inner>(*seed, *action, *fold);
  auto builder = pipeline(device);
  builder.windows<maximum, tile>(body, rund::compute::window(*count),
                                 read(*outer_buffer, *queue, *domain),
                                 write_final(*output));
  const auto plan = builder.plan();
  constexpr std::uint64_t state_bytes = (outer + 5u) * sizeof(std::uint32_t);
  if (!plan || plan->outer_window_count != outer ||
      plan->tile_capacity != tile || plan->inner_iteration_count != inner ||
      plan->prepared_template_count != templates ||
      plan->prepared_command_count != commands ||
      plan->state_bytes != state_bytes || plan->resource_count != 11u ||
      plan->physical_bytes != plan->peak_bytes ||
      plan->logical_bytes <= plan->physical_bytes) {
    if (plan) {
      std::fprintf(
          stderr,
          "nested product plan outer=%llu tile=%llu inner=%llu "
          "templates=%llu commands=%llu state=%llu resources=%llu "
          "logical/physical=%llu/%llu\n",
          static_cast<unsigned long long>(plan->outer_window_count),
          static_cast<unsigned long long>(plan->tile_capacity),
          static_cast<unsigned long long>(plan->inner_iteration_count),
          static_cast<unsigned long long>(plan->prepared_template_count),
          static_cast<unsigned long long>(plan->prepared_command_count),
          static_cast<unsigned long long>(plan->state_bytes),
          static_cast<unsigned long long>(plan->resource_count),
          static_cast<unsigned long long>(plan->logical_bytes),
          static_cast<unsigned long long>(plan->physical_bytes));
    }
    return 2;
  }
  auto prepared = std::move(builder)
                      .budget(MemoryBudget{.bytes = plan->peak_bytes})
                      .prepare();
  const Status ran =
      prepared ? prepared->run() : Status::fail(prepared.reason());
  const Stats stats = prepared ? prepared->stats() : Stats{};
  std::array<std::uint32_t, 1u> actual{};
  std::uint32_t expected = initial[0];
  for (const std::uint32_t item : queue_values) {
    expected += kDomainValues[item] + static_cast<std::uint32_t>(inner);
  }
  const bool unexpected_direct = backend == Backend::Metal &&
                                 stats.dispatches == 2u &&
                                 stats.pipeline.control_command_count == 1u;
  if (!prepared || !ran || !prepared->read(*output, actual) ||
      actual[0] != expected || unexpected_direct ||
      stats.pipeline.executed_outer_window_count != outer ||
      stats.pipeline.executed_inner_iteration_count != outer * inner ||
      stats.pipeline.prepared_command_count != commands ||
      stats.command_submits != (backend == Backend::Cpu ? 0u : 1u)) {
    std::fprintf(
        stderr,
        "nested product fallback backend=%u prepared=%u status=%u/%u "
        "output=%u/%u dispatches=%llu control=%llu outer=%llu inner=%llu "
        "commands=%llu submits=%llu\n",
        static_cast<unsigned>(backend), static_cast<unsigned>(prepared.ok()),
        static_cast<unsigned>(ran.ok()), static_cast<unsigned>(ran.reason()),
        actual[0], expected, static_cast<unsigned long long>(stats.dispatches),
        static_cast<unsigned long long>(stats.pipeline.control_command_count),
        static_cast<unsigned long long>(
            stats.pipeline.executed_outer_window_count),
        static_cast<unsigned long long>(
            stats.pipeline.executed_inner_iteration_count),
        static_cast<unsigned long long>(stats.pipeline.prepared_command_count),
        static_cast<unsigned long long>(stats.command_submits));
    return 3;
  }
  return 0;
}

} // namespace

[[nodiscard]] int CheckNestedWindow(rund::compute::Device &device,
                                    const rund::compute::Backend backend) {
  auto seed = SeedProgram<kMaximum, kTile>(device);
  auto action = ActionProgram(device);
  auto fold = FoldProgram(device);
  if (!seed || !action || !fold) {
    return 1;
  }
  for (const std::uint32_t count : kCounts) {
    const int checked =
        CheckCount(device, backend, *seed, *action, *fold, count);
    if (checked != 0) {
      return 10 + checked;
    }
  }
  const int binding_identity =
      CheckTransactionalBindingIdentity(device, backend);
  if (binding_identity != 0) {
    return 20 + binding_identity;
  }
  const int terminal = CheckNestedTerminal(device, backend);
  if (terminal != 0) {
    return 30 + terminal;
  }
  const int failures = CheckNestedFailures(device, backend);
  if (failures != 0) {
    return 80 + failures;
  }
  const int profile =
      CheckNestedProfile(device, backend, *seed, *action, *fold);
  if (profile != 0) {
    return 150 + profile;
  }
  const int reuse = CheckRetainedReuse(device, *seed, *fold);
  if (reuse != 0) {
    return 160 + reuse;
  }
  const int composition =
      CheckComposition(device, backend, *seed, *action, *fold);
  if (composition != 0) {
    return 170 + composition;
  }
  const int aggregate = CheckNestedAggregateStats(device, backend);
  if (aggregate != 0) {
    return 175 + aggregate;
  }
  const int aggregate_failures =
      CheckAggregateSeedFailures(device, backend, *seed, *action, *fold);
  if (aggregate_failures != 0) {
    return 178 + aggregate_failures;
  }
  const int maximum = CheckMaximumPlan(device, *action, *fold);
  if (maximum != 0) {
    return 180 + maximum;
  }
  const int product = CheckProductPlan(device, backend);
  return product == 0 ? 0 : 190 + product;
}

} // namespace rund::node::test_contract::window
