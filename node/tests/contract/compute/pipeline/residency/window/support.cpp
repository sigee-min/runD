#include "local.hpp"

#include <utility>

namespace rund_node_test_pipeline_residency::window {

residency::FrameRegion region(const residency::FrameTier tier,
                              const residency::FrameRole role,
                              const std::uint32_t first) {
  return residency::FrameRegion{
      .tier = tier, .role = role, .first = first, .count = 1u};
}

execution::SealResult plan(const std::uint64_t page_count) {
  const residency::CacheKey input{
      .backing = 31u, .version = 1u, .materialization_hi = 41u};
  const residency::CacheKey output{
      .backing = 32u, .version = 1u, .materialization_hi = 42u};
  return execution::seal(execution::Request{
      .page_count = page_count,
      .frame_capacity = 1u,
      .input =
          execution::Materialization{
              .cache =
                  residency::GraphMaterialization{.resource = 1u,
                                                  .key = input,
                                                  .page_bytes = 64u,
                                                  .page_count = page_count},
              .access = residency::Access::Read,
              .logical_bytes = page_count * 64u,
              .payload_bytes = 64u,
              .next_use_base = page_count,
          },
      .output =
          execution::Materialization{
              .cache =
                  residency::GraphMaterialization{.resource = 2u,
                                                  .key = output,
                                                  .page_bytes = 64u,
                                                  .page_count = page_count},
              .access = residency::Access::Write,
              .logical_bytes = page_count * 64u,
              .payload_bytes = 64u,
          },
      .publication =
          execution::Publication{
              .backing = output.backing,
              .version = output.version,
              .extent = residency::DirtyExtent{.bytes = page_count * 64u},
          },
      .host_input = {region(residency::FrameTier::Host,
                            residency::FrameRole::Input, 0u),
                     region(residency::FrameTier::Host,
                            residency::FrameRole::Input, 1u)},
      .device_input = {region(residency::FrameTier::Device,
                              residency::FrameRole::Input, 2u),
                       region(residency::FrameTier::Device,
                              residency::FrameRole::Input, 3u)},
      .device_output = {region(residency::FrameTier::Device,
                               residency::FrameRole::Output, 4u),
                        region(residency::FrameTier::Device,
                               residency::FrameRole::Output, 5u)},
      .host_output = {region(residency::FrameTier::Host,
                             residency::FrameRole::Output, 6u),
                      region(residency::FrameTier::Host,
                             residency::FrameRole::Output, 7u)},
  });
}

bool frames(residency::Authority &authority) {
  constexpr std::array<std::pair<residency::FrameTier, residency::FrameRole>,
                       8u>
      owners{{
          {residency::FrameTier::Host, residency::FrameRole::Input},
          {residency::FrameTier::Host, residency::FrameRole::Input},
          {residency::FrameTier::Device, residency::FrameRole::Input},
          {residency::FrameTier::Device, residency::FrameRole::Input},
          {residency::FrameTier::Device, residency::FrameRole::Output},
          {residency::FrameTier::Device, residency::FrameRole::Output},
          {residency::FrameTier::Host, residency::FrameRole::Output},
          {residency::FrameTier::Host, residency::FrameRole::Output},
      }};
  for (std::size_t index = 0u; index < owners.size(); ++index) {
    std::uint32_t first = 0u;
    if (!authority.register_frames(owners[index].first, owners[index].second,
                                   1u, first) ||
        first != index) {
      return false;
    }
  }
  return true;
}

execution::Release release(const residency::ExecutionLease &lease,
                           const std::uint64_t epoch, const Status status,
                           const execution::TerminalKind terminal,
                           const bool dispatched, const bool completed,
                           const bool may_write) {
  return execution::Release{
      .status = status,
      .terminal = terminal,
      .plan_identity = lease.plan,
      .token = lease.token,
      .generation = lease.generation,
      .epoch = epoch,
      .backend_sequence = epoch + 1u,
      .bank = static_cast<std::uint8_t>(epoch % execution::BankCapacity),
      .dispatched = dispatched,
      .completed = completed,
      .may_write = may_write,
  };
}

detail::PipelineWindowRelease pipeline(const execution::Release value) {
  // Epoch zero deliberately owns attempt generation zero. Zero is a valid
  // first attempt, not the suppressed/no-attempt sentinel; publication and
  // reseed are the canonical existence proof.
  const std::uint64_t attempt = value.epoch;
  return detail::PipelineWindowRelease{.release = value,
                                       .attempt_generation = attempt,
                                       .published_generation =
                                           attempt + (value.status ? 1u : 0u),
                                       .terminal_published = true,
                                       .reseeded = true};
}

execution::WindowEvidence chunk(const residency::ExecutionLease &lease,
                                const std::uint64_t first,
                                const std::size_t count, const Status status,
                                const execution::TerminalKind terminal) {
  execution::WindowEvidence value{
      .status = status,
      .terminal = terminal,
      .plan_identity = lease.plan,
      .token = lease.token,
      .generation = lease.generation,
      .first_epoch = first,
      .epoch_count = lease.epochs,
      .public_handoffs = 1u,
      .native_batches = count,
      .queue_calls = 1u,
      .native_inflight_peak = count == 1u ? 1u : 2u,
      .release_count = count,
      .completed_ns = first + count + 1u,
  };
  for (std::size_t slot = 0u; slot < count; ++slot) {
    const std::uint64_t epoch = first + slot;
    value.releases[slot] =
        release(lease, epoch, slot + 1u == count ? status : Status::success(),
                slot + 1u == count ? terminal : execution::TerminalKind::Known,
                true, terminal != execution::TerminalKind::UnknownMayWrite,
                status || slot + 1u != count ||
                    terminal == execution::TerminalKind::UnknownMayWrite);
  }
  return value;
}

} // namespace rund_node_test_pipeline_residency::window
