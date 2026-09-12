#include "internal.hpp"
#include "worker.hpp"

#include "src/compute/virtual/graph/reduce/evidence.hpp"
#include "src/compute/virtual/graph/reduce/lease.hpp"

#include <algorithm>
#include <cstdio>
#include <type_traits>

namespace rund_node_test_virtual::product::graph_forecast_window {
namespace {
using namespace rund::compute;
using namespace detail::graph_reduce;
namespace residency = detail::residency;

// Projection/lease views cannot silently grow into another owned table.
static_assert(sizeof(StageScratch) < 128u);
static_assert(sizeof(Ticket) < 160u * 1024u);
static_assert(
    std::is_same_v<decltype(Ticket::prefix_lease), residency::EpochLease>);

int check_views() {
  auto owner = std::make_unique<Ticket>();
  Ticket &ticket = *owner;
  if (!middle_scratch(ticket).uses.empty())
    return 1;
  ticket.phase = TicketPhase::IntermediateDirty;
  const auto scratch = middle_scratch(ticket);
  if (scratch.uses.data() != ticket.stage_uses.data() ||
      scratch.requests.data() != ticket.stage_requests.data())
    return 2;
  ticket.prefix_lease.token = 1u;
  if (!middle_scratch(ticket).uses.empty() || reset_ticket(ticket))
    return 3;
  ticket.prefix_lease = {};
  ticket.submitted = ExecutionStage::Prefix;
  if (!middle_scratch(ticket).uses.empty() || reset_ticket(ticket))
    return 4;
  ticket.submitted = ExecutionStage::None;
  if (!reset_ticket(ticket) || ticket.phase != TicketPhase::Empty)
    return 5;

  // The evidence adapter must preserve input hits, fetches, bytes and
  // read-region evictions without copying a Ticket or folding output rows.
  std::array<residency::CacheBinding, 3u> bindings{};
  bindings[0].fetch = true;
  bindings[2].fetch = true;
  const std::array<residency::GraphLeasePort, 2u> ports{
      residency::GraphLeasePort{.access = residency::Access::Read,
                                .region = {.first = 3u, .count = 2u},
                                .first_binding = 0u,
                                .binding_count = 2u},
      residency::GraphLeasePort{.access = residency::Access::Write,
                                .region = {.first = 5u, .count = 1u},
                                .first_binding = 2u,
                                .binding_count = 1u},
  };
  const std::array<residency::CacheTransition, 2u> transitions{
      residency::CacheTransition{.frame = 3u,
                                 .kind = residency::TransitionKind::Unmap},
      residency::CacheTransition{.frame = 5u,
                                 .kind = residency::TransitionKind::Unmap},
  };
  detail::VirtualRunProjection run{};
  run.input_page_bytes = 128u;
  Stats stats{};
  if (!record_input_evidence(stats, run, Backend::Metal, ports, bindings,
                             transitions, 1u, 128u) ||
      stats.pipeline.residency.cache_hit_count != 1u ||
      stats.pipeline.residency.page_in_count != 1u ||
      stats.pipeline.residency.page_in_bytes != 128u ||
      stats.pipeline.residency.eviction_count != 1u ||
      record_input_evidence(stats, run, Backend::Cpu, ports, bindings,
                            transitions, 0u, 0u))
    return 6;
  auto invalid_ports = ports;
  invalid_ports[0].binding_count = bindings.size() + 1u;
  if (record_input_evidence(stats, run, Backend::Metal, invalid_ports, bindings,
                            {}, 0u, 0u) ||
      stats.pipeline.residency.cache_hit_count != 1u ||
      stats.pipeline.residency.eviction_count != 1u)
    return 7;
  return 0;
}
} // namespace

int check_scratch(const rund::compute::Device &device) {
  if (const int views = check_views(); views != 0)
    return views;
  const auto backend = device.backend();
  if (!backend)
    return 8;
  auto program = build_program(device);
  if (!program)
    return 8;
  std::array<std::shared_ptr<Backing>, Inputs> backings{};
  for (std::size_t input = 0u; input < Inputs; ++input)
    backings[input] = std::make_shared<Backing>(nullptr, input);
  auto output = std::make_shared<Backing>(nullptr, Inputs);
  auto a = virtual_buffer<std::uint64_t>(Elements, backings[0]);
  auto b = virtual_buffer<std::uint64_t>(Elements, backings[1]);
  auto c = virtual_buffer<std::uint64_t>(Elements, backings[2]);
  auto d = virtual_buffer<std::uint64_t>(Elements, backings[3]);
  auto out = virtual_buffer<std::uint64_t>(Elements, output);
  if (!a || !b || !c || !d || !out)
    return 9;
  auto pipeline =
      virtual_pipeline(*program, *a, *b, *c, *d, *out, ResidencyConfig{});
  if (!pipeline)
    return 10;
  // Reuse both ticket banks and the short tail, then repeat with invalidated
  // backing versions. CPU and native accelerators share this public check.
  for (unsigned repeat = 0u; repeat < 4u; ++repeat) {
    for (auto &input : backings)
      if (!input->invalidate())
        return 11;
    const Status ran = run_on_worker(*pipeline);
    const bool exact = std::all_of(
        output->values.begin(), output->values.end(),
        [index = std::size_t{0u}](std::uint64_t value) mutable {
          return value == expected(index++);
        });
    const auto stats = pipeline->stats();
    const auto submits = stats.command_submits;
    // CPU executes Jobs and has no native command submission. Epochs prove
    // the same five-stage, three-batch Graph execution on every backend.
    const auto expected_submits = *backend == Backend::Cpu ? 0u : Stages * 3u;
    if (!ran || !exact || submits != expected_submits ||
        stats.pipeline.residency.epoch_count != Stages * 3u ||
        stats.pipeline.residency.backing_read_bytes != Inputs * Elements * 8u ||
        stats.pipeline.residency.backing_write_bytes != Elements * 8u) {
      std::fprintf(stderr,
                   "Graph scratch repeat=%u status=%.*s exact=%u submits=%llu\n",
                   repeat, static_cast<int>(ran.error().size()),
                   ran.error().data(), static_cast<unsigned>(exact),
                   static_cast<unsigned long long>(submits));
      return 12;
    }
  }
  return 0;
}
} // namespace rund_node_test_virtual::product::graph_forecast_window
