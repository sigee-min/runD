#pragma once

#include "internal.hpp"

namespace rund::compute::detail::residency::physical::fetch {

inline void mint_ticket(const Identity plan, const std::uint64_t token,
                        const std::uint64_t generation,
                        const std::uint64_t owner,
                        const execution::SlidingProjection &projection,
                        const std::size_t use, const Projection &validated,
                        const Selection selected, const Commit committed,
                        execution::SlidingTicket &credential,
                        execution::FetchSource &source,
                        execution::FetchReuseSource &reuse,
                        std::uint32_t &frame, std::uint32_t &reuse_frame,
                        std::uint64_t &backing_bytes, std::uint64_t &nonce,
                        bool &backing, bool &reuses_frame) noexcept {
  credential = execution::SlidingTicket{
      .plan = plan,
      .coordinate = projection.coordinate,
      .token = token,
      .generation = generation,
      .owner = owner,
      .turn = committed.turn,
      .expected_bytes = validated.expected_bytes,
      .slot = static_cast<std::uint32_t>(selected.slot),
      .use = static_cast<std::uint32_t>(use),
      .kind = execution::SlidingTicketKind::Fetch,
  };
  source = validated.source;
  reuse =
      committed.reuses_frame ? validated.reuse : execution::FetchReuseSource{};
  frame = committed.physical_frame;
  reuse_frame = committed.reuses_frame ? committed.reuse_frame : 0u;
  backing_bytes = committed.reuses_frame ? validated.reuse.read_bytes
                                         : validated.source.bytes;
  nonce = committed.turn;
  backing = !selected.hit;
  reuses_frame = committed.reuses_frame;
}

} // namespace rund::compute::detail::residency::physical::fetch
