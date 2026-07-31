#include "local/model.hpp"

#include <rund/compute.hpp>
#include <rund/compute/pipeline.hpp>
#include <rund/compute/session.hpp>
#include <rund/session.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <node/runtime/compute/access.hpp>

#include "src/compute/pipeline/local.hpp"

namespace runtime_compute_pipeline {

int Claims(rund::Session &session, rund::compute::Device &device) {
  using namespace rund::compute;
  constexpr std::size_t count = 64u;
  std::vector<std::int32_t> values(count, 2);
  auto program =
      on(device)
          .map<std::int32_t>("session-pipeline-claims", count,
                             [](auto value) { return value * 3 + 1; })
          .map("session-pipeline-claims-second",
               [](auto value) { return value * 5 - 2; })
          .compile();
  auto input = device.upload<std::int32_t>(values);
  auto shared = device.buffer<std::int32_t>(count);
  auto first_output = device.buffer<std::int32_t>(count);
  auto second_output = device.buffer<std::int32_t>(count);
  if (!program || !input || !shared || !first_output || !second_output) {
    return 1;
  }
  auto first =
      pipeline(device).then(*program, read(*input), write(*shared)).prepare();
  auto same_writer =
      pipeline(device).then(*program, read(*input), write(*shared)).prepare();
  auto shared_read_one =
      pipeline(device)
          .then(*program, read(*shared), write(*first_output))
          .prepare();
  auto shared_read_two =
      pipeline(device)
          .then(*program, read(*shared), write(*second_output))
          .prepare();
  auto rollback_probe =
      pipeline(device).then(*program, read(*shared), write(*input)).prepare();
  if (!first || !same_writer || !shared_read_one || !shared_read_two ||
      !rollback_probe) {
    return 2;
  }

  if (!first->run()) {
    return 3;
  }
  const std::shared_ptr<detail::PipelineState> &active =
      detail::PipelineStateAccess::state(*same_writer);
  if (!detail::queue_pipeline(active)) {
    return 4;
  }
  auto busy_pipeline = session.compute(*same_writer).submit().wait();
  if (busy_pipeline || busy_pipeline.reason() != Reason::PipelineBusy) {
    return 5;
  }
  auto busy_buffer = session.compute(*first).submit().wait();
  if (busy_buffer || busy_buffer.reason() != Reason::BufferBusy ||
      first->stats().command_submits != 0u ||
      first->stats().pipeline.claim_conflict_count != 1u) {
    return 6;
  }
  std::vector<std::int32_t> unavailable(count);
  const auto busy_read = first->read(*shared, unavailable);
  if (busy_read || busy_read.reason() != Reason::BufferBusy) {
    return 7;
  }
  const Status released = detail::cancel_pipeline(active);
  if (released || released.reason() != Reason::Cancelled ||
      same_writer->poisoned()) {
    return 8;
  }
  if (!rollback_probe->run()) {
    return 9;
  }

  auto first_reader = session.compute(*shared_read_one).submit();
  auto second_reader = session.compute(*shared_read_two).submit();
  if (!first_reader.wait() || !second_reader.wait()) {
    return 10;
  }
  if (shared_read_one->stats().pipeline.claim_conflict_count != 0u ||
      shared_read_two->stats().pipeline.claim_conflict_count != 0u) {
    return 11;
  }
  return 0;
}

} // namespace runtime_compute_pipeline
