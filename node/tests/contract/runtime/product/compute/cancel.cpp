#include "../support.hpp"

#include <rund/compute.hpp>
#include <rund/compute/session.hpp>
#include <rund/compute/virtual.hpp>
#include <rund/task/api.hpp>
#include <rund/task/await.hpp>

#include <array>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <optional>
#include <span>
#include <string_view>

namespace rund::node::test_contract {

namespace {

using Virtual = compute::VirtualPipeline<std::int32_t(std::int32_t)>;

rund::task::Task<void>
CancelVirtualBeforeYield(::rund::Session &session, Virtual &pipeline,
                         compute::Status &cancelled,
                         std::optional<compute::Completion> &result) {
  auto submission = session.compute(pipeline).submit();
  cancelled = submission.cancel();
  (void)co_await rund::task::yield();
  result.emplace(submission.wait());
}

} // namespace

int CheckComputeCancelEpochs();

int CheckComputeCancelRace(::rund::Session &session) {
  constexpr std::array<std::int32_t, 4> input{1, 2, 3, 4};
  auto program =
      compute::on(compute::Target::cpu(2u))
          .map<std::int32_t>("node-host-cancel-race", input.size(),
                             [](auto value) { return value * 3 + 1; })
          .compile();
  if (!program) {
    return 1;
  }
  for (std::uint32_t iteration = 0u; iteration < 256u; ++iteration) {
    auto job = program->resident(input);
    if (!job) {
      return 2;
    }
    auto task = session.compute(*job).submit();
    const compute::Status cancelled = task.cancel();
    const compute::Completion result = task.wait();
    if (cancelled &&
        (result || result.error() != std::string_view{"compute_cancelled"})) {
      return 3;
    }
    if (!cancelled &&
        cancelled.error() != std::string_view{"compute_already_completed"}) {
      return 4;
    }
  }
  const int epochs = CheckComputeCancelEpochs();
  if (epochs != 0) {
    std::fprintf(stderr, "compute cancellation epoch contract failed: %d\n",
                 epochs);
  }
  if (epochs != 0) {
    return epochs;
  }

  auto device = compute::open(compute::Target::cpu(2u));
  if (!device) {
    return 5;
  }
  auto virtual_program =
      compute::on(*device)
          .map<std::int32_t>("node-host-virtual-cancel", input.size(),
                             [](auto value) { return value * 3 + 1; })
          .compile();
  auto input_backing =
      device ? compute::resident_virtual_backing<std::int32_t>(*device,
                                                               input.size())
             : compute::Result<std::shared_ptr<compute::VirtualBacking>>::fail(
                   compute::Reason::DeviceInvalid);
  auto output_backing =
      device ? compute::resident_virtual_backing<std::int32_t>(*device,
                                                               input.size())
             : compute::Result<std::shared_ptr<compute::VirtualBacking>>::fail(
                   compute::Reason::DeviceInvalid);
  constexpr std::array<std::int32_t, 4u> sentinel{91, 92, 93, 94};
  if (!virtual_program || !input_backing || !output_backing ||
      !(*output_backing)->write(0u, std::as_bytes(std::span{sentinel}))) {
    return 6;
  }
  auto input_buffer =
      compute::virtual_buffer<std::int32_t>(input.size(), *input_backing);
  auto output_buffer =
      compute::virtual_buffer<std::int32_t>(input.size(), *output_backing);
  if (!input_buffer || !output_buffer) {
    return 7;
  }
  auto pipeline = compute::virtual_pipeline(*virtual_program, *input_buffer,
                                            *output_buffer);
  if (!pipeline) {
    return 8;
  }
  compute::Status cancelled =
      compute::Status::fail(compute::Reason::TaskInvalid);
  std::optional<compute::Completion> result{};
  bool joined = false;
  const rund::Session::Result scope = session.scope([&] {
    const rund::task::Handle task = rund::task::spawn(
        "compute-virtual-cancel",
        CancelVirtualBeforeYield(session, *pipeline, cancelled, result));
    joined = static_cast<bool>(rund::task::join(task));
  });
  std::array<std::int32_t, 4u> observed{};
  if (!scope || !joined || !cancelled || !result || *result ||
      result->reason() != compute::Reason::Cancelled ||
      result->stats().dispatches != 0u ||
      !(*output_backing)
           ->read(0u, std::as_writable_bytes(std::span{observed})) ||
      observed != sentinel) {
    return 9;
  }
  return 0;
}

} // namespace rund::node::test_contract
