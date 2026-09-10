#include "local.hpp"

#include <rund/compute/pipeline.hpp>
#include <rund/compute/session.hpp>
#include <rund/session.hpp>

#include <array>
#include <cstdint>
#include <utility>

namespace {

struct Value final {};
struct Doubled final {};

} // namespace

namespace rund::package_example::pipeline {

[[nodiscard]] int CheckRecordInSession(rund::compute::Device &device) {
  constexpr std::array<std::int32_t, 4u> input{2, 4, 6, 8};
  auto program =
      rund::compute::on(device)
          .map<std::int32_t>("pipeline-record", input.size(),
                             [](auto value) {
                               return rund::compute::record(
                                   rund::compute::field<Value>(value),
                                   rund::compute::field<Doubled>(value * 2));
                             })
          .compile();
  auto source = device.upload<std::int32_t>(input);
  auto values = device.buffer<std::int32_t>(input.size());
  auto doubled = device.buffer<std::int32_t>(input.size());
  if (!program) {
    return program.exit_code();
  }
  if (!source) {
    return source.exit_code();
  }
  if (!values) {
    return values.exit_code();
  }
  if (!doubled) {
    return doubled.exit_code();
  }

  auto prepared = rund::compute::pipeline(device)
                      .then(*program, rund::compute::read(*source),
                            rund::compute::write(*values, *doubled))
                      .prepare();
  if (!prepared) {
    return prepared.exit_code();
  }
  rund::compute::Pipeline pipeline = std::move(prepared).value();

  rund::Session session{};
  const auto opened = session.open(rund::SessionConfig{.workers = 2u});
  if (!opened) {
    return opened.exit_code();
  }
  auto submission = session.compute(pipeline).submit();
  const auto admitted = submission.poll();
  const auto completed = submission.wait();
  if (!admitted.submitted || admitted.reason() != rund::compute::Reason::Ok ||
      !completed) {
    (void)session.close();
    return !completed ? completed.exit_code() : 2;
  }
  const auto closed = session.close();
  if (!closed) {
    return closed.exit_code();
  }

  std::array<std::int32_t, input.size()> value_result{};
  std::array<std::int32_t, input.size()> doubled_result{};
  const auto value_read = pipeline.read(*values, value_result);
  const auto doubled_read = pipeline.read(*doubled, doubled_result);
  if (!value_read) {
    return value_read.exit_code();
  }
  if (!doubled_read) {
    return doubled_read.exit_code();
  }
  return value_result == input &&
                 doubled_result == std::array<std::int32_t, 4u>{4, 8, 12, 16}
             ? 0
             : 2;
}

} // namespace rund::package_example::pipeline
