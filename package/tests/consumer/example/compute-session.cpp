#include <rund/compute.hpp>
#include <rund/compute/session.hpp>
#include <rund/session.hpp>

#include <array>
#include <cstdint>
#include <vector>

int main() {
  constexpr std::array<std::int32_t, 4u> input{1, 2, 3, 4};
  auto program =
      rund::compute::on(rund::compute::Target::cpu(2u))
          .map<std::int32_t>("session-adjust", input.size(),
                             [](auto value) { return value * 2 + 5; })
          .compile();
  if (!program) {
    return program.exit_code();
  }

  auto job = program->resident(input);
  if (!job) {
    return job.exit_code();
  }

  int operation = 0;
  const rund::Session::Result hosted =
      rund::run(rund::SessionConfig{.workers = 2u},
                [&](rund::Session &session) {
                  const rund::compute::Completion completed =
                      session.compute(*job).submit().wait();
                  operation = completed.exit_code();
                });
  if (!hosted) {
    return hosted.exit_code();
  }
  if (operation != 0) {
    return operation;
  }

  auto output = job->read();
  if (!output) {
    return output.exit_code();
  }
  return *output == std::vector<std::int32_t>{7, 9, 11, 13} ? 0 : 2;
}
