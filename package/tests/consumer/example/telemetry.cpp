#include <rund/replay.hpp>

#include <cstdio>
#include <string_view>

int main() {
  auto observe = [](const rund::telemetry::Event &event) {
    rund::telemetry::describe(event, [](const std::string_view text) {
      std::printf("%.*s", static_cast<int>(text.size()), text.data());
    });
    std::putchar('\n');
  };

  rund::SessionConfig config{.workers = 1u};
  config.telemetry = rund::telemetry::bind(observe);
  rund::Session session{};
  const rund::Session::Status opened = session.open(config);
  if (!opened) {
    return opened.exit_code();
  }

  const rund::replay::Live result =
      rund::replay::live(session, [](rund::replay::Context &) {});
  const rund::Session::Status closed = session.close();
  if (!result) {
    return result.exit_code();
  }
  return closed.exit_code();
}
