#include <rund/session.hpp>

int main() {
  const auto result = rund::run(rund::SessionConfig{.workers = 1u}, [] {});
  if (!result) {
    return result.exit_code();
  }
  return result.tasks().spawned() == 1u ? 0 : 2;
}
