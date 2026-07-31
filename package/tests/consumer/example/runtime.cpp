#include <rund/session.hpp>

int main() {
  const rund::Session::Result result =
      rund::run(rund::SessionConfig{.workers = 1u}, [] {});
  return result.exit_code();
}
