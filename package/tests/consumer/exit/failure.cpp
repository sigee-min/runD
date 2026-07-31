#include <rund/session.hpp>

int main() {
  rund::Session session{};
  const auto opened = session.open(rund::SessionConfig{.id = 0u});
  if (opened) {
    const auto closed = session.close();
    if (!closed) {
      return closed.exit_code();
    }
    return 2;
  }
  return opened.exit_code();
}
