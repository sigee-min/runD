#include "window/local.hpp"
#include "window/nested/local.hpp"

#include "../target/selection.hpp"

#include <cstdio>

namespace {

using rund::node::test_contract::window::Backend;
using rund::node::test_contract::window::CheckNestedWindow;
using rund::node::test_contract::window::CheckOverflow32;
using rund::node::test_contract::window::CheckOverflow64;
using rund::node::test_contract::window::CheckParity32;
using rund::node::test_contract::window::CheckParity64;
using rund::node::test_contract::window::CheckPlan;
using rund::node::test_contract::window::CheckWindowChain;
using rund::node::test_contract::window::CheckWindowFreeze;
using rund::node::test_contract::window::CheckWindowMatrix;
using rund::node::test_contract::window::CheckWindowTerminal;
using rund::node::test_contract::window::Identity;

[[nodiscard]] int CheckBackend(const Backend backend, Identity &identity) {
  auto opened =
      rund::compute::open(rund::node::test_contract::target_for(backend, 2u));
  if (!opened) {
    return 1;
  }
  if (const int plan = CheckPlan(*opened); plan != 0) {
    return 10 + plan;
  }
  if (const int fixed32 = CheckParity32(*opened, backend, identity);
      fixed32 != 0) {
    return 20 + fixed32;
  }
  if (const int fixed64 = CheckParity64(*opened, backend, identity);
      fixed64 != 0) {
    return 40 + fixed64;
  }
  if (const int overflow32 = CheckOverflow32(*opened, backend);
      overflow32 != 0) {
    return 60 + overflow32;
  }
  if (const int overflow64 = CheckOverflow64(*opened, backend);
      overflow64 != 0) {
    return 80 + overflow64;
  }
  if (const int matrix = CheckWindowMatrix(*opened, backend, identity.matrix);
      matrix != 0) {
    return 100 + matrix;
  }
  if (const int chain = CheckWindowChain(*opened, backend); chain != 0) {
    return 140 + chain;
  }
  if (const int freeze = CheckWindowFreeze(*opened, backend); freeze != 0) {
    return 160 + freeze;
  }
  if (const int terminal =
          CheckWindowTerminal(*opened, backend, identity.terminal);
      terminal != 0) {
    return 180 + terminal;
  }
  if (const int nested = CheckNestedWindow(*opened, backend); nested != 0) {
    return 220 + nested;
  }
  return 0;
}

} // namespace

int RunComputeWindowContract() {
  Identity identity{};
  for (const Backend backend :
       rund::node::test_contract::selected_compute_backends()) {
    const int result = CheckBackend(backend, identity);
    if (result != 0) {
      std::fprintf(stderr, "window contract backend=%u result=%d\n",
                   static_cast<unsigned>(backend), result);
      return static_cast<int>(backend) * 1000 + result;
    }
  }
  return identity.body32 && identity.pipeline32 && identity.output32 != 0u &&
                 identity.body64 && identity.pipeline64 &&
                 identity.output64 != 0u && identity.matrix.body &&
                 identity.matrix.pipeline && identity.matrix.output != 0u &&
                 identity.terminal.body && identity.terminal.pipeline &&
                 identity.terminal.output != 0u
             ? 0
             : 1;
}
