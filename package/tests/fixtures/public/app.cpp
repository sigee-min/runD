#include "api.hpp"

static_assert(__cplusplus >= 202002L);

int main() {
  const rund::Session::Result result = engine::execute();
  return result ? 0 : 1;
}
