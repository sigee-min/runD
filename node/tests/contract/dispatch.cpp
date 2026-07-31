#include "case/table.hpp"

#ifndef RUND_NODE_TEST_CASES
#error "RUND_NODE_TEST_CASES must name the node contract case fragment"
#endif

#define RUND_NODE_TEST_CASE(name, run) int run();
#include RUND_NODE_TEST_CASES
#undef RUND_NODE_TEST_CASE

namespace {

constexpr rund::node::test_contract::Case kCases[] = {
#define RUND_NODE_TEST_CASE(name, run) {name, run},
#include RUND_NODE_TEST_CASES
#undef RUND_NODE_TEST_CASE
};

} // namespace

namespace rund::node::test_contract {

std::span<const Case> case_table() noexcept { return kCases; }

} // namespace rund::node::test_contract
