#include "test/assert.hpp"

#include <node/runtime/replay/accel.hpp>

#include <array>
#include <cstddef>
#include <vector>

int RunRuntimeTaskReplayAccelContract() {
  rund::node::replay::AccelDesc desc{};
  desc.graph_hash = 21u;
  desc.kernel_hash = 22u;
  desc.backend_hash = 23u;
  desc.caps_hash = 24u;
  desc.binding_hash = 25u;
  desc.buffer_shape_hash = 26u;
  desc.dispatch_hash = 27u;
  desc.output_hash = 28u;
  desc.code = rund::replay::Code::Ok;
  desc.diagnostic_runtime_ns = 100u;
  desc.diagnostic_driver_hash = 31u;
  desc.diagnostic_cache_hash = 32u;
  desc.diagnostic_submit_count = 33u;

  const rund::node::replay::AccelRecord first =
      rund::node::replay::MakeAccelRecord(desc);
  desc.diagnostic_runtime_ns = 200u;
  const rund::node::replay::AccelRecord second =
      rund::node::replay::MakeAccelRecord(desc);

  TEST_ASSERT(first.ok());
  TEST_ASSERT(second.ok());
  TEST_ASSERT(first.code == rund::replay::Code::Ok);
  TEST_ASSERT(first.semantic_hash == 14723540812307079449ull);
  TEST_ASSERT(first.diagnostic_hash == 5300219888254224623ull);
  TEST_ASSERT(second.diagnostic_hash == 16892463274255990339ull);
  TEST_ASSERT(first.record_hash == first.semantic_hash);
  TEST_ASSERT(first.semantic_hash == second.semantic_hash);
  TEST_ASSERT(first.diagnostic_hash != second.diagnostic_hash);
  const std::array<rund::node::replay::AccelRecord, 2u> records{first, second};
  TEST_ASSERT(rund::node::replay::HashAccelRecords(records) ==
              8497143065440037507ull);
  TEST_ASSERT(noexcept(rund::node::replay::HashAccelRecord(first)));
  TEST_ASSERT(noexcept(rund::node::replay::HashAccelRecords(records)));

  const std::vector<std::byte> encoded =
      rund::node::replay::EncodeAccelRecord(first);
  TEST_ASSERT(!encoded.empty());

  rund::node::replay::AccelRecord decoded{};
  TEST_ASSERT(rund::node::replay::DecodeAccelRecord(encoded, decoded));
  TEST_ASSERT(decoded.semantic_hash == first.semantic_hash);
  TEST_ASSERT(decoded.diagnostic_hash == first.diagnostic_hash);
  TEST_ASSERT(rund::node::replay::CheckAccelRecord(first, decoded).ok());

  std::vector<std::byte> tampered = encoded;
  tampered.back() ^= std::byte{1u};
  rund::node::replay::AccelRecord bad{};
  TEST_ASSERT(!rund::node::replay::DecodeAccelRecord(tampered, bad));
  TEST_ASSERT(bad.code == rund::replay::Code::AccelHashInvalid);

  desc.graph_hash = 0u;
  const rund::node::replay::AccelRecord missing =
      rund::node::replay::MakeAccelRecord(desc);
  TEST_ASSERT(!missing.ok());
  TEST_ASSERT(missing.code == rund::replay::Code::AccelIdentityMissing);
  return 0;
}
