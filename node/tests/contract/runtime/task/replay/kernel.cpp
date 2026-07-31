#include "test/assert.hpp"

#include <node/runtime/replay/kernel.hpp>

#include <array>
#include <cstddef>
#include <vector>

int RunRuntimeTaskReplayKernelContract() {
  const rund::node::replay::KernelRecord record =
      rund::node::replay::MakeKernelRecord(rund::node::replay::KernelDesc{
          .run_key_hash = 11u,
          .program_hash = 12u,
          .phase_hash = 13u,
          .dispatch_hash = 14u,
          .reduction_hash = 15u,
          .capacity_hash = 16u,
          .output_hash = 17u,
          .code = rund::replay::Code::Ok,
      });
  const rund::node::replay::KernelRecord repeated =
      rund::node::replay::MakeKernelRecord(rund::node::replay::KernelDesc{
          .run_key_hash = 11u,
          .program_hash = 12u,
          .phase_hash = 13u,
          .dispatch_hash = 14u,
          .reduction_hash = 15u,
          .capacity_hash = 16u,
          .output_hash = 17u,
          .code = rund::replay::Code::Ok,
      });
  TEST_ASSERT(record.ok());
  TEST_ASSERT(repeated.ok());
  TEST_ASSERT(record.code == rund::replay::Code::Ok);
  TEST_ASSERT(record.record_hash == 16795975691365769299ull);
  TEST_ASSERT(record.record_hash == repeated.record_hash);
  const std::array<rund::node::replay::KernelRecord, 2u> records{record,
                                                                 repeated};
  TEST_ASSERT(rund::node::replay::HashKernelRecords(records) ==
              1203101478108524731ull);
  TEST_ASSERT(noexcept(rund::node::replay::HashKernelRecord(record)));
  TEST_ASSERT(noexcept(rund::node::replay::HashKernelRecords(records)));

  const std::vector<std::byte> encoded =
      rund::node::replay::EncodeKernelRecord(record);
  TEST_ASSERT(!encoded.empty());

  rund::node::replay::KernelRecord decoded{};
  TEST_ASSERT(rund::node::replay::DecodeKernelRecord(encoded, decoded));
  TEST_ASSERT(decoded.record_hash == record.record_hash);
  TEST_ASSERT(rund::node::replay::CheckKernelRecord(record, decoded).ok());

  std::vector<std::byte> tampered = encoded;
  tampered.back() ^= std::byte{1u};
  rund::node::replay::KernelRecord bad{};
  TEST_ASSERT(!rund::node::replay::DecodeKernelRecord(tampered, bad));
  TEST_ASSERT(bad.code == rund::replay::Code::KernelHashInvalid);

  const rund::node::replay::KernelRecord missing =
      rund::node::replay::MakeKernelRecord(rund::node::replay::KernelDesc{
          .run_key_hash = 0u,
          .program_hash = 12u,
          .phase_hash = 13u,
          .dispatch_hash = 14u,
          .reduction_hash = 15u,
          .capacity_hash = 16u,
          .output_hash = 17u,
          .code = rund::replay::Code::Ok,
      });
  TEST_ASSERT(!missing.ok());
  TEST_ASSERT(missing.code == rund::replay::Code::KernelIdentityMissing);
  return 0;
}
