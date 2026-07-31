#include "local/model.hpp"

#include <array>

int RunRuntimeTaskReplayPayloadStoreContract() {
  using replay_payload_store::Contract;
  return replay_payload_store::Run(std::array<Contract, 6u>{
      replay_payload_store::MemoryContract,
      replay_payload_store::ArchiveContract,
      replay_payload_store::PublishContract,
      replay_payload_store::InputContract,
      replay_payload_store::DiagnosticContract,
      replay_payload_store::MaterializationContract,
  });
}
