namespace program_compute_contract {
int RunPartitionIdentityContract();
int RunPartitionPlanContract();
int RunPartitionReferenceContract();

int RunPartitionContract() {
  if (const int rc = RunPartitionPlanContract(); rc != 0) {
    return rc;
  }
  if (const int rc = RunPartitionIdentityContract(); rc != 0) {
    return rc;
  }
  return RunPartitionReferenceContract();
}

}  // namespace program_compute_contract
