namespace program_compute_contract {

int RunComputeTileContract();
int RunComputeTileAsyncContract();
int RunComputeLoweringContract();

} // namespace program_compute_contract

int RunProgramComputeContract() {
  if (const int rc = program_compute_contract::RunComputeTileContract();
      rc != 0) {
    return rc;
  }
  if (const int rc = program_compute_contract::RunComputeTileAsyncContract();
      rc != 0) {
    return rc;
  }
  return program_compute_contract::RunComputeLoweringContract();
}
