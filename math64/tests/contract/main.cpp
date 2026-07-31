int RunMath64PublicContract();
int RunMath64SimdContract();
int RunMath64FixedContract();
int RunMath64NonlinearContract();
int RunMath64TurnContract();
int RunMath64StatContract();
int RunMath64GeometryVectorContract();
int RunMath64SoaContract();
int RunMath64QuantContract();
int RunMath64NnContract();
int RunMath64ProbContract();
int RunMath64RlContract();

int main() {
  RunMath64PublicContract();
  RunMath64SimdContract();
  RunMath64FixedContract();
  RunMath64NonlinearContract();
  RunMath64TurnContract();
  RunMath64StatContract();
  RunMath64GeometryVectorContract();
  RunMath64SoaContract();
  RunMath64QuantContract();
  RunMath64NnContract();
  RunMath64ProbContract();
  RunMath64RlContract();
  return 0;
}
