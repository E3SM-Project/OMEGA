#ifndef OMEGA_TRIDIAGONAL_H
#define OMEGA_TRIDIAGONAL_H

#include "DataTypes.h"
#include "MachEnv.h"
#include "OmegaKokkos.h"

namespace OMEGA {

struct ThomasSolver;
struct PCRSolver;
struct ThomasDiffusionSolver;
struct PCRDiffusionSolver;

#ifdef OMEGA_TARGET_DEVICE
using TriDiagSolver     = PCRSolver;
using TriDiagDiffSolver = PCRDiffusionSolver;
#else
using TriDiagSolver     = ThomasSolver;
using TriDiagDiffSolver = ThomasDiffusionSolver;
#endif

using ScratchArray2DReal = Kokkos::View<Real *[VecLength], MemLayout,
                                        ScratchMemSpace, MemoryUnmanaged>;
struct TriDiagScratch {
   ScratchArray2DReal DL;
   ScratchArray2DReal D;
   ScratchArray2DReal DU;
   ScratchArray2DReal X;

   KOKKOS_FUNCTION TriDiagScratch(const TeamMember &Member, int NRow)
       : DL(Member.team_scratch(0), NRow), D(Member.team_scratch(0), NRow),
         DU(Member.team_scratch(0), NRow), X(Member.team_scratch(0), NRow) {}
};

struct ThomasSolver {

   static TeamPolicy makeTeamPolicy(int NBatch, int NRow) {
      TeamPolicy Policy((NBatch + VecLength - 1) / VecLength, 1, 1);
      Policy.set_scratch_size(
          0, Kokkos::PerTeam(4 * NRow * VecLength * sizeof(Real)));
      return Policy;
   }

   static void KOKKOS_FUNCTION solve(const TeamMember &Member,
                                     const TriDiagScratch &Scratch) {
      const int NRow = Scratch.X.extent_int(0);

      for (int K = 1; K < NRow; ++K) {
         for (int IVec = 0; IVec < VecLength; ++IVec) {
            const Real W = Scratch.DL(K, IVec) / Scratch.D(K - 1, IVec);
            Scratch.D(K, IVec) -= W * Scratch.DU(K - 1, IVec);
            Scratch.X(K, IVec) -= W * Scratch.X(K - 1, IVec);
         }
      }

      for (int IVec = 0; IVec < VecLength; ++IVec) {
         Scratch.X(NRow - 1, IVec) /= Scratch.D(NRow - 1, IVec);
      }

      for (int K = NRow - 2; K >= 0; --K) {
         for (int IVec = 0; IVec < VecLength; ++IVec) {
            Scratch.X(K, IVec) =
                (Scratch.X(K, IVec) -
                 Scratch.DU(K, IVec) * Scratch.X(K + 1, IVec)) /
                Scratch.D(K, IVec);
         }
      }
   }

   static void KOKKOS_FUNCTION solve(const TeamMember &Member,
                                     const Array2DReal &DL,
                                     const Array2DReal &D,
                                     const Array2DReal &DU,
                                     const Array2DReal &X) {
      const int NBatch = X.extent_int(0);
      const int NRow   = X.extent_int(1);

      const int IStart = Member.league_rank() * VecLength;

      TriDiagScratch Scratch(Member, NRow);

      for (int K = 0; K < NRow; ++K) {
         for (int IVec = 0; IVec < VecLength; ++IVec) {
            const int I = IStart + IVec;
            if (I < NBatch) {
               Scratch.DL(K, IVec) = DL(I, K);
               Scratch.D(K, IVec)  = D(I, K);
               Scratch.DU(K, IVec) = DU(I, K);
               Scratch.X(K, IVec)  = X(I, K);
            } else {
               Scratch.DL(K, IVec) = 0;
               Scratch.D(K, IVec)  = 1;
               Scratch.DU(K, IVec) = 0;
               Scratch.X(K, IVec)  = 0;
            }
         }
      }

      solve(Member, Scratch);

      for (int IVec = 0; IVec < VecLength; ++IVec) {
         for (int K = 0; K < NRow; ++K) {
            const int I = IStart + IVec;
            if (I < NBatch) {
               X(I, K) = Scratch.X(K, IVec);
            }
         }
      }
   }
};

struct PCRSolver {

   static TeamPolicy makeTeamPolicy(int NBatch, int NRow) {
      TeamPolicy Policy(NBatch, NRow, 1);
      Policy.set_scratch_size(
          0, Kokkos::PerTeam(4 * NRow * VecLength * sizeof(Real)));
      return Policy;
   }

   static void KOKKOS_FUNCTION solve(const TeamMember &Member,
                                     const TriDiagScratch &Scratch) {
      const int NRow = Scratch.X.extent_int(0);

      const int K = Member.team_rank();

      const int NLevels = Kokkos::ceil(Kokkos::log2(NRow));

      for (int Lev = 1; Lev < NLevels; ++Lev) {

         const int Stride     = 1 << Lev;
         const int HalfStride = 1 << (Lev - 1);

         int Kmh = K - HalfStride;
         Kmh     = Kmh < 0 ? 0 : Kmh;
         int Kph = K + HalfStride;
         Kph     = Kph >= NRow ? NRow - 1 : Kph;

         const Real alpha = -Scratch.DL(K, 0) / Scratch.D(Kmh, 0);
         const Real gamma = -Scratch.DU(K, 0) / Scratch.D(Kph, 0);

         const Real NewD = Scratch.D(K, 0) + alpha * Scratch.DU(Kmh, 0) +
                           gamma * Scratch.DL(Kph, 0);
         const Real NewX = Scratch.X(K, 0) + alpha * Scratch.X(Kmh, 0) +
                           gamma * Scratch.X(Kph, 0);
         const Real NewDL = alpha * Scratch.DL(Kmh, 0);
         const Real NewDU = gamma * Scratch.DU(Kph, 0);

         Member.team_barrier();

         Scratch.D(K, 0)  = NewD;
         Scratch.X(K, 0)  = NewX;
         Scratch.DL(K, 0) = NewDL;
         Scratch.DU(K, 0) = NewDU;

         Member.team_barrier();
      }

      const int Stride = 1 << (NLevels - 1);

      if (K + Stride < NRow || K - Stride >= 0) {
         if (K < NRow / 2) {
            const Real Det = Scratch.D(K, 0) * Scratch.D(K + Stride, 0) -
                             Scratch.DL(K + Stride, 0) * Scratch.DU(K, 0);
            const Real Xk   = Scratch.X(K, 0);
            const Real Xkps = Scratch.X(K + Stride, 0);
            Scratch.X(K, 0) =
                (Scratch.D(K + Stride, 0) * Xk - Scratch.DU(K, 0) * Xkps) / Det;
            Scratch.X(K + Stride, 0) =
                (-Scratch.DL(K + Stride, 0) * Xk + Scratch.D(K, 0) * Xkps) /
                Det;
         }
      } else {
         Scratch.X(K, 0) /= Scratch.D(K, 0);
      }
   }

   static void KOKKOS_FUNCTION solve(const TeamMember &Member,
                                     const Array2DReal &DL,
                                     const Array2DReal &D,
                                     const Array2DReal &DU,
                                     const Array2DReal &X) {
      const int NBatch = X.extent_int(0);
      const int NRow   = X.extent_int(1);

      const int I = Member.league_rank();
      const int K = Member.team_rank();

      TriDiagScratch Scratch(Member, NRow);

      if (I < NBatch) {
         Scratch.DL(K, 0) = DL(I, K);
         Scratch.D(K, 0)  = D(I, K);
         Scratch.DU(K, 0) = DU(I, K);
         Scratch.X(K, 0)  = X(I, K);
      } else {
         Scratch.DL(K, 0) = 0;
         Scratch.D(K, 0)  = 1;
         Scratch.DU(K, 0) = 0;
         Scratch.X(K, 0)  = 0;
      }

      Member.team_barrier();

      solve(Member, Scratch);

      Member.team_barrier();

      if (I < NBatch) {
         X(I, K) = Scratch.X(K, 0);
      }
   }
};

struct TriDiagDiffScratch {
   ScratchArray2DReal G;
   ScratchArray2DReal H;
   ScratchArray2DReal X;
   ScratchArray2DReal Alpha;

   KOKKOS_FUNCTION TriDiagDiffScratch(const TeamMember &Member, int NRow)
       : G(Member.team_scratch(0), NRow), H(Member.team_scratch(0), NRow),
         X(Member.team_scratch(0), NRow), Alpha(Member.team_scratch(0), NRow) {}
};

struct ThomasDiffusionSolver {

   static TeamPolicy makeTeamPolicy(int NBatch, int NRow) {
      TeamPolicy Policy((NBatch + VecLength - 1) / VecLength, 1, 1);
      Policy.set_scratch_size(
          0, Kokkos::PerTeam(4 * NRow * VecLength * sizeof(Real)));
      return Policy;
   }

   static void KOKKOS_FUNCTION solve(const TeamMember &Member,
                                     const TriDiagDiffScratch &Scratch) {
      const int NRow = Scratch.X.extent_int(0);

      for (int IVec = 0; IVec < VecLength; ++IVec) {
         Scratch.Alpha(0, IVec) = 0;
      }

      for (int K = 1; K < NRow; ++K) {
         for (int IVec = 0; IVec < VecLength; ++IVec) {
            Scratch.Alpha(K, IVec) =
                Scratch.G(K - 1, IVec) *
                (Scratch.H(K - 1, IVec) + Scratch.Alpha(K - 1, IVec)) /
                (Scratch.H(K - 1, IVec) + Scratch.Alpha(K - 1, IVec) +
                 Scratch.G(K - 1, IVec));
         }
      }

      for (int IVec = 0; IVec < VecLength; ++IVec) {
         Scratch.H(0, IVec) += Scratch.G(0, IVec);
      }

      for (int K = 1; K < NRow; ++K) {
         for (int IVec = 0; IVec < VecLength; ++IVec) {
            const Real AddH = Scratch.Alpha(K, IVec) + Scratch.G(K, IVec);

            Scratch.H(K, IVec) += AddH;
            Scratch.X(K, IVec) += Scratch.G(K - 1, IVec) /
                                  Scratch.H(K - 1, IVec) *
                                  Scratch.X(K - 1, IVec);
         }
      }

      for (int IVec = 0; IVec < VecLength; ++IVec) {
         Scratch.X(NRow - 1, IVec) /= Scratch.H(NRow - 1, IVec);
      }

      for (int K = NRow - 2; K >= 0; --K) {
         for (int IVec = 0; IVec < VecLength; ++IVec) {
            Scratch.X(K, IVec) = (Scratch.X(K, IVec) +
                                  Scratch.G(K, IVec) * Scratch.X(K + 1, IVec)) /
                                 Scratch.H(K, IVec);
         }
      }
   }

   static void KOKKOS_FUNCTION solve(const TeamMember &Member,
                                     const Array2DReal &G, const Array2DReal &H,
                                     const Array2DReal &X) {

      const int NBatch = X.extent_int(0);
      const int NRow   = X.extent_int(1);

      const int IStart = Member.league_rank() * VecLength;

      TriDiagDiffScratch Scratch(Member, NRow);

      for (int K = 0; K < NRow; ++K) {
         for (int IVec = 0; IVec < VecLength; ++IVec) {
            const int I = IStart + IVec;
            if (I < NBatch) {
               Scratch.G(K, IVec) = G(I, K);
               Scratch.H(K, IVec) = H(I, K);
               Scratch.X(K, IVec) = X(I, K);
            } else {
               Scratch.G(K, IVec) = 0;
               Scratch.H(K, IVec) = 1;
               Scratch.X(K, IVec) = 0;
            }
         }
      }

      solve(Member, Scratch);

      for (int IVec = 0; IVec < VecLength; ++IVec) {
         for (int K = 0; K < NRow; ++K) {
            const int I = IStart + IVec;
            if (I < NBatch) {
               X(I, K) = Scratch.X(K, IVec);
            }
         }
      }
   }
};

struct PCRDiffusionSolver {

   static TeamPolicy makeTeamPolicy(int NBatch, int NRow) {
      TeamPolicy Policy(NBatch, NRow, 1);
      Policy.set_scratch_size(
          0, Kokkos::PerTeam(4 * NRow * VecLength * sizeof(Real)));
      return Policy;
   }

   static void KOKKOS_FUNCTION solve(const TeamMember &Member,
                                     const TriDiagDiffScratch &Scratch) {
      const int NRow = Scratch.X.extent_int(0);

      const int K = Member.team_rank();

      const int NLevels = Kokkos::ceil(Kokkos::log2(NRow));

      for (int Lev = 1; Lev < NLevels; ++Lev) {

         const int Stride     = 1 << Lev;
         const int HalfStride = 1 << (Lev - 1);

         int Kmh         = K - HalfStride;
         const Real Gkmh = Kmh < 0 ? 0 : Scratch.G(Kmh, 0);
         Kmh             = Kmh < 0 ? 0 : Kmh;

         const int Kms   = K - Stride;
         const Real Gkms = Kms < 0 ? 0 : Scratch.G(Kms, 0);

         int Kph = K + HalfStride;
         Kph     = Kph >= NRow ? NRow - 1 : Kph;

         const Real Alpha = Gkmh / (Scratch.H(Kmh, 0) + Gkms + Gkmh);
         const Real Beta =
             Scratch.G(K, 0) /
             (Scratch.H(Kph, 0) + Scratch.G(K, 0) + Scratch.G(Kph, 0));

         const Real NewG = Scratch.G(Kph, 0) * Beta;
         const Real NewX = Scratch.X(K, 0) + Alpha * Scratch.X(Kmh, 0) +
                           Beta * Scratch.X(Kph, 0);
         const Real NewH = Scratch.H(K, 0) + Alpha * Scratch.H(Kmh, 0) +
                           Beta * Scratch.H(Kph, 0);

         Member.team_barrier();

         Scratch.H(K, 0) = NewH;
         Scratch.G(K, 0) = NewG;
         Scratch.X(K, 0) = NewX;

         Member.team_barrier();
      }

      const int Stride = 1 << (NLevels - 1);

      if (K + Stride < NRow || K - Stride >= 0) {
         if (K < NRow / 2) {
            const int Kms   = K - Stride;
            const Real Gkms = Kms < 0 ? 0 : Scratch.G(Kms, 0);

            const Real Dk   = Scratch.H(K, 0) + Gkms + Scratch.G(K, 0);
            const Real Dkps = Scratch.H(K + Stride, 0) + Scratch.G(K, 0) +
                              Scratch.G(K + Stride, 0);
            const Real DUk   = -Scratch.G(K, 0);
            const Real DLkps = -Scratch.G(K, 0);

            const Real Det = Dk * Dkps - DLkps * DUk;

            const Real Xk   = Scratch.X(K, 0);
            const Real Xkps = Scratch.X(K + Stride, 0);

            Scratch.X(K, 0)          = (Dkps * Xk - DUk * Xkps) / Det;
            Scratch.X(K + Stride, 0) = (-DLkps * Xk + Dk * Xkps) / Det;
         }

      } else {
         int Kms         = K - Stride;
         const Real Gkms = Kms < 0 ? 0 : Scratch.G(Kms, 0);
         Scratch.X(K, 0) /= (Scratch.H(K, 0) + Gkms + Scratch.G(K, 0));
      }
   }

   static void KOKKOS_FUNCTION solve(const TeamMember &Member,
                                     const Array2DReal &G, const Array2DReal &H,
                                     const Array2DReal &X) {
      const int NBatch = X.extent_int(0);
      const int NRow   = X.extent_int(1);

      const int I = Member.league_rank();
      const int K = Member.team_rank();

      TriDiagDiffScratch Scratch(Member, NRow);

      Scratch.G(K, 0) = G(I, K);
      Scratch.H(K, 0) = H(I, K);
      Scratch.X(K, 0) = X(I, K);

      Member.team_barrier();

      solve(Member, Scratch);

      Member.team_barrier();

      X(I, K) = Scratch.X(K, 0);
   }
};

} // namespace OMEGA
#endif