// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep_chunks/aah_sweep_chunk.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/aah_fluds.h"
#include "framework/mesh/mesh_continuum/mesh_continuum.h"
#include "caliper/cali.h"

#if defined(__AVX512F__) || defined(__AVX2__)
  #include <immintrin.h>
#endif

namespace opensn
{

#if defined(__AVX512F__)
// Solve 8 independent 6x6 systems: (Amat + sigma*M) * x = b
static inline void
AVX512_Solve(const double* __restrict Am,
             const double* __restrict Mm,
             const double* __restrict sigma_t,
             double* __restrict b)
{
  // For lane k in [0..7], element i in [0..5] at byte offset (6*k + i)*8
  const __m512i idx0 = _mm512_setr_epi64( 0*8,  6*8, 12*8, 18*8, 24*8, 30*8, 36*8, 42*8);
  const __m512i idx1 = _mm512_setr_epi64( 1*8,  7*8, 13*8, 19*8, 25*8, 31*8, 37*8, 43*8);
  const __m512i idx2 = _mm512_setr_epi64( 2*8,  8*8, 14*8, 20*8, 26*8, 32*8, 38*8, 44*8);
  const __m512i idx3 = _mm512_setr_epi64( 3*8,  9*8, 15*8, 21*8, 27*8, 33*8, 39*8, 45*8);
  const __m512i idx4 = _mm512_setr_epi64( 4*8, 10*8, 16*8, 22*8, 28*8, 34*8, 40*8, 46*8);
  const __m512i idx5 = _mm512_setr_epi64( 5*8, 11*8, 17*8, 23*8, 29*8, 35*8, 41*8, 47*8);

  char* bbytes = reinterpret_cast<char*>(b);
  __m512d b0 = _mm512_i64gather_pd(idx0, bbytes, 1);
  __m512d b1 = _mm512_i64gather_pd(idx1, bbytes, 1);
  __m512d b2 = _mm512_i64gather_pd(idx2, bbytes, 1);
  __m512d b3 = _mm512_i64gather_pd(idx3, bbytes, 1);
  __m512d b4 = _mm512_i64gather_pd(idx4, bbytes, 1);
  __m512d b5 = _mm512_i64gather_pd(idx5, bbytes, 1);

  const __m512d sv = _mm512_loadu_pd(sigma_t);

  auto splat = [](double x) { return _mm512_set1_pd(x); };
  auto madd  = [&sv](const __m512d& a, const __m512d& m) { return _mm512_fmadd_pd(sv, m, a); };

  __m512d A00 = madd(splat(Am[ 0]), splat(Mm[ 0]));
  __m512d A01 = madd(splat(Am[ 1]), splat(Mm[ 1]));
  __m512d A02 = madd(splat(Am[ 2]), splat(Mm[ 2]));
  __m512d A03 = madd(splat(Am[ 3]), splat(Mm[ 3]));
  __m512d A04 = madd(splat(Am[ 4]), splat(Mm[ 4]));
  __m512d A05 = madd(splat(Am[ 5]), splat(Mm[ 5]));

  __m512d A10 = madd(splat(Am[ 6]), splat(Mm[ 6]));
  __m512d A11 = madd(splat(Am[ 7]), splat(Mm[ 7]));
  __m512d A12 = madd(splat(Am[ 8]), splat(Mm[ 8]));
  __m512d A13 = madd(splat(Am[ 9]), splat(Mm[ 9]));
  __m512d A14 = madd(splat(Am[10]), splat(Mm[10]));
  __m512d A15 = madd(splat(Am[11]), splat(Mm[11]));

  __m512d A20 = madd(splat(Am[12]), splat(Mm[12]));
  __m512d A21 = madd(splat(Am[13]), splat(Mm[13]));
  __m512d A22 = madd(splat(Am[14]), splat(Mm[14]));
  __m512d A23 = madd(splat(Am[15]), splat(Mm[15]));
  __m512d A24 = madd(splat(Am[16]), splat(Mm[16]));
  __m512d A25 = madd(splat(Am[17]), splat(Mm[17]));

  __m512d A30 = madd(splat(Am[18]), splat(Mm[18]));
  __m512d A31 = madd(splat(Am[19]), splat(Mm[19]));
  __m512d A32 = madd(splat(Am[20]), splat(Mm[20]));
  __m512d A33 = madd(splat(Am[21]), splat(Mm[21]));
  __m512d A34 = madd(splat(Am[22]), splat(Mm[22]));
  __m512d A35 = madd(splat(Am[23]), splat(Mm[23]));

  __m512d A40 = madd(splat(Am[24]), splat(Mm[24]));
  __m512d A41 = madd(splat(Am[25]), splat(Mm[25]));
  __m512d A42 = madd(splat(Am[26]), splat(Mm[26]));
  __m512d A43 = madd(splat(Am[27]), splat(Mm[27]));
  __m512d A44 = madd(splat(Am[28]), splat(Mm[28]));
  __m512d A45 = madd(splat(Am[29]), splat(Mm[29]));

  __m512d A50 = madd(splat(Am[30]), splat(Mm[30]));
  __m512d A51 = madd(splat(Am[31]), splat(Mm[31]));
  __m512d A52 = madd(splat(Am[32]), splat(Mm[32]));
  __m512d A53 = madd(splat(Am[33]), splat(Mm[33]));
  __m512d A54 = madd(splat(Am[34]), splat(Mm[34]));
  __m512d A55 = madd(splat(Am[35]), splat(Mm[35]));

  // Forward elimination
  const __m512d invA00 = _mm512_div_pd(_mm512_set1_pd(1.0), A00);

  __m512d v10 = _mm512_mul_pd(A10, invA00);
  b1 = _mm512_fnmadd_pd(v10, b0, b1);
  A11 = _mm512_fnmadd_pd(v10, A01, A11);
  A12 = _mm512_fnmadd_pd(v10, A02, A12);
  A13 = _mm512_fnmadd_pd(v10, A03, A13);
  A14 = _mm512_fnmadd_pd(v10, A04, A14);
  A15 = _mm512_fnmadd_pd(v10, A05, A15);

  __m512d v20 = _mm512_mul_pd(A20, invA00);
  b2 = _mm512_fnmadd_pd(v20, b0, b2);
  A21 = _mm512_fnmadd_pd(v20, A01, A21);
  A22 = _mm512_fnmadd_pd(v20, A02, A22);
  A23 = _mm512_fnmadd_pd(v20, A03, A23);
  A24 = _mm512_fnmadd_pd(v20, A04, A24);
  A25 = _mm512_fnmadd_pd(v20, A05, A25);

  __m512d v30 = _mm512_mul_pd(A30, invA00);
  b3 = _mm512_fnmadd_pd(v30, b0, b3);
  A31 = _mm512_fnmadd_pd(v30, A01, A31);
  A32 = _mm512_fnmadd_pd(v30, A02, A32);
  A33 = _mm512_fnmadd_pd(v30, A03, A33);
  A34 = _mm512_fnmadd_pd(v30, A04, A34);
  A35 = _mm512_fnmadd_pd(v30, A05, A35);

  __m512d v40 = _mm512_mul_pd(A40, invA00);
  b4 = _mm512_fnmadd_pd(v40, b0, b4);
  A41 = _mm512_fnmadd_pd(v40, A01, A41);
  A42 = _mm512_fnmadd_pd(v40, A02, A42);
  A43 = _mm512_fnmadd_pd(v40, A03, A43);
  A44 = _mm512_fnmadd_pd(v40, A04, A44);
  A45 = _mm512_fnmadd_pd(v40, A05, A45);

  __m512d v50 = _mm512_mul_pd(A50, invA00);
  b5 = _mm512_fnmadd_pd(v50, b0, b5);
  A51 = _mm512_fnmadd_pd(v50, A01, A51);
  A52 = _mm512_fnmadd_pd(v50, A02, A52);
  A53 = _mm512_fnmadd_pd(v50, A03, A53);
  A54 = _mm512_fnmadd_pd(v50, A04, A54);
  A55 = _mm512_fnmadd_pd(v50, A05, A55);

  const __m512d invA11 = _mm512_div_pd(_mm512_set1_pd(1.0), A11);

  __m512d v21 = _mm512_mul_pd(A21, invA11);
  b2 = _mm512_fnmadd_pd(v21, b1, b2);
  A22 = _mm512_fnmadd_pd(v21, A12, A22);
  A23 = _mm512_fnmadd_pd(v21, A13, A23);
  A24 = _mm512_fnmadd_pd(v21, A14, A24);
  A25 = _mm512_fnmadd_pd(v21, A15, A25);

  __m512d v31 = _mm512_mul_pd(A31, invA11);
  b3 = _mm512_fnmadd_pd(v31, b1, b3);
  A32 = _mm512_fnmadd_pd(v31, A12, A32);
  A33 = _mm512_fnmadd_pd(v31, A13, A33);
  A34 = _mm512_fnmadd_pd(v31, A14, A34);
  A35 = _mm512_fnmadd_pd(v31, A15, A35);

  __m512d v41 = _mm512_mul_pd(A41, invA11);
  b4 = _mm512_fnmadd_pd(v41, b1, b4);
  A42 = _mm512_fnmadd_pd(v41, A12, A42);
  A43 = _mm512_fnmadd_pd(v41, A13, A43);
  A44 = _mm512_fnmadd_pd(v41, A14, A44);
  A45 = _mm512_fnmadd_pd(v41, A15, A45);

  __m512d v51 = _mm512_mul_pd(A51, invA11);
  b5 = _mm512_fnmadd_pd(v51, b1, b5);
  A52 = _mm512_fnmadd_pd(v51, A12, A52);
  A53 = _mm512_fnmadd_pd(v51, A13, A53);
  A54 = _mm512_fnmadd_pd(v51, A14, A54);
  A55 = _mm512_fnmadd_pd(v51, A15, A55);

  const __m512d invA22 = _mm512_div_pd(_mm512_set1_pd(1.0), A22);

  __m512d v32 = _mm512_mul_pd(A32, invA22);
  b3 = _mm512_fnmadd_pd(v32, b2, b3);
  A33 = _mm512_fnmadd_pd(v32, A23, A33);
  A34 = _mm512_fnmadd_pd(v32, A24, A34);
  A35 = _mm512_fnmadd_pd(v32, A25, A35);

  __m512d v42 = _mm512_mul_pd(A42, invA22);
  b4 = _mm512_fnmadd_pd(v42, b2, b4);
  A43 = _mm512_fnmadd_pd(v42, A23, A43);
  A44 = _mm512_fnmadd_pd(v42, A24, A44);
  A45 = _mm512_fnmadd_pd(v42, A25, A45);

  __m512d v52 = _mm512_mul_pd(A52, invA22);
  b5 = _mm512_fnmadd_pd(v52, b2, b5);
  A53 = _mm512_fnmadd_pd(v52, A23, A53);
  A54 = _mm512_fnmadd_pd(v52, A24, A54);
  A55 = _mm512_fnmadd_pd(v52, A25, A55);

  const __m512d invA33 = _mm512_div_pd(_mm512_set1_pd(1.0), A33);

  __m512d v43 = _mm512_mul_pd(A43, invA33);
  b4 = _mm512_fnmadd_pd(v43, b3, b4);
  A44 = _mm512_fnmadd_pd(v43, A34, A44);
  A45 = _mm512_fnmadd_pd(v43, A35, A45);

  __m512d v53 = _mm512_mul_pd(A53, invA33);
  b5 = _mm512_fnmadd_pd(v53, b3, b5);
  A54 = _mm512_fnmadd_pd(v53, A34, A54);
  A55 = _mm512_fnmadd_pd(v53, A35, A55);

  const __m512d invA44 = _mm512_div_pd(_mm512_set1_pd(1.0), A44);

  __m512d v54 = _mm512_mul_pd(A54, invA44);
  b5 = _mm512_fnmadd_pd(v54, b4, b5);
  A55 = _mm512_fnmadd_pd(v54, A45, A55);

  // Back substitution
  b5 = _mm512_div_pd(b5, A55);
  b4 = _mm512_mul_pd(_mm512_sub_pd(b4, _mm512_mul_pd(A45, b5)), invA44);
  b3 = _mm512_mul_pd(_mm512_sub_pd(_mm512_sub_pd(b3, _mm512_mul_pd(A34, b4)),
                                   _mm512_mul_pd(A35, b5)), invA33);
  b2 = _mm512_mul_pd(_mm512_sub_pd(_mm512_sub_pd(_mm512_sub_pd(b2, _mm512_mul_pd(A23, b3)),
                                                 _mm512_mul_pd(A24, b4)),
                                   _mm512_mul_pd(A25, b5)), invA22);
  b1 = _mm512_mul_pd(_mm512_sub_pd(_mm512_sub_pd(_mm512_sub_pd(_mm512_sub_pd(b1, _mm512_mul_pd(A12, b2)),
                                                               _mm512_mul_pd(A13, b3)),
                                                 _mm512_mul_pd(A14, b4)),
                                   _mm512_mul_pd(A15, b5)), invA11);
  b0 = _mm512_mul_pd(_mm512_sub_pd(_mm512_sub_pd(_mm512_sub_pd(_mm512_sub_pd(_mm512_sub_pd(b0, _mm512_mul_pd(A01, b1)),
                                                                            _mm512_mul_pd(A02, b2)),
                                                               _mm512_mul_pd(A03, b3)),
                                                 _mm512_mul_pd(A04, b4)),
                                   _mm512_mul_pd(A05, b5)), invA00);

  _mm512_i64scatter_pd(bbytes, idx0, b0, 1);
  _mm512_i64scatter_pd(bbytes, idx1, b1, 1);
  _mm512_i64scatter_pd(bbytes, idx2, b2, 1);
  _mm512_i64scatter_pd(bbytes, idx3, b3, 1);
  _mm512_i64scatter_pd(bbytes, idx4, b4, 1);
  _mm512_i64scatter_pd(bbytes, idx5, b5, 1);
}
#endif // __AVX512F__

#if defined(__AVX2__)
// Solve 4 independent 6x6 systems: (Amat + sigma*M) * x = b
static inline void
AVX2_Solve(const double* __restrict Am,
           const double* __restrict Mm,
           const double* __restrict sigma_t,
           double* __restrict b)
{
  // For lane k in [0..3], element i in [0..5] at byte offset (6*k + i)*8
  const __m128i idx0 = _mm_setr_epi32( 0*8,  6*8, 12*8, 18*8);
  const __m128i idx1 = _mm_setr_epi32( 1*8,  7*8, 13*8, 19*8);
  const __m128i idx2 = _mm_setr_epi32( 2*8,  8*8, 14*8, 20*8);
  const __m128i idx3 = _mm_setr_epi32( 3*8,  9*8, 15*8, 21*8);
  const __m128i idx4 = _mm_setr_epi32( 4*8, 10*8, 16*8, 22*8);
  const __m128i idx5 = _mm_setr_epi32( 5*8, 11*8, 17*8, 23*8);

  const double* bbase = b;
  __m256d b0 = _mm256_i32gather_pd(bbase, idx0, 1);
  __m256d b1 = _mm256_i32gather_pd(bbase, idx1, 1);
  __m256d b2 = _mm256_i32gather_pd(bbase, idx2, 1);
  __m256d b3 = _mm256_i32gather_pd(bbase, idx3, 1);
  __m256d b4 = _mm256_i32gather_pd(bbase, idx4, 1);
  __m256d b5 = _mm256_i32gather_pd(bbase, idx5, 1);

  const __m256d sv = _mm256_loadu_pd(sigma_t);

  auto splat = [](double x) { return _mm256_set1_pd(x); };
#if defined(__FMA__)
  auto madd   = [&sv](const __m256d& a, const __m256d& m) { return _mm256_fmadd_pd(sv, m, a); };
  auto fnmadd = [](const __m256d& x, const __m256d& y, const __m256d& z){ return _mm256_fnmadd_pd(x,y,z); };
#else
  auto madd   = [&sv](const __m256d& a, const __m256d& m){ return _mm256_add_pd(_mm256_mul_pd(sv,m), a); };
  auto fnmadd = [](const __m256d& x, const __m256d& y, const __m256d& z){ return _mm256_sub_pd(z, _mm256_mul_pd(x,y)); };
#endif

  __m256d A00 = madd(splat(Am[ 0]), splat(Mm[ 0]));
  __m256d A01 = madd(splat(Am[ 1]), splat(Mm[ 1]));
  __m256d A02 = madd(splat(Am[ 2]), splat(Mm[ 2]));
  __m256d A03 = madd(splat(Am[ 3]), splat(Mm[ 3]));
  __m256d A04 = madd(splat(Am[ 4]), splat(Mm[ 4]));
  __m256d A05 = madd(splat(Am[ 5]), splat(Mm[ 5]));

  __m256d A10 = madd(splat(Am[ 6]), splat(Mm[ 6]));
  __m256d A11 = madd(splat(Am[ 7]), splat(Mm[ 7]));
  __m256d A12 = madd(splat(Am[ 8]), splat(Mm[ 8]));
  __m256d A13 = madd(splat(Am[ 9]), splat(Mm[ 9]));
  __m256d A14 = madd(splat(Am[10]), splat(Mm[10]));
  __m256d A15 = madd(splat(Am[11]), splat(Mm[11]));

  __m256d A20 = madd(splat(Am[12]), splat(Mm[12]));
  __m256d A21 = madd(splat(Am[13]), splat(Mm[13]));
  __m256d A22 = madd(splat(Am[14]), splat(Mm[14]));
  __m256d A23 = madd(splat(Am[15]), splat(Mm[15]));
  __m256d A24 = madd(splat(Am[16]), splat(Mm[16]));
  __m256d A25 = madd(splat(Am[17]), splat(Mm[17]));

  __m256d A30 = madd(splat(Am[18]), splat(Mm[18]));
  __m256d A31 = madd(splat(Am[19]), splat(Mm[19]));
  __m256d A32 = madd(splat(Am[20]), splat(Mm[20]));
  __m256d A33 = madd(splat(Am[21]), splat(Mm[21]));
  __m256d A34 = madd(splat(Am[22]), splat(Mm[22]));
  __m256d A35 = madd(splat(Am[23]), splat(Mm[23]));

  __m256d A40 = madd(splat(Am[24]), splat(Mm[24]));
  __m256d A41 = madd(splat(Am[25]), splat(Mm[25]));
  __m256d A42 = madd(splat(Am[26]), splat(Mm[26]));
  __m256d A43 = madd(splat(Am[27]), splat(Mm[27]));
  __m256d A44 = madd(splat(Am[28]), splat(Mm[28]));
  __m256d A45 = madd(splat(Am[29]), splat(Mm[29]));

  __m256d A50 = madd(splat(Am[30]), splat(Mm[30]));
  __m256d A51 = madd(splat(Am[31]), splat(Mm[31]));
  __m256d A52 = madd(splat(Am[32]), splat(Mm[32]));
  __m256d A53 = madd(splat(Am[33]), splat(Mm[33]));
  __m256d A54 = madd(splat(Am[34]), splat(Mm[34]));
  __m256d A55 = madd(splat(Am[35]), splat(Mm[35]));

  // Forward elimination
  const __m256d invA00 = _mm256_div_pd(_mm256_set1_pd(1.0), A00);

  __m256d v10 = _mm256_mul_pd(A10, invA00);
  b1 = fnmadd(v10, b0, b1);
  A11 = fnmadd(v10, A01, A11);
  A12 = fnmadd(v10, A02, A12);
  A13 = fnmadd(v10, A03, A13);
  A14 = fnmadd(v10, A04, A14);
  A15 = fnmadd(v10, A05, A15);

  __m256d v20 = _mm256_mul_pd(A20, invA00);
  b2 = fnmadd(v20, b0, b2);
  A21 = fnmadd(v20, A01, A21);
  A22 = fnmadd(v20, A02, A22);
  A23 = fnmadd(v20, A03, A23);
  A24 = fnmadd(v20, A04, A24);
  A25 = fnmadd(v20, A05, A25);

  __m256d v30 = _mm256_mul_pd(A30, invA00);
  b3 = fnmadd(v30, b0, b3);
  A31 = fnmadd(v30, A01, A31);
  A32 = fnmadd(v30, A02, A32);
  A33 = fnmadd(v30, A03, A33);
  A34 = fnmadd(v30, A04, A34);
  A35 = fnmadd(v30, A05, A35);

  __m256d v40 = _mm256_mul_pd(A40, invA00);
  b4 = fnmadd(v40, b0, b4);
  A41 = fnmadd(v40, A01, A41);
  A42 = fnmadd(v40, A02, A42);
  A43 = fnmadd(v40, A03, A43);
  A44 = fnmadd(v40, A04, A44);
  A45 = fnmadd(v40, A05, A45);

  __m256d v50 = _mm256_mul_pd(A50, invA00);
  b5 = fnmadd(v50, b0, b5);
  A51 = fnmadd(v50, A01, A51);
  A52 = fnmadd(v50, A02, A52);
  A53 = fnmadd(v50, A03, A53);
  A54 = fnmadd(v50, A04, A54);
  A55 = fnmadd(v50, A05, A55);

  const __m256d invA11 = _mm256_div_pd(_mm256_set1_pd(1.0), A11);

  __m256d v21 = _mm256_mul_pd(A21, invA11);
  b2 = fnmadd(v21, b1, b2);
  A22 = fnmadd(v21, A12, A22);
  A23 = fnmadd(v21, A13, A23);
  A24 = fnmadd(v21, A14, A24);
  A25 = fnmadd(v21, A15, A25);

  __m256d v31 = _mm256_mul_pd(A31, invA11);
  b3 = fnmadd(v31, b1, b3);
  A32 = fnmadd(v31, A12, A32);
  A33 = fnmadd(v31, A13, A33);
  A34 = fnmadd(v31, A14, A34);
  A35 = fnmadd(v31, A15, A35);

  __m256d v41 = _mm256_mul_pd(A41, invA11);
  b4 = fnmadd(v41, b1, b4);
  A42 = fnmadd(v41, A12, A42);
  A43 = fnmadd(v41, A13, A43);
  A44 = fnmadd(v41, A14, A44);
  A45 = fnmadd(v41, A15, A45);

  __m256d v51 = _mm256_mul_pd(A51, invA11);
  b5 = fnmadd(v51, b1, b5);
  A52 = fnmadd(v51, A12, A52);
  A53 = fnmadd(v51, A13, A53);
  A54 = fnmadd(v51, A14, A54);
  A55 = fnmadd(v51, A15, A55);

  const __m256d invA22 = _mm256_div_pd(_mm256_set1_pd(1.0), A22);

  __m256d v32 = _mm256_mul_pd(A32, invA22);
  b3 = fnmadd(v32, b2, b3);
  A33 = fnmadd(v32, A23, A33);
  A34 = fnmadd(v32, A24, A34);
  A35 = fnmadd(v32, A25, A35);

  __m256d v42 = _mm256_mul_pd(A42, invA22);
  b4 = fnmadd(v42, b2, b4);
  A43 = fnmadd(v42, A23, A43);
  A44 = fnmadd(v42, A24, A44);
  A45 = fnmadd(v42, A25, A45);

  __m256d v52 = _mm256_mul_pd(A52, invA22);
  b5 = fnmadd(v52, b2, b5);
  A53 = fnmadd(v52, A23, A53);
  A54 = fnmadd(v52, A24, A54);
  A55 = fnmadd(v52, A25, A55);

  const __m256d invA33 = _mm256_div_pd(_mm256_set1_pd(1.0), A33);

  __m256d v43 = _mm256_mul_pd(A43, invA33);
  b4 = fnmadd(v43, b3, b4);
  A44 = fnmadd(v43, A34, A44);
  A45 = fnmadd(v43, A35, A45);

  __m256d v53 = _mm256_mul_pd(A53, invA33);
  b5 = fnmadd(v53, b3, b5);
  A54 = fnmadd(v53, A34, A54);
  A55 = fnmadd(v53, A35, A55);

  const __m256d invA44 = _mm256_div_pd(_mm256_set1_pd(1.0), A44);

  __m256d v54 = _mm256_mul_pd(A54, invA44);
  b5 = fnmadd(v54, b4, b5);
  A55 = fnmadd(v54, A45, A55);

  // Back substitution
  b5 = _mm256_div_pd(b5, A55);
  b4 = _mm256_mul_pd(_mm256_sub_pd(b4, _mm256_mul_pd(A45, b5)), invA44);
  b3 = _mm256_mul_pd(_mm256_sub_pd(_mm256_sub_pd(b3, _mm256_mul_pd(A34, b4)),
                                   _mm256_mul_pd(A35, b5)), invA33);
  b2 = _mm256_mul_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(b2, _mm256_mul_pd(A23, b3)),
                                                 _mm256_mul_pd(A24, b4)),
                                   _mm256_mul_pd(A25, b5)), invA22);
  b1 = _mm256_mul_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(b1, _mm256_mul_pd(A12, b2)),
                                                               _mm256_mul_pd(A13, b3)),
                                                 _mm256_mul_pd(A14, b4)),
                                   _mm256_mul_pd(A15, b5)), invA11);
  b0 = _mm256_mul_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(_mm256_sub_pd(b0, _mm256_mul_pd(A01, b1)),
                                                                            _mm256_mul_pd(A02, b2)),
                                                               _mm256_mul_pd(A03, b3)),
                                                 _mm256_mul_pd(A04, b4)),
                                   _mm256_mul_pd(A05, b5)),
                     _mm256_div_pd(_mm256_set1_pd(1.0), A00));

  // Pseudo-scatter
  alignas(32) double t0[4], t1[4], t2[4], t3[4], t4[4], t5[4];
  _mm256_store_pd(t0, b0); _mm256_store_pd(t1, b1);
  _mm256_store_pd(t2, b2); _mm256_store_pd(t3, b3);
  _mm256_store_pd(t4, b4); _mm256_store_pd(t5, b5);

  b[ 0] = t0[0]; b[ 6] = t0[1]; b[12] = t0[2]; b[18] = t0[3];
  b[ 1] = t1[0]; b[ 7] = t1[1]; b[13] = t1[2]; b[19] = t1[3];
  b[ 2] = t2[0]; b[ 8] = t2[1]; b[14] = t2[2]; b[20] = t2[3];
  b[ 3] = t3[0]; b[ 9] = t3[1]; b[15] = t3[2]; b[21] = t3[3];
  b[ 4] = t4[0]; b[10] = t4[1]; b[16] = t4[2]; b[22] = t4[3];
  b[ 5] = t5[0]; b[11] = t5[1]; b[17] = t5[2]; b[23] = t5[3];
}
#endif // __AVX2__

static inline void
Scalar_Solve(const double* __restrict Am,
             const double* __restrict Mm,
             const double  sigma,
             double* __restrict b) // length 6
{
  // Form A = Am + sigma*M (row-major 6x6)
  double A[36];
#pragma GCC unroll 6
  for (int r = 0; r < 6; ++r)
  {
#pragma GCC unroll 6
    for (int c = 0; c < 6; ++c)
      A[6*r + c] = Am[6*r + c] + sigma * Mm[6*r + c];
  }

  // Forward elimination
  const double invA00 = 1.0 / A[0];

  {
    const int r1 = 6;
    const double v = A[r1 + 0] * invA00;
    b[1] -= v * b[0];
    A[r1 + 1] -= v * A[0 + 1];
    A[r1 + 2] -= v * A[0 + 2];
    A[r1 + 3] -= v * A[0 + 3];
    A[r1 + 4] -= v * A[0 + 4];
    A[r1 + 5] -= v * A[0 + 5];
  }

  {
    const int r2 = 12;
    const double v = A[r2 + 0] * invA00;
    b[2] -= v * b[0];
    A[r2 + 1] -= v * A[0 + 1];
    A[r2 + 2] -= v * A[0 + 2];
    A[r2 + 3] -= v * A[0 + 3];
    A[r2 + 4] -= v * A[0 + 4];
    A[r2 + 5] -= v * A[0 + 5];
  }

  {
    const int r3 = 18;
    const double v = A[r3 + 0] * invA00;
    b[3] -= v * b[0];
    A[r3 + 1] -= v * A[0 + 1];
    A[r3 + 2] -= v * A[0 + 2];
    A[r3 + 3] -= v * A[0 + 3];
    A[r3 + 4] -= v * A[0 + 4];
    A[r3 + 5] -= v * A[0 + 5];
  }

  {
    const int r4 = 24;
    const double v = A[r4 + 0] * invA00;
    b[4] -= v * b[0];
    A[r4 + 1] -= v * A[0 + 1];
    A[r4 + 2] -= v * A[0 + 2];
    A[r4 + 3] -= v * A[0 + 3];
    A[r4 + 4] -= v * A[0 + 4];
    A[r4 + 5] -= v * A[0 + 5];
  }

  {
    const int r5 = 30;
    const double v = A[r5 + 0] * invA00;
    b[5] -= v * b[0];
    A[r5 + 1] -= v * A[0 + 1];
    A[r5 + 2] -= v * A[0 + 2];
    A[r5 + 3] -= v * A[0 + 3];
    A[r5 + 4] -= v * A[0 + 4];
    A[r5 + 5] -= v * A[0 + 5];
  }

  const double invA11 = 1.0 / A[6 + 1];

  {
    const int r2 = 12;
    const double v = A[r2 + 1] * invA11;
    b[2] -= v * b[1];
    A[r2 + 2] -= v * A[6 + 2];
    A[r2 + 3] -= v * A[6 + 3];
    A[r2 + 4] -= v * A[6 + 4];
    A[r2 + 5] -= v * A[6 + 5];
  }

  {
    const int r3 = 18;
    const double v = A[r3 + 1] * invA11;
    b[3] -= v * b[1];
    A[r3 + 2] -= v * A[6 + 2];
    A[r3 + 3] -= v * A[6 + 3];
    A[r3 + 4] -= v * A[6 + 4];
    A[r3 + 5] -= v * A[6 + 5];
  }

  {
    const int r4 = 24;
    const double v = A[r4 + 1] * invA11;
    b[4] -= v * b[1];
    A[r4 + 2] -= v * A[6 + 2];
    A[r4 + 3] -= v * A[6 + 3];
    A[r4 + 4] -= v * A[6 + 4];
    A[r4 + 5] -= v * A[6 + 5];
  }

  {
    const int r5 = 30;
    const double v = A[r5 + 1] * invA11;
    b[5] -= v * b[1];
    A[r5 + 2] -= v * A[6 + 2];
    A[r5 + 3] -= v * A[6 + 3];
    A[r5 + 4] -= v * A[6 + 4];
    A[r5 + 5] -= v * A[6 + 5];
  }

  const double invA22 = 1.0 / A[12 + 2];

  {
    const int r3 = 18;
    const double v = A[r3 + 2] * invA22;
    b[3] -= v * b[2];
    A[r3 + 3] -= v * A[12 + 3];
    A[r3 + 4] -= v * A[12 + 4];
    A[r3 + 5] -= v * A[12 + 5];
  }

  {
    const int r4 = 24;
    const double v = A[r4 + 2] * invA22;
    b[4] -= v * b[2];
    A[r4 + 3] -= v * A[12 + 3];
    A[r4 + 4] -= v * A[12 + 4];
    A[r4 + 5] -= v * A[12 + 5];
  }

  {
    const int r5 = 30;
    const double v = A[r5 + 2] * invA22;
    b[5] -= v * b[2];
    A[r5 + 3] -= v * A[12 + 3];
    A[r5 + 4] -= v * A[12 + 4];
    A[r5 + 5] -= v * A[12 + 5];
  }

  const double invA33 = 1.0 / A[18 + 3];

  {
    const int r4 = 24;
    const double v = A[r4 + 3] * invA33;
    b[4] -= v * b[3];
    A[r4 + 4] -= v * A[18 + 4];
    A[r4 + 5] -= v * A[18 + 5];
  }

  {
    const int r5 = 30;
    const double v = A[r5 + 3] * invA33;
    b[5] -= v * b[3];
    A[r5 + 4] -= v * A[18 + 4];
    A[r5 + 5] -= v * A[18 + 5];
  }

  const double invA44 = 1.0 / A[24 + 4]; // (4,4)

  {
    const int r5 = 30;
    const double v = A[r5 + 4] * invA44;
    b[5] -= v * b[4];
    A[r5 + 5] -= v * A[24 + 5];
  }

  // Back substitution
  b[5] = b[5] / A[30 + 5];
  b[4] = (b[4] - A[24 + 5]*b[5]) * invA44;
  b[3] = (b[3] - A[18 + 4]*b[4] - A[18 + 5]*b[5]) * invA33;
  b[2] = (b[2] - A[12 + 3]*b[3] - A[12 + 4]*b[4] - A[12 + 5]*b[5]) * invA22;
  b[1] = (b[1] - A[ 6 + 2]*b[2] - A[ 6 + 3]*b[3] - A[ 6 + 4]*b[4] - A[ 6 + 5]*b[5]) * invA11;
  b[0] = (b[0] - A[ 0 + 1]*b[1] - A[ 0 + 2]*b[2] - A[ 0 + 3]*b[3] - A[ 0 + 4]*b[4] - A[ 0 + 5]*b[5]) * invA00;
}

void
AAHSweepChunk::CPUSweep_N6(AngleSet& angle_set)
{
  CALI_CXX_MARK_FUNCTION;

  auto gs_size = groupset_.groups.size();
  auto gs_gi = groupset_.groups.front().id;

  int deploc_face_counter = -1;
  int preloc_face_counter = -1;

  auto& fluds = dynamic_cast<AAH_FLUDS&>(angle_set.GetFLUDS());
  const auto& m2d_op = groupset_.quadrature->GetMomentToDiscreteOperator();
  const auto& d2m_op = groupset_.quadrature->GetDiscreteToMomentOperator();

  DenseMatrix<double> Amat(6, 6);
  DenseMatrix<double> Atemp(6, 6);
  std::vector<double> b(groupset_.groups.size() * 6, 0.0);

  // Loop over each cell
  const auto& spds = angle_set.GetSPDS();
  const auto& spls = spds.GetLocalSubgrid();
  const size_t num_spls = spls.size();
  for (size_t spls_index = 0; spls_index < num_spls; ++spls_index)
  {
    auto cell_local_id = spls[spls_index];
    auto& cell = grid_->local_cells[cell_local_id];
    auto& cell_transport_view = cell_transport_views_[cell_local_id];
    const auto& cell_mapping = discretization_.GetCellMapping(cell);

    const size_t cell_num_faces = cell.faces.size();
    std::vector<double> face_mu_values(cell_num_faces, 0.0);
    const auto& face_orientations = spds.GetCellFaceOrientations()[cell_local_id];
    const int ni_deploc_face_counter = deploc_face_counter;
    const int ni_preloc_face_counter = preloc_face_counter;

    const auto& rho = densities_[cell.local_id];
    const auto& sigma_t = xs_.at(cell.block_id)->GetSigmaTotal();

    const auto& G      = unit_cell_matrices_[cell_local_id].intV_shapeI_gradshapeJ;
    const auto& M      = unit_cell_matrices_[cell_local_id].intV_shapeI_shapeJ;
    const auto& M_surf = unit_cell_matrices_[cell_local_id].intS_shapeI_shapeJ;

    // Loop over angles in angleset (as = angleset, ss = subset)
    const std::vector<std::uint32_t>& as_angle_indices = angle_set.GetAngleIndices();
    for (size_t as_ss_idx = 0; as_ss_idx < as_angle_indices.size(); ++as_ss_idx)
    {
      auto direction_num = as_angle_indices[as_ss_idx];
      auto omega = groupset_.quadrature->omegas[direction_num];
      auto wt = groupset_.quadrature->weights[direction_num];

      deploc_face_counter = ni_deploc_face_counter;
      preloc_face_counter = ni_preloc_face_counter;

      std::fill(b.begin(), b.end(), 0.0);

      for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j)
          Amat(i, j) = omega.Dot(G(i, j));

      for (size_t f = 0; f < cell_num_faces; ++f)
        face_mu_values[f] = omega.Dot(cell.faces[f].normal);

      int in_face_counter = -1;
      for (size_t f = 0; f < cell_num_faces; ++f)
      {
        if (face_orientations[f] != FaceOrientation::INCOMING) continue;

        auto& cell_face = cell.faces[f];
        const auto& Ms_f = M_surf[f];
        const auto num_face_nodes = cell_mapping.GetNumFaceNodes(f);
        const bool is_local_face = cell_transport_view.IsFaceLocal(f);
        const bool is_boundary_face = !cell_face.has_neighbor;
        const double mu_f = -face_mu_values[f];

        if (is_local_face) ++in_face_counter;
        else if (!is_boundary_face) ++preloc_face_counter;

        for (size_t fj = 0; fj < num_face_nodes; ++fj)
        {
          const int j = cell_mapping.MapFaceNode(f, fj);
          const double* psi = nullptr;

          if (is_local_face)
            psi = fluds.UpwindPsi(spls_index, in_face_counter, fj, 0, as_ss_idx);
          else if (!is_boundary_face)
            psi = fluds.NLUpwindPsi(preloc_face_counter, fj, 0, as_ss_idx);
          else
            psi = angle_set.PsiBoundary(cell_face.neighbor_id,
                                        direction_num,
                                        cell_local_id,
                                        f,
                                        fj,
                                        gs_gi,
                                        IsSurfaceSourceActive());

          for (size_t fi = 0; fi < num_face_nodes; ++fi)
          {
            const int i = cell_mapping.MapFaceNode(f, fi);
            const double mu_Nij = mu_f * Ms_f(i, j);

            Amat(i, j) += mu_Nij;

            if (!psi) continue;

            for (size_t gsg = 0; gsg < gs_size; ++gsg)
              b[gsg * 6 + i] += psi[gsg] * mu_Nij;
          }
        }
      }

      const double M00 = M(0,0), M01 = M(0,1), M02 = M(0,2), M03 = M(0,3), M04 = M(0,4), M05 = M(0,5);
      const double M10 = M(1,0), M11 = M(1,1), M12 = M(1,2), M13 = M(1,3), M14 = M(1,4), M15 = M(1,5);
      const double M20 = M(2,0), M21 = M(2,1), M22 = M(2,2), M23 = M(2,3), M24 = M(2,4), M25 = M(2,5);
      const double M30 = M(3,0), M31 = M(3,1), M32 = M(3,2), M33 = M(3,3), M34 = M(3,4), M35 = M(3,5);
      const double M40 = M(4,0), M41 = M(4,1), M42 = M(4,2), M43 = M(4,3), M44 = M(4,4), M45 = M(4,5);
      const double M50 = M(5,0), M51 = M(5,1), M52 = M(5,2), M53 = M(5,3), M54 = M(5,4), M55 = M(5,5);

#if defined(__AVX512F__) || defined(__AVX2__)
      const double Mm[36] = {
        M00,M01,M02,M03,M04,M05,
        M10,M11,M12,M13,M14,M15,
        M20,M21,M22,M23,M24,M25,
        M30,M31,M32,M33,M34,M35,
        M40,M41,M42,M43,M44,M45,
        M50,M51,M52,M53,M54,M55};
#endif

      const double Am00 = Amat(0,0), Am01 = Amat(0,1), Am02 = Amat(0,2), Am03 = Amat(0,3), Am04 = Amat(0,4), Am05 = Amat(0,5);
      const double Am10 = Amat(1,0), Am11 = Amat(1,1), Am12 = Amat(1,2), Am13 = Amat(1,3), Am14 = Amat(1,4), Am15 = Amat(1,5);
      const double Am20 = Amat(2,0), Am21 = Amat(2,1), Am22 = Amat(2,2), Am23 = Amat(2,3), Am24 = Amat(2,4), Am25 = Amat(2,5);
      const double Am30 = Amat(3,0), Am31 = Amat(3,1), Am32 = Amat(3,2), Am33 = Amat(3,3), Am34 = Amat(3,4), Am35 = Amat(3,5);
      const double Am40 = Amat(4,0), Am41 = Amat(4,1), Am42 = Amat(4,2), Am43 = Amat(4,3), Am44 = Amat(4,4), Am45 = Amat(4,5);
      const double Am50 = Amat(5,0), Am51 = Amat(5,1), Am52 = Amat(5,2), Am53 = Amat(5,3), Am54 = Amat(5,4), Am55 = Amat(5,5);

#if defined(__AVX512F__) || defined(__AVX2__)
      const double Am[36] = {
        Am00,Am01,Am02,Am03,Am04,Am05,
        Am10,Am11,Am12,Am13,Am14,Am15,
        Am20,Am21,Am22,Am23,Am24,Am25,
        Am30,Am31,Am32,Am33,Am34,Am35,
        Am40,Am41,Am42,Am43,Am44,Am45,
        Am50,Am51,Am52,Am53,Am54,Am55};
#endif

      const double* __restrict m2d_row = m2d_op[direction_num].data();
      const double* __restrict d2m_row = d2m_op[direction_num].data();

      // Process groups in blocks
      for (size_t g0 = 0; g0 < gs_size; g0 += group_block_size_)
      {
        const size_t g1 = std::min(g0 + group_block_size_, gs_size);

        std::vector<double> sigma_block(g1 - g0);

        for (size_t gsg = g0; gsg < g1; ++gsg)
        {
          const size_t rel = gsg - g0;
          const double sigma_tg = rho * sigma_t[gs_gi + gsg];
          sigma_block[rel] = sigma_tg;

          for (int m = 0; m < num_moments_; ++m)
          {
            const size_t dof0 = cell_transport_view.MapDOF(0, m, gs_gi);
            const size_t dof1 = cell_transport_view.MapDOF(1, m, gs_gi);
            const size_t dof2 = cell_transport_view.MapDOF(2, m, gs_gi);
            const size_t dof3 = cell_transport_view.MapDOF(3, m, gs_gi);
            const size_t dof4 = cell_transport_view.MapDOF(4, m, gs_gi);
            const size_t dof5 = cell_transport_view.MapDOF(5, m, gs_gi);

            double* __restrict bg = &b[gsg * 6];
            const double w = m2d_row[m];
            const double s0 = w * source_moments_[dof0 + gsg];
            const double s1 = w * source_moments_[dof1 + gsg];
            const double s2 = w * source_moments_[dof2 + gsg];
            const double s3 = w * source_moments_[dof3 + gsg];
            const double s4 = w * source_moments_[dof4 + gsg];
            const double s5 = w * source_moments_[dof5 + gsg];

            bg[0] += M00*s0 + M01*s1 + M02*s2 + M03*s3 + M04*s4 + M05*s5;
            bg[1] += M10*s0 + M11*s1 + M12*s2 + M13*s3 + M14*s4 + M15*s5;
            bg[2] += M20*s0 + M21*s1 + M22*s2 + M23*s3 + M24*s4 + M25*s5;
            bg[3] += M30*s0 + M31*s1 + M32*s2 + M33*s3 + M34*s4 + M35*s5;
            bg[4] += M40*s0 + M41*s1 + M42*s2 + M43*s3 + M44*s4 + M45*s5;
            bg[5] += M50*s0 + M51*s1 + M52*s2 + M53*s3 + M54*s4 + M55*s5;
          }
        }

        size_t block_len = g1 - g0;
        size_t k = 0;

#if defined(__AVX512F__)
        for (; k + 8 <= block_len; k += 8)
          AVX512_Solve(Am, Mm, &sigma_block[k], &b[(g0 + k) * 6]);
#endif

#if defined(__AVX2__)
        for (; k + 4 <= block_len; k += 4)
          AVX2_Solve(Am, Mm, &sigma_block[k], &b[(g0 + k) * 6]);
#endif

        // Scalar tail
        for (; k < block_len; ++k)
        {
          const size_t gsg = g0 + k;
          Scalar_Solve(Am, Mm, sigma_block[k], &b[gsg * 6]);
        }

        for (size_t gsg = g0; gsg < g1; ++gsg)
        {
          const double* __restrict bg = &b[gsg * 6];

          for (int m = 0; m < num_moments_; ++m)
          {
            const double w = d2m_row[m];
            const size_t dof0 = cell_transport_view.MapDOF(0, m, gs_gi);
            const size_t dof1 = cell_transport_view.MapDOF(1, m, gs_gi);
            const size_t dof2 = cell_transport_view.MapDOF(2, m, gs_gi);
            const size_t dof3 = cell_transport_view.MapDOF(3, m, gs_gi);
            const size_t dof4 = cell_transport_view.MapDOF(4, m, gs_gi);
            const size_t dof5 = cell_transport_view.MapDOF(5, m, gs_gi);

            destination_phi_[dof0 + gsg] += w * bg[0];
            destination_phi_[dof1 + gsg] += w * bg[1];
            destination_phi_[dof2 + gsg] += w * bg[2];
            destination_phi_[dof3 + gsg] += w * bg[3];
            destination_phi_[dof4 + gsg] += w * bg[4];
            destination_phi_[dof5 + gsg] += w * bg[5];
          }
        }
      }

      if (save_angular_flux_)
      {
        double* cell_psi_data =
          &destination_psi_[discretization_.MapDOFLocal(cell, 0, groupset_.psi_uk_man_, 0, 0)];

        for (int i = 0; i < 6; ++i)
        {
          const size_t imap =
            i * groupset_angle_group_stride_ + direction_num * groupset_group_stride_;
          for (size_t gsg = 0; gsg < gs_size; ++gsg)
            cell_psi_data[imap + gsg] = b[gsg * 6 + i];
        }
      }

      int out_face_counter = -1;
      for (size_t f = 0; f < cell_num_faces; ++f)
      {
        if (face_orientations[f] != FaceOrientation::OUTGOING) continue;

        const auto& face = cell.faces[f];
        const auto& IntF_shapeI = unit_cell_matrices_[cell_local_id].intS_shapeI[f];
        const size_t num_face_nodes = cell_mapping.GetNumFaceNodes(f);
        const bool is_local_face = cell_transport_view.IsFaceLocal(f);
        const bool is_boundary = !face.has_neighbor;
        const bool is_reflecting =
          is_boundary && angle_set.GetBoundaries()[face.neighbor_id]->IsReflecting();
        const double mu_wt_f = wt * face_mu_values[f];

        ++out_face_counter;

        if (!is_boundary && !is_local_face)
          ++deploc_face_counter;

        for (size_t fi = 0; fi < num_face_nodes; ++fi)
        {
          const int i = cell_mapping.MapFaceNode(f, fi);

          if (is_boundary)
          {
            const double flux_i = mu_wt_f * IntF_shapeI(i);
            for (size_t gsg = 0; gsg < gs_size; ++gsg)
              cell_transport_view.AddOutflow(f, gs_gi + gsg, flux_i * b[gsg * 6 + i]);
          }

          double* psi = nullptr;
          if (is_local_face)
            psi = fluds.OutgoingPsi(spls_index, out_face_counter, fi, as_ss_idx);
          else if (!is_boundary)
            psi = fluds.NLOutgoingPsi(deploc_face_counter, fi, as_ss_idx);
          else if (is_reflecting)
            psi = angle_set.PsiReflected(face.neighbor_id, direction_num, cell_local_id, f, fi);
          else
            continue;

          if (!is_boundary || is_reflecting)
          {
            for (size_t gsg = 0; gsg < gs_size; ++gsg)
              psi[gsg] = b[gsg * 6 + i];
          }
        }
      }
    } // angle subset
  }   // cells
}

} // namespace opensn
