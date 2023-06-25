// OCN dummy driver
// partially copied from
// https://github.com/mrnorman/YAKL/blob/main/unit/CArray/CArray.cpp

#include "ocndummy.h"

void dummy(int argc, char **argv) {

   yakl::init();
   {
      int constexpr d1 = 2;
      int constexpr d2 = 3;

      real1d test1d("test1d", d1);
      real2d test2d("test2d", d1, d2);

      yakl::memset(test1d, 0.f);
      yakl::memset(test2d, 0.f);

      parallel_for(
          YAKL_AUTO_LABEL(), Bounds<1>(d1),
          YAKL_LAMBDA(int i1) { test1d(i1) = 1; });
      parallel_for(
          YAKL_AUTO_LABEL(), Bounds<2>(d1, d2),
          YAKL_LAMBDA(int i1, int i2) { test2d(i1, i2) = 1; });

      if (yakl::intrinsics::sum(test1d) != d1) {
         die("LOOPS: wrong sum for test1d");
      }
      if (yakl::intrinsics::sum(test2d) != d1 * d2) {
         die("LOOPS: wrong sum for test2d");
      }

      if (test1d.get_rank() != 1) {
         die("Ranks: wrong rank for test1d");
      }
      if (test2d.get_rank() != 2) {
         die("Ranks: wrong rank for test2d");
      }

      if (test1d.get_elem_count() != d1) {
         die("get_elem_count: wrong value for test1d");
      }
      if (test2d.get_elem_count() != d1 * d2) {
         die("get_elem_count: wrong value for test2d");
      }

      if (yakl::intrinsics::sum(test1d.get_dimensions()) != d1) {
         die("get_dimensions: wrong value for test1d");
      }
      if (yakl::intrinsics::sum(test2d.get_dimensions()) != d1 + d2) {
         die("get_dimensions: wrong value for test2d");
      }

      if (test1d.extent(0) != d1) {
         die("extent: wrong value for test1d");
      }
      if (test2d.extent(1) != d2) {
         die("extent: wrong value for test2d");
      }
   }
   yakl::finalize();
}
