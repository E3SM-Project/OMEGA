string(APPEND CPPDEFS " -DLINUX")

if (COMP_NAME STREQUAL gptl)
    string(APPEND CPPDEFS " -DHAVE_NANOTIME -DBIT64 -DHAVE_SLASHPROC -DHAVE_COMM_F2C -DHAVE_TIMES -DHAVE_GETTIMEOFDAY")
endif()

string(APPEND CMAKE_C_FLAGS_RELEASE   " -O2")
string(APPEND CMAKE_CXX_FLAGS_RELEASE " -O2")
string(APPEND CMAKE_Fortran_FLAGS_RELEASE   " -O2")

string(APPEND SPIO_CMAKE_OPTS " -DPIO_ENABLE_TOOLS:BOOL=OFF")
string(APPEND CMAKE_EXE_LINKER_FLAGS " -L$ENV{CRAY_LIBSCI_PREFIX_DIR}/lib -lsci_amd")

string(APPEND KOKKOS_OPTIONS " -DKokkos_ENABLE_HIP=On -DKokkos_ARCH_ZEN3=On -DKokkos_ARCH_VEGA90A=On")

set(USE_HIP "TRUE")