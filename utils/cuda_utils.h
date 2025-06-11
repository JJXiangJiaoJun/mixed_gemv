#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

inline int getSMVersion()
{
   int device{-1};
   cudaGetDevice(&device);
   int sm_major = 0;
   int sm_minor = 0;
   cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, device);
   cudaDeviceGetAttribute(&sm_minor, cudaDevAttrComputeCapabilityMinor, device);
   return sm_major * 10 + sm_minor;
}
