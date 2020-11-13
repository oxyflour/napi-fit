#include "cuda_utils.h"
#include <time.h>

constexpr int NX = 96, NY = 160, NZ = 160, NXY = NX * NY, NXYZ = NXY * NZ, NT = 500;
template <typename F>
void bench(F &&func, int NXYZ, int NT) {
    auto start = clock();
    func();
    float total_time = ((float) (clock() - start) / CLOCKS_PER_SEC);
    float step_time = total_time / NT;
    float perf = NXYZ / step_time / 1e6;
    printf("total time: %f s, step time: %f s, perf %f MCells/s\n", total_time, step_time, perf);
}

#ifndef USE_DYNAMIC_IDX
template <int NX, int NY, int NZ, int NXY = NX * NY, int NXYZ = NXY * NZ>
#endif
class Chunk {
    __device__ __forceinline__ int get_idx(int i, int j, int k) {
        return i + j * NX + k * NXY;
    }
    __device__ __forceinline__ void get_ijk(int g, int &i, int &j, int &k) {
        k = g / NXY;
        g -= k * NXY;
        j = g / NX;
        g -= j * NX;
        i = g;
    }
public:
#ifdef USE_DYNAMIC_IDX
    int NX, NY, NZ, NXY, NXYZ;
#endif
#ifdef USE_DYNAMIC_MEM
    float *Ex, *Ey, *Ez, *Hx, *Hy, *Hz,
        *LEx, *LEy, *LEz, *LHx, *LHy, *LHz,
        *REx, *REy, *REz, *RHx, *RHy, *RHz;
#else
    float Ex[NXYZ], Ey[NXYZ], Ez[NXYZ], Hx[NXYZ], Hy[NXYZ], Hz[NXYZ];
    float LEx[NXYZ], LEy[NXYZ], LEz[NXYZ], LHx[NXYZ], LHy[NXYZ], LHz[NXYZ];
    float REx[NXYZ], REy[NXYZ], REz[NXYZ], RHx[NXYZ], RHy[NXYZ], RHz[NXYZ];
#endif
    __device__ __forceinline__ void step_h(
#ifdef USE_DYNAMIC_POINTER
    const float * __restrict__ Ex, const float * __restrict__ Ey, const float * __restrict__ Ez,
    float *Hx, float *Hy, float *Hz,
    float *LHx, float *LHy, float *LHz, float *RHx, float *RHy, float *RHz
#endif
    ) {
        int i, j, k;
        for (auto g = cuIdx(x); g < NXYZ; g += cuDim(x)) {
            get_ijk(g, i, j, k);
            if (i > 0 && j > 0 && k > 0 && i < NX - 1 && j < NY - 1 && k < NZ - 1) {
                Hx[g] = LHx[g] * Hx[g] + RHx[g] * (Ey[get_idx(i, j, k+1)] - Ey[g] - Ez[get_idx(i, j+1, k)] + Ez[g]);
                Hy[g] = LHy[g] * Hy[g] + RHy[g] * (Ez[get_idx(i+1, j, k)] - Ez[g] - Ex[get_idx(i, j, k+1)] + Ex[g]);
                Hz[g] = LHz[g] * Hz[g] + RHz[g] * (Ex[get_idx(i, j+1, k)] - Ex[g] - Ey[get_idx(i+1, j, k)] + Ey[g]);
            }
        }
    }
    __device__ __forceinline__ void step_e(
#ifdef USE_DYNAMIC_POINTER
    float *Ex, float *Ey, float *Ez,
    const float * __restrict__ Hx, const float * __restrict__ Hy, const float * __restrict__ Hz,
    float *LEx, float *LEy, float *LEz, float *REx, float *REy, float *REz
#endif
    ) {
        int i, j, k;
        for (auto g = cuIdx(x); g < NXYZ; g += cuDim(x)) {
            get_ijk(g, i, j, k);
            if (i > 0 && j > 0 && k > 0 && i < NX - 1 && j < NY - 1 && k < NZ - 1) {
                Ex[g] = LEx[g] * Ex[g] + REx[g] * (Hy[get_idx(i, j, k-1)] - Hy[g] - Hz[get_idx(i, j-1, k)] + Hz[g]);
                Ey[g] = LEy[g] * Ey[g] + REy[g] * (Hz[get_idx(i-1, j, k)] - Hz[g] - Hx[get_idx(i, j, k-1)] + Hx[g]);
                Ez[g] = LEz[g] * Ez[g] + REz[g] * (Hx[get_idx(i, j-1, k)] - Hx[g] - Hy[get_idx(i-1, j, k)] + Hy[g]);
            }
        }
    }
};

#ifndef USE_DYNAMIC_IDX
__device__ Chunk<NX, NY, NZ> chunk;
#else
__device__ Chunk chunk;
#endif

__global__ void kernel_init(int nx, int ny, int nz,
    float *Ex, float *Ey, float *Ez, float *Hx, float *Hy, float *Hz,
    float *LEx, float *LEy, float *LEz, float *LHx, float *LHy, float *LHz,
    float *REx, float *REy, float *REz, float *RHx, float *RHy, float *RHz) {
#ifdef USE_DYNAMIC_IDX
    chunk.NX = nx; chunk.NY = ny; chunk.NZ = nz;
    chunk.NXY = nx * ny; chunk.NXYZ = nx * ny * nz;
#endif
#ifdef USE_DYNAMIC_MEM
    chunk.Ex = Ex; chunk.Ey = Ey; chunk.Ez = Ez; chunk.Hx = Hx; chunk.Hy = Hy; chunk.Hz = Hz;
    chunk.LEx = LEx; chunk.LEy = LEy; chunk.LEz = LEz; chunk.LHx = LHx; chunk.LHy = LHy; chunk.LHz = LHz;
    chunk.REx = REx; chunk.REy = REy; chunk.REz = REz; chunk.RHx = RHx; chunk.RHy = RHy; chunk.RHz = RHz;
#endif
}
__global__ void kernel_step_h(
#ifdef USE_DYNAMIC_POINTER
    const float * __restrict__ Ex, const float * __restrict__ Ey, const float * __restrict__ Ez,
    float *Hx, float *Hy, float *Hz,
    float *LHx, float *LHy, float *LHz, float *RHx, float *RHy, float *RHz
#endif
) {
    chunk.step_h(
#ifdef USE_DYNAMIC_POINTER
    Ex, Ey, Ez, Hx, Hy, Hz, LHx, LHy, LHz, RHx, RHy, RHz
#endif
    );
}
__global__ void kernel_step_e(
#ifdef USE_DYNAMIC_POINTER
    float *Ex, float *Ey, float *Ez,
    const float * __restrict__ Hx, const float * __restrict__ Hy, const float * __restrict__ Hz,
    float *LEx, float *LEy, float *LEz, float *REx, float *REy, float *REz
#endif
) {
    chunk.step_e(
#ifdef USE_DYNAMIC_POINTER
    Ex, Ey, Ez, Hx, Hy, Hz, LEx, LEy, LEz, REx, REy, REz
#endif
    );
}

int main() {
    float *Ex, *Ey, *Ez, *Hx, *Hy, *Hz,
        *LEx, *LEy, *LEz, *REx, *REy, *REz,
        *LHx, *LHy, *LHz, *RHx, *RHy, *RHz;
    kernel_init CU_DIM(1, 1) (NX, NY, NZ,
        Ex = malloc_device<float>(NXYZ), Ey = malloc_device<float>(NXYZ), Ez = malloc_device<float>(NXYZ),
        Hx = malloc_device<float>(NXYZ), Hy = malloc_device<float>(NXYZ), Hz = malloc_device<float>(NXYZ),
        LEx = malloc_device<float>(NXYZ), LEy = malloc_device<float>(NXYZ), LEz = malloc_device<float>(NXYZ),
        REx = malloc_device<float>(NXYZ), REy = malloc_device<float>(NXYZ), REz = malloc_device<float>(NXYZ),
        LHx = malloc_device<float>(NXYZ), LHy = malloc_device<float>(NXYZ), LHz = malloc_device<float>(NXYZ),
        RHx = malloc_device<float>(NXYZ), RHy = malloc_device<float>(NXYZ), RHz = malloc_device<float>(NXYZ));
    CU_ASSERT(cudaGetLastError());
    cudaDeviceSynchronize();

    bench([&] {
        for (int t = 0; t < NT; t ++) {
            kernel_step_h CU_DIM(2048, 512) (
#ifdef USE_DYNAMIC_POINTER
    Ex, Ey, Ez, Hx, Hy, Hz, LHx, LHy, LHz, RHx, RHy, RHz
#endif
            );
            kernel_step_e CU_DIM(2048, 512) (
#ifdef USE_DYNAMIC_POINTER
    Ex, Ey, Ez, Hx, Hy, Hz, LEx, LEy, LEz, REx, REy, REz
#endif
            );
        }
        cudaDeviceSynchronize();
    }, NXYZ, NT);
    return 0;
}

