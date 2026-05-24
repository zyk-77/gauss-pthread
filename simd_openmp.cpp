#include <iostream>
#include <iomanip>
#include <cstdlib>
#include<cstdio>
#include <windows.h>   // Windows计时器

#ifdef _OPENMP
#include <omp.h>
#endif 

using namespace std;

const int N = 1024;
const int NUM_THREADS = 8;

float A[N][N];

int main() {
    system("chcp 65001");
    int i, j, k;
    float tmp;
    bool parallel = true;
    for(int i = 0; i < N; ++i) {
        for(int j = 0; j < N; ++j) {
            A[i][j] = rand()%100;
        }
    }

    cout << "开始计算"
         << N << "x" << N
         << " 的OPENMP+SIMD高斯消元" << endl;

    // Windows高精度计时
    LARGE_INTEGER freq;
    LARGE_INTEGER start;
    LARGE_INTEGER end;

    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&start);

    // OpenMP 并行区域
    #pragma omp parallel if(parallel) \
        num_threads(NUM_THREADS) \
        private(i, j, k, tmp)
    {
        for(k = 0; k < N; ++k) {
            // 主行处理
            #pragma omp single
            {
                tmp = A[k][k];

                for(j = k + 1; j < N; ++j) {
                    A[k][j] /= tmp;
                }

                A[k][k] = 1.0f;
            }

            // 并行消元
            #pragma omp for
            for(i = k + 1; i < N; ++i) {
                tmp = A[i][k];
                #pragma omp simd
                for(j = k + 1; j < N; ++j) A[i][j] -= tmp * A[k][j];
                A[i][k] = 0.0f;
            }
        }
    }

    QueryPerformanceCounter(&end);

    double elapsed =
        (double)(end.QuadPart - start.QuadPart)
        * 1000.0
        / freq.QuadPart;

    cout << "计算完成" << endl;
    cout << "average latency : "
         << elapsed
         << " (ms)" << endl;
    return 0;
}
