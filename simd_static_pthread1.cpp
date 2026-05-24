// x86 SIMD(SSE) + pthread 高斯消元
#include <iostream>
#include <pthread.h>
#include <semaphore.h>
#include <xmmintrin.h>   // SSE
#include <emmintrin.h>
#include<cstdio>
#include<windows.h>

using namespace std;

// 矩阵规模
const int N = 1024;

// 线程数量
const int NUM_THREADS = 8;

// 全局矩阵
float A[N][N];

// 信号量
sem_t sem_main;
sem_t sem_workerstart[NUM_THREADS];
sem_t sem_workerend[NUM_THREADS];

// 线程参数
typedef struct {

    int t_id;

} threadParam_t;

// 线程函数
void* threadFunc(void* param) {

    threadParam_t* p =
        (threadParam_t*)param;

    int t_id = p->t_id;

    for (int k = 0; k < N; ++k) {

        // 等待主线程完成除法
        sem_wait(
            &sem_workerstart[t_id]
        );

        // 静态划分任务
        for (int i = k + 1 + t_id;
             i < N;
             i += NUM_THREADS) {

            // A[i][k]
            __m128 aik =
                _mm_set1_ps(A[i][k]);

            int j = k + 1;

            // SSE 每次处理4个float
            for (; j + 4 <= N; j += 4) {

                // 加载
                __m128 rowi =
                    _mm_loadu_ps(
                        &A[i][j]
                    );

                __m128 rowk =
                    _mm_loadu_ps(
                        &A[k][j]
                    );

                // rowi = rowi - aik * rowk
                __m128 temp =
                    _mm_mul_ps(aik, rowk);

                rowi =
                    _mm_sub_ps(rowi, temp);

                // 写回
                _mm_storeu_ps(
                    &A[i][j],
                    rowi
                );
            }

            // 处理剩余元素
            for (; j < N; ++j) {

                A[i][j] =
                    A[i][j]
                    - A[i][k] * A[k][j];
            }

            A[i][k] = 0.0f;
        }

        // 通知主线程
        sem_post(&sem_main);

        // 最后一轮不等待
        if (k != N - 1) {

            sem_wait(
                &sem_workerend[t_id]
            );
        }
    }

    return NULL;
}

int main() {
    system("chcp 65001");
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) A[i][j] = rand()%100;
    }
    // 初始化信号量
    sem_init(&sem_main, 0, 0);
    for (int i = 0;
         i < NUM_THREADS;
         ++i) {
        sem_init(
            &sem_workerstart[i],0,0
        );
        sem_init(
            &sem_workerend[i],0,0
        );
    }
    LARGE_INTEGER freq;
    LARGE_INTEGER start;
    LARGE_INTEGER end;

    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&start);
    // 创建线程
    pthread_t handles[NUM_THREADS];
    threadParam_t param[NUM_THREADS];

    for (int t_id = 0;
         t_id < NUM_THREADS;
         ++t_id) {

        param[t_id].t_id = t_id;

        pthread_create(
            &handles[t_id],
            NULL,
            threadFunc,
            (void*)&param[t_id]
        );
    }

    cout << "开始 x86 SIMD + pthread 高斯消元..."
         << endl;
    // 高斯消元
    for (int k = 0; k < N; ++k) {

        // 主元
        float pivot = A[k][k];

        // SSE 除法
        __m128 vpivot =
            _mm_set1_ps(pivot);

        int j = k + 1;

        for (; j + 4 <= N; j += 4) {

            __m128 rowk =
                _mm_loadu_ps(
                    &A[k][j]
                );

            rowk =
                _mm_div_ps(
                    rowk,
                    vpivot
                );

            _mm_storeu_ps(
                &A[k][j],
                rowk
            );
        }

        // 剩余元素
        for (; j < N; ++j) {

            A[k][j] =
                A[k][j] / pivot;
        }

        A[k][k] = 1.0f;

        // 唤醒工作线程
        for (int t_id = 0;
             t_id < NUM_THREADS;
             ++t_id) {

            sem_post(
                &sem_workerstart[t_id]
            );
        }

        // 等待所有线程完成
        for (int t_id = 0;
             t_id < NUM_THREADS;
             ++t_id) {

            sem_wait(&sem_main);
        }

        // 最后一轮不再post
        if (k != N - 1) {

            for (int t_id = 0;
                 t_id < NUM_THREADS;
                 ++t_id) {

                sem_post(
                    &sem_workerend[t_id]
                );
            }
        }
    }
    // 等待线程结束
    for (int t_id = 0;
         t_id < NUM_THREADS;
         ++t_id) {

        pthread_join(
            handles[t_id],
            NULL
        );
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
    // 销毁信号量
    sem_destroy(&sem_main);

    for (int i = 0;
         i < NUM_THREADS;
         ++i) {

        sem_destroy(
            &sem_workerstart[i]
        );

        sem_destroy(
            &sem_workerend[i]
        );
    }

    return 0;
}
