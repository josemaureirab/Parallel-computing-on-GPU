#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <cstdlib> // Para atoi

// Kernel sin memoria compartida
__global__ void matmul_no_shared(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float value = 0.0f;
        for (int k = 0; k < N; ++k) {
            value += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = value;
    }
}

// Kernel con memoria compartida
__global__ void matmul_shared(float *A, float *B, float *C, int N, int BSIZE) {
    extern __shared__ float shared_memory[]; // Memoria compartida dinámica
    float* tileA = shared_memory;
    float* tileB = shared_memory + BSIZE * BSIZE;

    int row = blockIdx.y * BSIZE + threadIdx.y;
    int col = blockIdx.x * BSIZE + threadIdx.x;
    float value = 0.0f;

    for (int tileIdx = 0; tileIdx < (N + BSIZE - 1) / BSIZE; ++tileIdx) {
        // Cargar bloques de A y B en shared memory
        if (row < N && (tileIdx * BSIZE + threadIdx.x) < N)
            tileA[threadIdx.y * BSIZE + threadIdx.x] = A[row * N + tileIdx * BSIZE + threadIdx.x];
        else
            tileA[threadIdx.y * BSIZE + threadIdx.x] = 0.0f;

        if (col < N && (tileIdx * BSIZE + threadIdx.y) < N)
            tileB[threadIdx.y * BSIZE + threadIdx.x] = B[(tileIdx * BSIZE + threadIdx.y) * N + col];
        else
            tileB[threadIdx.y * BSIZE + threadIdx.x] = 0.0f;

        __syncthreads();

        // Calcular productos parciales
        for (int k = 0; k < BSIZE; ++k) {
            value += tileA[threadIdx.y * BSIZE + k] * tileB[k * BSIZE + threadIdx.x];
        }
        __syncthreads();
    }

    // Escribir el resultado final en C
    if (row < N && col < N) {
        C[row * N + col] = value;
    }
}

// Función para inicializar matrices en el host
void initialize_matrix(float *matrix, int size) {
    for (int i = 0; i < size * size; ++i) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

int main(int argc, char** argv) {
    // Verificar parámetros
    if (argc != 3) {
        std::cerr << "Uso: " << argv[0] << " <N> <BSIZE>" << std::endl;
        return 1;
    }

    int N = atoi(argv[1]);
    int BSIZE = atoi(argv[2]);

    size_t bytes = N * N * sizeof(float);

    // Host matrices
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C_no_shared = (float *)malloc(bytes);
    float *h_C_shared = (float *)malloc(bytes);

    initialize_matrix(h_A, N);
    initialize_matrix(h_B, N);

    // Device matrices
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 threads(BSIZE, BSIZE);
    dim3 blocks((N + BSIZE - 1) / BSIZE, (N + BSIZE - 1) / BSIZE);

    // Medir tiempo para matmul_no_shared
    auto start = std::chrono::high_resolution_clock::now();
    matmul_no_shared<<<blocks, threads>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration_no_shared = end - start;

    cudaMemcpy(h_C_no_shared, d_C, bytes, cudaMemcpyDeviceToHost);

    // Medir tiempo para matmul_shared
    start = std::chrono::high_resolution_clock::now();
    size_t shared_mem_size = 2 * BSIZE * BSIZE * sizeof(float);
    matmul_shared<<<blocks, threads, shared_mem_size>>>(d_A, d_B, d_C, N, BSIZE);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration_shared = end - start;

    cudaMemcpy(h_C_shared, d_C, bytes, cudaMemcpyDeviceToHost);

    // Mostrar resultados
    std::cout << "Tiempo (sin shared memory): " << duration_no_shared.count() << " ms\n";
    std::cout << "Tiempo (con shared memory): " << duration_shared.count() << " ms\n";

    // Limpiar memoria
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C_no_shared);
    free(h_C_shared);

    return 0;
}
