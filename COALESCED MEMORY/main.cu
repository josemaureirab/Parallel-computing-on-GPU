#include <stdio.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define N 1024 * 1024 // Tamaño del vector

// Kernel para inicializar índices aleatorios
__global__ void initRandomIndices(int *indices, int n, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        indices[idx] = curand(&state) % n; // Genera índices aleatorios dentro de [0, n)
    }
}

// Kernel para acceso coalesced
__global__ void coalescedAccess(float *input, float *output) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        output[idx] = input[idx]; // Acceso alineado
    }
}

// Kernel para acceso non-coalesced
__global__ void nonCoalescedAccess(float *input, float *output, int *indices) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        output[idx] = input[indices[idx]]; // Acceso aleatorio
    }
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Uso: %s <0: coalesced | 1: non-coalesced>\n", argv[0]);
        return -1;
    }

    // Reservar memoria en el host
    float *h_input = (float *)malloc(N * sizeof(float));
    float *h_output = (float *)malloc(N * sizeof(float));
    int *h_indices = (int *)malloc(N * sizeof(int));

    // Inicializar el vector de entrada
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)i;
    }

    // Reservar memoria en el dispositivo
    float *d_input, *d_output;
    int *d_indices;
    cudaMalloc((void **)&d_input, N * sizeof(float));
    cudaMalloc((void **)&d_output, N * sizeof(float));
    cudaMalloc((void **)&d_indices, N * sizeof(int));

    // Copiar los datos al dispositivo
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    // Configurar la ejecución del kernel
    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);

    // Inicializar índices aleatorios si se elige non-coalesced
    if (atoi(argv[1]) == 1) {
        initRandomIndices<<<gridSize, blockSize>>>(d_indices, N, time(NULL));
    }

    // Elegir el kernel según el argumento
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    if (atoi(argv[1]) == 0) {
        printf("Ejecutando acceso coalesced...\n");
        cudaEventRecord(start);
        coalescedAccess<<<gridSize, blockSize>>>(d_input, d_output);
    } else if (atoi(argv[1]) == 1) {
        printf("Ejecutando acceso non-coalesced...\n");
        cudaEventRecord(start);
        nonCoalescedAccess<<<gridSize, blockSize>>>(d_input, d_output, d_indices);
    } else {
        printf("Opción no válida: %s\n", argv[1]);
        return -1;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calcular el tiempo de ejecución
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Tiempo de ejecución: %.3f ms\n", milliseconds);

    // Liberar memoria
    free(h_input);
    free(h_output);
    free(h_indices);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_indices);

    return 0;
}

