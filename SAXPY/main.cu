#include <stdio.h>
#include <cuda_runtime.h>

// Kernel de SAXPY
__global__ void saxpy(int n, float a, float *x, float *y) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        y[index] = a * x[index] + y[index];
    }
}

int main() {
    int N = 1<<20; // Tamaño del vector (1 millón de elementos)
    float *h_x, *h_y;  // Vectores en el host
    float *d_x, *d_y;  // Vectores en el dispositivo (GPU)
    float alpha = 2.0f;

    // Asignar memoria en el host
    h_x = (float*)malloc(N * sizeof(float));
    h_y = (float*)malloc(N * sizeof(float));

    // Inicializar los vectores X e Y en el host
    for (int i = 0; i < N; i++) {
        h_x[i] = 1.0f;  // Todos los elementos de X en 1.0
        h_y[i] = 2.0f;  // Todos los elementos de Y en 2.0
    }

    // Asignar memoria en el dispositivo
    cudaMalloc((void**)&d_x, N * sizeof(float));
    cudaMalloc((void**)&d_y, N * sizeof(float));

    // Copiar los datos del host al dispositivo
    cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, N * sizeof(float), cudaMemcpyHostToDevice);

    // Definir el tamaño de bloques e hilos
    int blockSize = 256;  // Tamaño de bloque
    int numBlocks = (N + blockSize - 1) / blockSize;  // Número de bloques

    // Lanzar el kernel SAXPY
    saxpy<<<numBlocks, blockSize>>>(N, alpha, d_x, d_y);

    // Sincronizar para esperar a que termine el kernel
    cudaDeviceSynchronize();

    // Copiar el resultado de vuelta al host
    cudaMemcpy(h_y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Verificar el resultado
    for (int i = 0; i < 10; i++) { // Solo imprimimos los primeros 10 valores
        printf("Y[%d] = %f\n", i, h_y[i]);
    }

    // Liberar memoria
    free(h_x);
    free(h_y);
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}

