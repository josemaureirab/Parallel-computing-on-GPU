#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include "tools.h"

#define PRINT 1
//#define BSIZE 4

using namespace std;
int main( int argc, char**  argv  ){
	if (argc < 4){
		fprintf(stderr, "run as ./prog n nt gpu\n\nn: matriz size\nnt: num threads (CPU)\ngpu: 1 on, 0 off\n\n");
		exit(EXIT_FAILURE);	
	}

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Select Device
	// HANDLE_ERROR(  cudaSetDevice(0)  ) ;
	
	// Size
	int n = atoi(argv[1]);
	int nt = atoi(argv[2]);
	int gpu = atoi(argv[3]);

	// Create Data host n x n
	REAL *a, *b, *c;	
	a = (REAL *)malloc( sizeof(REAL) * n * n  );
	b = (REAL *)malloc( sizeof(REAL) * n * n  );
    c = (REAL *)malloc( sizeof(REAL) * n * n  );
	for (int i =0; i<n*n ; i++){
		a[i] = i;
		b[i] = i;
		c[i] = 0;
	}

    if(gpu == 0){
        // VERSION CPU
	    printf("CPU matmul (%i threads)\n", nt); fflush(stdout);
	    print_dmatrix(a,n,"MAT A");
	    print_dmatrix(b,n,"MAT B");
	    cudaEventRecord(start);
	    printf("transpose..."); fflush(stdout);
        transpose(n, b);
	    printf("ok\n"); fflush(stdout);
        print_dmatrix(b,n,"MAT B TRANSPOSE");
	    printf("computing..."); fflush(stdout);
	    //cudaEventRecord(start);
	    matmul_cpu_transp_b(n, nt, a, b, c);
	    cudaEventRecord(stop);
	    printf("ok\n"); fflush(stdout);
	    cudaEventSynchronize(stop);
        float milliseconds = 0.0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("Time CPU: %f\n", milliseconds );	
        print_dmatrix(c,n,"MAT C");
    }

    if(gpu > 0){
        if(gpu == 1){
            printf("GPU matmul (classic)\n"); fflush(stdout);
        }
        else{
            printf("GPU matmul (shared mem)\n"); fflush(stdout);
        }
        // CUDA data
        REAL *a_dev, *b_dev, *c_dev;
        HANDLE_ERROR(cudaMalloc((void **)&a_dev, sizeof(REAL)*n*n));
        HANDLE_ERROR(cudaMalloc((void **)&b_dev, sizeof(REAL)*n*n));
        HANDLE_ERROR(cudaMalloc((void **)&c_dev, sizeof(REAL)*n*n));

	    cudaEventRecord(start);
        // Memcpy
        HANDLE_ERROR(cudaMemcpy(a_dev,a,sizeof(REAL)*n*n,cudaMemcpyHostToDevice)     );
        HANDLE_ERROR(cudaMemcpy(b_dev,b,sizeof(REAL)*n*n,cudaMemcpyHostToDevice)     );	
        
        // Kernel
        // asumir que n multiplo de BSIZE	
        dim3 block(BSIZE, BSIZE, 1);
        dim3 grid((n + BSIZE - 1)/BSIZE, (n + BSIZE - 1)/BSIZE, 1);
	printf("B(%i, %i, %i),   G(%i, %i, %i)\n", block.x, block.y, block.z, grid.x, grid.y, grid.z); 
	    //cudaEventRecord(start);
        if(gpu == 1){
            matmul_classic<<< grid, block >>>(n, a_dev, b_dev, c_dev);
        }
        else{
            // LOOP REALIZACIONES
                // timer t1 
                //for( LOOP K VECES )
                    matmul_sm<<< grid, block >>>(n, a_dev, b_dev, c_dev);
                    //sync
                //endfor
                //sync
                // timer t2
                //tavg = (t2-t1/k)
                // stats s;
                // s.register(tavg);
            //
            // s.stdev, s.var, s.errstd
        }
        cudaDeviceSynchronize();
        HANDLE_ERROR(cudaMemcpy(c, c_dev, sizeof(REAL) * n * n, cudaMemcpyDeviceToHost )     );



        cudaEventRecord(stop);
        printf("ok\n"); fflush(stdout);
        // Get data Devices
        cudaEventSynchronize(stop);
        float milliseconds = 0.0f;
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("Time: %f\n", milliseconds );	
        print_dmatrix(a,n,"MAT A");
        print_dmatrix(b,n,"MAT B");
        print_dmatrix(c,n,"MAT C (GPU)");
        //Free
        cudaFree(a_dev);
        cudaFree(b_dev);
        cudaFree(c_dev);
    }

	free(a);
	free(b);
	free(c);
	return 0;
}
