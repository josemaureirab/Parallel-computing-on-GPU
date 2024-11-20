#ifndef _TOOLS_H
#define _TOOLS_H

#define REAL float

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))





// pedir el 'n' como parametro de ejecucion del kernel
__global__ void matmul_classic(int n, REAL *a, REAL *b, REAL *c){
    // 1) coordenadada del thread --> donde en C
	int idy = threadIdx.y + blockDim.y * blockIdx.y;
	int idx = threadIdx.x + blockDim.x * blockIdx.x;

    // 2) fila idy * columna idx
	REAL r = 0.0;
	for (int k=0; k<n; k++){
		r +=  a[n*idy + k] *  b[n*k + idx];
	}
    // 3) escribir celda de C en matrix C (memoria RAM GPU)
	c[n*idy + idx] = r;
}








__global__ void matmul_sm(int n, REAL *a, REAL *b, REAL *c){
	// (1) crear memoria compartida
	__shared__ REAL as[BSIZE*BSIZE];
	__shared__ REAL bs[BSIZE*BSIZE];
	__shared__ REAL cs[BSIZE*BSIZE];


	int ltidx = threadIdx.x;
	int ltidy = threadIdx.y;
	int tidx = threadIdx.x + blockDim.x * blockIdx.x;
	int tidy = threadIdx.y + blockDim.y * blockIdx.y;

	// (2) insertar elementos en cache
	//int w=0;
	cs[ltidy*BSIZE + ltidx] = 0.0;
	__syncthreads();
	#pragma unroll
	for(int w=0; w<n; w+=BSIZE){
        // put values in cache
		as[ltidy*BSIZE + ltidx] = a[tidy*n + (ltidx + w)];
		bs[ltidy*BSIZE + ltidx] = b[(ltidy + w)*n + tidx];
		__syncthreads();

		// (3) matmul en cache
		REAL r = 0.0;
		for (int k=0; k<BSIZE; k++){
			r +=  as[BSIZE*ltidy + k] *  bs[BSIZE*k + ltidx];
		}
		cs[ltidy*BSIZE + ltidx] += r;
		__syncthreads(); // todos los threads de un mismo bloque, se esperan hasta este punto.
		//w += BSIZE;
	}
	// (4) escribir en c global
	//__syncthreads();
	c[tidy*n + tidx] = cs[ltidy*BSIZE + ltidx];
}


void matmul_cpu(int n, REAL *a, REAL *b, REAL *c){
	//#pragma omp parallel for num_threads(6)
	for(int i=0; i<n; ++i){
		for(int j=0; j<n; ++j){
			REAL sum=0.0;
			for(int k=0; k<n; ++k){
				sum += a[i*n + k]*b[k*n + j];
			}
			c[i*n + j] = sum;
		}
	}
}

void matmul_cpu_transp_b(int n, int nt, REAL *a, REAL *b, REAL *c){
	#pragma omp parallel for num_threads(nt)
	for(int i=0; i<n; ++i){
		for(int j=0; j<n; ++j){
			REAL sum=0.0;
			for(int k=0; k<n; ++k){
				sum += a[i*n + k]*b[j*n + k];
			}
			c[i*n + j] = sum;
		}
	}
}

void transpose(int n, REAL *m){
	for(int i=0; i<n; ++i){
		for(int j=0; j<n; ++j){
			if(i>j){
				REAL aux = m[i*n + j];
				m[i*n + j] = m[j*n + i];
				m[j*n + i] = aux;
			}
		}
	}
}

void print_dmatrix(REAL *matrix, int n, const char *msg ){
    if(n < 32){
        printf("\n%s:\n", msg);
        int i,j;
        for(i =0; i<n; i++){
            for( j=0; j<n; j++ ){
                printf("%f ", matrix[ n*i + j ]);
            }
            printf("\n");
        }
        printf("\n");
    }
}
#endif
