// kmeans_cuda.cu
//
// Versão CUDA do K-Means.
// Compilação sugerida:
//   nvcc -O3 -arch=sm_70 -o kmeans_cuda kmeans_cuda.cu
//
// PREENCHER TEMPOS NO CABEÇALHO DEPOIS DAS MEDIÇÕES.
//

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <ctime>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Simples leitura de CSV (igual aos anteriores, versão C++)
float *read_csv(const char *filename, int *outN, int *outD) {
    FILE *f = fopen(filename, "r");
    if (!f) { perror("fopen"); return NULL; }

    char *line = NULL;
    size_t len = 0;
    ssize_t read;
    int N = 0, D = -1;
    while ((read = getline(&line, &len, f)) != -1) {
        if (read <= 1) continue;
        N++;
        if (D < 0) {
            int d = 1;
            for (ssize_t i = 0; i < read; i++)
                if (line[i] == ',') d++;
            D = d;
        }
    }
    rewind(f);

    float *data = (float *)malloc((size_t)N * (size_t)D * sizeof(float));
    if (!data) { fprintf(stderr, "malloc failed\n"); fclose(f); free(line); return NULL; }

    int row = 0;
    while ((read = getline(&line, &len, f)) != -1 && row < N) {
        if (read <= 1) continue;
        int col = 0;
        char *token = strtok(line, ",\n\r");
        while (token && col < D) {
            data[row * D + col] = (float)atof(token);
            token = strtok(NULL, ",\n\r");
            col++;
        }
        row++;
    }

    free(line);
    fclose(f);
    *outN = N; *outD = D;
    return data;
}

void init_centroids(float *data, float *centroids, int N, int D, int K) {
    for (int k = 0; k < K; k++)
        for (int d = 0; d < D; d++)
            centroids[k * D + d] = data[k * D + d];
}

__global__ void assign_clusters_cuda(const float *data, const float *centroids,
                                     int *labels, int N, int D, int K) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float best_dist = INFINITY;
    int best_k = 0;

    for (int k = 0; k < K; k++) {
        float sum = 0.0f;
        for (int d = 0; d < D; d++) {
            float xi = data[i * D + d];
            float ck = centroids[k * D + d];
            float diff = xi - ck;
            sum += diff * diff;
        }
        if (sum < best_dist) {
            best_dist = sum;
            best_k = k;
        }
    }

    labels[i] = best_k;
}

// Atualização de centróides no host (CPU)
void update_centroids_cpu(const float *data, float *centroids,
                          const int *labels, int N, int D, int K) {
    float *sums   = (float *)calloc((size_t)K * (size_t)D, sizeof(float));
    int   *counts = (int *)calloc((size_t)K, sizeof(int));
    if (!sums || !counts) {
        fprintf(stderr, "malloc failed\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < N; i++) {
        int k = labels[i];
        counts[k]++;
        for (int d = 0; d < D; d++) {
            sums[k * D + d] += data[i * D + d];
        }
    }

    for (int k = 0; k < K; k++) {
        if (counts[k] > 0) {
            for (int d = 0; d < D; d++) {
                centroids[k * D + d] = sums[k * D + d] / (float)counts[k];
            }
        }
    }

    free(sums);
    free(counts);
}

void kmeans_cuda(float *h_data, int N, int D, int K, int max_iters) {
    size_t data_size = (size_t)N * (size_t)D * sizeof(float);
    size_t cent_size = (size_t)K * (size_t)D * sizeof(float);
    size_t lab_size  = (size_t)N * sizeof(int);

    float *h_centroids = (float *)malloc(cent_size);
    int   *h_labels    = (int *)malloc(lab_size);
    if (!h_centroids || !h_labels) {
        fprintf(stderr, "malloc failed\n");
        exit(EXIT_FAILURE);
    }

    init_centroids(h_data, h_centroids, N, D, K);

    // Aloca no device
    float *d_data = nullptr;
    float *d_centroids = nullptr;
    int   *d_labels = nullptr;

    cudaMalloc((void **)&d_data, data_size);
    cudaMalloc((void **)&d_centroids, cent_size);
    cudaMalloc((void **)&d_labels, lab_size);

    cudaMemcpy(d_data, h_data, data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, h_centroids, cent_size, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize  = (N + blockSize - 1) / blockSize;

    for (int it = 0; it < max_iters; it++) {
        // 1) Atribuição na GPU
        assign_clusters_cuda<<<gridSize, blockSize>>>(d_data, d_centroids,
                                                      d_labels, N, D, K);
        cudaDeviceSynchronize();

        // 2) Copia labels para CPU
        cudaMemcpy(h_labels, d_labels, lab_size, cudaMemcpyDeviceToHost);

        // 3) Atualiza centróides na CPU
        update_centroids_cpu(h_data, h_centroids, h_labels, N, D, K);

        // 4) Copia novos centróides para GPU
        cudaMemcpy(d_centroids, h_centroids, cent_size, cudaMemcpyHostToDevice);
    }

    printf("Centroides finais (primeiros 3 clusters):\n");
    int showK = (K < 3) ? K : 3;
    for (int k = 0; k < showK; k++) {
        printf("C%d: ", k);
        for (int d = 0; d < D; d++) {
            printf("%f ", h_centroids[k * D + d]);
        }
        printf("\n");
    }

    cudaFree(d_data);
    cudaFree(d_centroids);
    cudaFree(d_labels);
    free(h_centroids);
    free(h_labels);
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr,
                "Uso: %s <arquivo_csv> <K> <max_iters>\n",
                argv[0]);
        return EXIT_FAILURE;
    }

    const char *filename = argv[1];
    int K = atoi(argv[2]);
    int max_iters = atoi(argv[3]);

    int N, D;
    float *data = read_csv(filename, &N, &D);
    if (!data) {
        fprintf(stderr, "Falha ao ler o CSV.\n");
        return EXIT_FAILURE;
    }

    printf("N=%d, D=%d, K=%d, iters=%d\n", N, D, K, max_iters);

    double t0 = now_s();
    kmeans_cuda(data, N, D, K, max_iters);
    double t1 = now_s();

    printf("Tempo CUDA: %.6f s\n", t1 - t0);

    free(data);
    return EXIT_SUCCESS;
}
