/*
 * kmeans_omp_gpu.c
 *
 * Versão com OpenMP offload para GPU.
 *
 * PREENCHER APÓS MEDIÇÕES:
 *   Mesmos tempos da versão CPU + GPU (especificar dispositivo).
 *
 * Compilação (exemplo para NVPTX, ajustar para seu ambiente):
 *   clang -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda \
 *       -O3 -o kmeans_omp_gpu kmeans_omp_gpu.c -lm
 *
 * PRINCIPAL MUDANÇA:
 *   - assign_clusters_omp_gpu(): usa #pragma omp target teams distribute parallel for
 */

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <omp.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// (copie read_csv do código anterior)
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

float distance2(const float *a, const float *b, int D) {
    float sum = 0.0f;
    for (int d = 0; d < D; d++) {
        float diff = a[d] - b[d];
        sum += diff * diff;
    }
    return sum;
}

// Offload da etapa de atribuição
void assign_clusters_omp_gpu(float *data, float *centroids,
                             int *labels, int N, int D, int K) {
    // data: N*D, centroids: K*D
    size_t data_size = (size_t)N * (size_t)D;
    size_t cent_size = (size_t)K * (size_t)D;

    #pragma omp target teams distribute parallel for \
        map(to: data[0:data_size], centroids[0:cent_size]) \
        map(from: labels[0:N])
    for (int i = 0; i < N; i++) {
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
}

// Atualização dos centróides na CPU (sequencial ou com OpenMP se quiser)
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

void kmeans_omp_gpu(float *data, int N, int D, int K, int max_iters) {
    float *centroids = (float *)malloc((size_t)K * (size_t)D * sizeof(float));
    int   *labels    = (int *)malloc((size_t)N * sizeof(int));
    if (!centroids || !labels) {
        fprintf(stderr, "malloc failed\n");
        exit(EXIT_FAILURE);
    }

    init_centroids(data, centroids, N, D, K);

    for (int it = 0; it < max_iters; it++) {
        assign_clusters_omp_gpu(data, centroids, labels, N, D, K);
        update_centroids_cpu(data, centroids, labels, N, D, K);
    }

    printf("Centroides finais (primeiros 3 clusters):\n");
    int showK = (K < 3) ? K : 3;
    for (int k = 0; k < showK; k++) {
        printf("C%d: ", k);
        for (int d = 0; d < D; d++) {
            printf("%f ", centroids[k * D + d]);
        }
        printf("\n");
    }

    free(centroids);
    free(labels);
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr,
                "Uso: %s <arquivo_csv> <K> <max_iters> <num_teams_omp>\n",
                argv[0]);
        return EXIT_FAILURE;
    }

    const char *filename = argv[1];
    int K = atoi(argv[2]);
    int max_iters = atoi(argv[3]);
    int num_teams = atoi(argv[4]);

    // Opcional: controlar número de teams/threads
    // omp_set_num_threads(...) atua só na parte host.
    // Em target, pode-se usar env vars como OMP_TARGET_OFFLOAD, etc.

    int N, D;
    float *data = read_csv(filename, &N, &D);
    if (!data) {
        fprintf(stderr, "Falha ao ler o CSV.\n");
        return EXIT_FAILURE;
    }

    printf("N=%d, D=%d, K=%d, iters=%d\n", N, D, K, max_iters);

    double t0 = now_s();
    kmeans_omp_gpu(data, N, D, K, max_iters);
    double t1 = now_s();

    printf("Tempo OpenMP offload GPU: %.6f s\n", t1 - t0);

    free(data);
    return EXIT_SUCCESS;
}
