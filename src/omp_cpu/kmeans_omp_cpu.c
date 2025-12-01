/*
 * kmeans_omp_cpu.c
 *
 * Versão paralela com OpenMP (multicore CPU).
 *
 * PREENCHER APÓS MEDIÇÕES:
 * Tempos de execução (em segundos) no servidor de teste:
 *   T_seq (referência da versão sequencial) = ...
 *   Threads: 1   -> T_1  = ...
 *             2   -> T_2  = ...
 *             4   -> T_4  = ...
 *             8   -> T_8  = ...
 *            16   -> T_16 = ...
 *            32   -> T_32 = ...
 *
 * Speedup S_p = T_seq / T_p
 *
 * Compilação sugerida:
 *   gcc -O3 -march=native -fopenmp -o kmeans_omp_cpu kmeans_omp_cpu.c -lm
 *
 * ALTERAÇÕES PRINCIPAIS EM RELAÇÃO AO SEQUENCIAL:
 *  - Uso de #pragma omp parallel for em assign_clusters_omp()
 *  - Uso de acumulação por thread + região crítica em update_centroids_omp()
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

// Mesma função de leitura do sequencial (pode copiar/colar de kmeans_seq.c)
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

// Versão paralela (CPU) da atribuição de clusters
void assign_clusters_omp(const float *data, const float *centroids,
                         int *labels, int N, int D, int K) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        const float *xi = &data[i * D];
        float best_dist = INFINITY;
        int best_k = 0;
        for (int k = 0; k < K; k++) {
            const float *ck = &centroids[k * D];
            float dist = distance2(xi, ck, D);
            if (dist < best_dist) {
                best_dist = dist;
                best_k = k;
            }
        }
        labels[i] = best_k;
    }
}

// Versão paralela (CPU) da atualização de centróides
void update_centroids_omp(const float *data, float *centroids,
                          const int *labels, int N, int D, int K) {

    float *sums   = (float *)calloc((size_t)K * (size_t)D, sizeof(float));
    int   *counts = (int *)calloc((size_t)K, sizeof(int));
    if (!sums || !counts) {
        fprintf(stderr, "malloc failed\n");
        exit(EXIT_FAILURE);
    }

    int nthreads = 1;
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        #pragma omp single
        { nthreads = omp_get_num_threads(); }

        float *local_sums   = (float *)calloc((size_t)K * (size_t)D, sizeof(float));
        int   *local_counts = (int *)calloc((size_t)K, sizeof(int));
        if (!local_sums || !local_counts) {
            fprintf(stderr, "malloc failed local\n");
            exit(EXIT_FAILURE);
        }

        #pragma omp for schedule(static)
        for (int i = 0; i < N; i++) {
            int k = labels[i];
            local_counts[k]++;
            for (int d = 0; d < D; d++) {
                local_sums[k * D + d] += data[i * D + d];
            }
        }

        // Redução manual
        #pragma omp critical
        {
            for (int k = 0; k < K; k++) {
                counts[k] += local_counts[k];
                for (int d = 0; d < D; d++) {
                    sums[k * D + d] += local_sums[k * D + d];
                }
            }
        }

        free(local_sums);
        free(local_counts);
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

void kmeans_omp_cpu(float *data, int N, int D, int K, int max_iters) {
    float *centroids = (float *)malloc((size_t)K * (size_t)D * sizeof(float));
    int   *labels    = (int *)malloc((size_t)N * sizeof(int));
    if (!centroids || !labels) {
        fprintf(stderr, "malloc failed\n");
        exit(EXIT_FAILURE);
    }

    init_centroids(data, centroids, N, D, K);

    for (int it = 0; it < max_iters; it++) {
        assign_clusters_omp(data, centroids, labels, N, D, K);
        update_centroids_omp(data, centroids, labels, N, D, K);
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
                "Uso: %s <arquivo_csv> <K> <max_iters> <num_threads>\n",
                argv[0]);
        return EXIT_FAILURE;
    }

    const char *filename = argv[1];
    int K = atoi(argv[2]);
    int max_iters = atoi(argv[3]);
    int num_threads = atoi(argv[4]);

    omp_set_num_threads(num_threads);

    int N, D;
    float *data = read_csv(filename, &N, &D);
    if (!data) {
        fprintf(stderr, "Falha ao ler o CSV.\n");
        return EXIT_FAILURE;
    }

    printf("N=%d, D=%d, K=%d, iters=%d, threads=%d\n",
           N, D, K, max_iters, num_threads);

    double t0 = now_s();
    kmeans_omp_cpu(data, N, D, K, max_iters);
    double t1 = now_s();

    printf("Tempo OpenMP CPU (%d threads): %.6f s\n",
           num_threads, t1 - t0);

    free(data);
    return EXIT_SUCCESS;
}
