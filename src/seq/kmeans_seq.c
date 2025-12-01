/*
 * kmeans_seq.c
 *
 * Versão SEQUENCIAL do K-Means.
 *
 * PREENCHER APÓS MEDIÇÕES:
 * Tempos de execução (em segundos) no servidor de teste:
 * - Versão sequencial (N grande o suficiente para ~10s):
 *     T_seq = ...
 *
 * Compilação sugerida:
 *     gcc -O3 -march=native -o kmeans_seq kmeans_seq.c -lm
 *
 */

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Lê um CSV simples: N linhas, D colunas (todos floats).
// Formato: v11,v12,...,v1D
//          v21,v22,...,v2D
// Retorna data[N*D] e escreve N em *outN, D em *outD.
float *read_csv(const char *filename, int *outN, int *outD) {
    FILE *f = fopen(filename, "r");
    if (!f) {
        perror("fopen");
        return NULL;
    }

    // Primeiro, vamos contar linhas e colunas
    char *line = NULL;
    size_t len = 0;
    ssize_t read;
    int N = 0, D = -1;

    while ((read = getline(&line, &len, f)) != -1) {
        if (read <= 1) continue;
        N++;
        if (D < 0) {
            // Conta vírgulas + 1
            int d = 1;
            for (ssize_t i = 0; i < read; i++) {
                if (line[i] == ',') d++;
            }
            D = d;
        }
    }
    rewind(f);

    float *data = (float *)malloc((size_t)N * (size_t)D * sizeof(float));
    if (!data) {
        fprintf(stderr, "malloc failed\n");
        fclose(f);
        free(line);
        return NULL;
    }

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

    *outN = N;
    *outD = D;
    return data;
}

void init_centroids(float *data, float *centroids, int N, int D, int K) {
    // Inicializa escolhendo os K primeiros pontos como centróides
    for (int k = 0; k < K; k++) {
        for (int d = 0; d < D; d++) {
            centroids[k * D + d] = data[k * D + d];
        }
    }
}

float distance2(const float *a, const float *b, int D) {
    float sum = 0.0f;
    for (int d = 0; d < D; d++) {
        float diff = a[d] - b[d];
        sum += diff * diff;
    }
    return sum;
}

void assign_clusters(const float *data, const float *centroids,
                     int *labels, int N, int D, int K) {
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

void update_centroids(const float *data, float *centroids,
                      const int *labels, int N, int D, int K) {
    // Zera acumuladores
    float *sums = (float *)calloc((size_t)K * (size_t)D, sizeof(float));
    int   *counts = (int *)calloc((size_t)K, sizeof(int));

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

void kmeans_seq(float *data, int N, int D, int K, int max_iters) {
    float *centroids = (float *)malloc((size_t)K * (size_t)D * sizeof(float));
    int   *labels    = (int *)malloc((size_t)N * sizeof(int));
    if (!centroids || !labels) {
        fprintf(stderr, "malloc failed\n");
        exit(EXIT_FAILURE);
    }

    init_centroids(data, centroids, N, D, K);

    for (int it = 0; it < max_iters; it++) {
        assign_clusters(data, centroids, labels, N, D, K);
        update_centroids(data, centroids, labels, N, D, K);
    }

    // Apenas exemplo de saída
    printf("Centroides finais (primeiros 3 clusters, se existirem):\n");
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

    printf("Lidos N=%d pontos, D=%d dimensoes, K=%d, iters=%d\n",
           N, D, K, max_iters);

    double t0 = now_s();
    kmeans_seq(data, N, D, K, max_iters);
    double t1 = now_s();

    printf("Tempo sequencial: %.6f s\n", t1 - t0);

    free(data);
    return EXIT_SUCCESS;
}
