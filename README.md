Projeto: K-Means em C/C++ com OpenMP e CUDA

Descrição
---------
Este projeto implementa o algoritmo de agrupamento K-Means utilizando:

1) Versão sequencial em C (kmeans_seq.c)
2) Versão paralela em OpenMP para CPU multicore (kmeans_omp_cpu.c)
3) Versão paralela em CUDA (kmeans_cuda.cu)

A aplicação utiliza uma base de dados real em formato CSV, com N linhas
(pontos) e D colunas (features numéricas). O objetivo é agrupar os pontos
em K clusters utilizando distância Euclidiana.

Compilação
----------

Versão sequencial (CPU):
    gcc -O3 -march=native -o kmeans_seq kmeans_seq.c -lm

Versão OpenMP CPU:
    gcc -O3 -march=native -fopenmp -o kmeans_omp_cpu kmeans_omp_cpu.c -lm

Versão CUDA:
    nvcc -O3 -arch=sm_61 -std=c++11 -o kmeans_cuda kmeans_cuda.cu

Execução
--------

Formato geral do CSV:
    - Cada linha: D valores float separados por vírgula
    - Exemplo: 1.0,2.3,5.1

Exemplos de execução (mesma base de dados data.csv, K=8, max_iters=20):

1) Sequencial:
    ./kmeans_seq data.csv 8 20

3) OpenMP GPU:
    ./kmeans_omp_gpu data.csv 8 20 128

4) CUDA:
    ./kmeans_cuda data.csv 8 20

Medição de Tempo
----------------
O tempo de execução é medido internamente mediante clock_gettime
(CLOCK_MONOTONIC), e impresso em segundos no final de cada execução.

Para cumprir o requisito do trabalho, foi escolhida uma base de dados
real com tamanho suficiente para que a versão sequencial leve pelo
menos 10 segundos de execução no servidor de teste.

As tabelas com os tempos e speedups (T_seq / T_p) serão incluídas no
relatório e nos comentários do código.
