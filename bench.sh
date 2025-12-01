#!/bin/sh

################################################################################
# K-means Benchmark Test Script
#
# This script compiles and runs K-means implementations with various test datasets.
# Configure which tests to run and which implementations to test below.
################################################################################

# Force locale to use period as decimal separator
export LC_NUMERIC=C

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

################################################################################
# CONFIGURATION SECTION - Edit these variables to control the tests
################################################################################

# Which implementations to test (set to 1 to enable, 0 to disable)
RUN_SEQUENTIAL=1
RUN_OMP_CPU=1
RUN_OMP_GPU=0      # Set to 1 if you have OpenMP GPU offloading support
RUN_CUDA=1         # Set to 1 if you have NVIDIA GPU and CUDA

# Number of threads to use for OpenMP CPU/GPU implementations
OMP_NUM_THREADS=4

# Test datasets configuration
# Format: "filename:K:max_iters:description"
# where K = number of clusters, max_iters = maximum iterations
TESTS=(
    # Small tests for quick validation
    "dataset/small_2d.csv:5:10:Small 2D dataset"
    "dataset/small_5d.csv:5:10:Small 5D dataset"

    # Medium tests
    "dataset/medium_2d.csv:10:20:Medium 2D dataset"
    "dataset/medium_5d.csv:10:20:Medium 5D dataset"

    # Large tests for benchmarking
    "dataset/large_2d.csv:20:20:Large 2D dataset"
    "dataset/large_5d.csv:20:20:Large 5D dataset"

    # Extra large tests (uncomment for stress testing)
    # "dataset/xlarge_5d.csv:25:20:XLarge 5D dataset"
)

# Number of times to run each test for averaging (set to 1 for quick testing)
NUM_RUNS=3

# Clean build before testing (set to 1 to clean, 0 to skip)
CLEAN_BUILD=0

# Output directory for results
RESULTS_DIR="results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="${RESULTS_DIR}/benchmark_${TIMESTAMP}.txt"

################################################################################
# END OF CONFIGURATION SECTION
################################################################################

# Create results directory
mkdir -p "$RESULTS_DIR"

# Print header
print_header() {
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  K-means Parallel Implementation Benchmark${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
}

# Print section header
print_section() {
    echo ""
    echo -e "${YELLOW}━━━ $1 ━━━${NC}"
    echo ""
}

# Print test info
print_test() {
    echo -e "${GREEN}▶${NC} $1"
}

# Print error
print_error() {
    echo -e "${RED}✗ ERROR:${NC} $1"
}

# Print success
print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

# Print warning
print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Check if a file exists
check_file() {
    if [ ! -f "$1" ]; then
        print_error "File not found: $1"
        return 1
    fi
    return 0
}

# Check if executable exists
check_executable() {
    if [ ! -x "$1" ]; then
        print_error "Executable not found or not executable: $1"
        return 1
    fi
    return 0
}

# Compile implementations
compile_implementations() {
    print_section "Compilation Phase"

    # Clean if requested
    if [ $CLEAN_BUILD -eq 1 ]; then
        print_test "Cleaning previous builds..."
        make clean > /dev/null 2>&1
        print_success "Clean complete"
    fi

    local compile_errors=0

    # Compile sequential
    if [ $RUN_SEQUENTIAL -eq 1 ]; then
        print_test "Compiling sequential implementation..."
        if make -C src/seq BINDIR="$(pwd)/bin" > /dev/null 2>&1; then
            print_success "Sequential compilation successful"
        else
            print_error "Sequential compilation failed"
            compile_errors=$((compile_errors + 1))
            RUN_SEQUENTIAL=0
        fi
    fi

    # Compile OpenMP CPU
    if [ $RUN_OMP_CPU -eq 1 ]; then
        print_test "Compiling OpenMP CPU implementation..."
        if make -C src/omp_cpu BINDIR="$(pwd)/bin" > /dev/null 2>&1; then
            print_success "OpenMP CPU compilation successful"
        else
            print_error "OpenMP CPU compilation failed"
            compile_errors=$((compile_errors + 1))
            RUN_OMP_CPU=0
        fi
    fi

    # Compile OpenMP GPU
    if [ $RUN_OMP_GPU -eq 1 ]; then
        print_test "Compiling OpenMP GPU implementation..."
        if make -C src/omp_gpu BINDIR="$(pwd)/bin" > /dev/null 2>&1; then
            print_success "OpenMP GPU compilation successful"
        else
            print_warning "OpenMP GPU compilation failed (this is expected if you don't have GPU offloading support)"
            compile_errors=$((compile_errors + 1))
            RUN_OMP_GPU=0
        fi
    fi

    # Compile CUDA
    if [ $RUN_CUDA -eq 1 ]; then
        print_test "Compiling CUDA implementation..."
        if make -C src/cuda BINDIR="$(pwd)/bin" > /dev/null 2>&1; then
            print_success "CUDA compilation successful"
        else
            print_warning "CUDA compilation failed (this is expected if you don't have CUDA installed)"
            compile_errors=$((compile_errors + 1))
            RUN_CUDA=0
        fi
    fi

    echo ""
    if [ $compile_errors -eq 0 ]; then
        print_success "All enabled implementations compiled successfully"
    else
        print_warning "$compile_errors implementation(s) failed to compile"
    fi

    # Check if at least one implementation compiled
    if [ $RUN_SEQUENTIAL -eq 0 ] && [ $RUN_OMP_CPU -eq 0 ] && [ $RUN_OMP_GPU -eq 0 ] && [ $RUN_CUDA -eq 0 ]; then
        print_error "No implementations compiled successfully. Exiting."
        exit 1
    fi
}

# Run a single test
run_single_test() {
    local impl_name=$1
    local exec_path=$2
    local dataset=$3
    local k=$4
    local max_iters=$5
    local num_threads=$6  # Optional, for OpenMP/CUDA

    # Run the test and capture output
    local output
    if [ -n "$num_threads" ]; then
        # OpenMP or CUDA implementation (with num_threads parameter)
        output=$("$exec_path" "$dataset" "$k" "$max_iters" "$num_threads" 2>&1)
    else
        # Sequential implementation (no num_threads parameter)
        output=$("$exec_path" "$dataset" "$k" "$max_iters" 2>&1)
    fi
    local exit_code=$?

    if [ $exit_code -ne 0 ]; then
        echo "ERROR"
        return 1
    fi

    # Extract time from output (looking for "Tempo" patterns)
    local time
    time=$(echo "$output" | grep -oP "Tempo[^:]*:\s*\K[0-9]+\.[0-9]+")

    if [ -z "$time" ]; then
        # Try alternative patterns
        time=$(echo "$output" | grep -oP "Time[^:]*:\s*\K[0-9]+\.[0-9]+")
    fi

    if [ -z "$time" ]; then
        # If still not found, try to find any decimal number followed by 's'
        time=$(echo "$output" | grep -oP "[0-9]+\.[0-9]+(?=\s*s)")
    fi

    echo "$time"
    return 0
}

# Run benchmark for one implementation
run_implementation_benchmark() {
    local impl_name=$1
    local exec_path=$2
    local needs_threads=$3  # 1 if needs num_threads parameter, 0 otherwise

    echo -e "\n${BLUE}Testing: $impl_name${NC}"
    if [ "$needs_threads" -eq 1 ]; then
        echo -e "${BLUE}Threads: $OMP_NUM_THREADS${NC}"
    fi
    echo "────────────────────────────────────────────────────────────────"

    # Check if executable exists
    if ! check_executable "$exec_path"; then
        return 1
    fi

    # Iterate through tests
    for test_spec in "${TESTS[@]}"; do
        IFS=':' read -r dataset k max_iters description <<< "$test_spec"

        # Check if dataset exists
        if ! check_file "$dataset"; then
            print_warning "Skipping test: $description"
            continue
        fi

        echo -e "\n  Test: ${description}"
        echo -e "  File: ${dataset} | K=${k} | Iterations=${max_iters}"

        # Run multiple times and collect times
        local times=()
        local total_time=0
        local successful_runs=0

        for ((run=1; run<=NUM_RUNS; run++)); do
            printf "    Run %d/%d: " "$run" "$NUM_RUNS"

            local time
            if [ "$needs_threads" -eq 1 ]; then
                time=$(run_single_test "$impl_name" "$exec_path" "$dataset" "$k" "$max_iters" "$OMP_NUM_THREADS")
            else
                time=$(run_single_test "$impl_name" "$exec_path" "$dataset" "$k" "$max_iters")
            fi
            local status=$?

            if [ $status -eq 0 ] && [ -n "$time" ]; then
                times+=("$time")
                total_time=$(echo "$total_time + $time" | bc -l)
                successful_runs=$((successful_runs + 1))
                printf "${GREEN}%.6f s${NC}\n" "$time"
            else
                print_error "Failed"
            fi
        done

        # Calculate and display average
        if [ $successful_runs -gt 0 ]; then
            local avg_time
            avg_time=$(echo "scale=6; $total_time / $successful_runs" | bc -l)
            echo -e "  ${GREEN}Average:${NC} ${avg_time} s (${successful_runs}/${NUM_RUNS} successful runs)"

            # Log to results file
            local impl_label="$impl_name"
            if [ "$needs_threads" -eq 1 ]; then
                impl_label="${impl_name} (${OMP_NUM_THREADS} threads)"
            fi
            echo "$impl_label,$description,$dataset,$k,$max_iters,$avg_time,$successful_runs" >> "$RESULTS_FILE"
        else
            print_error "All runs failed for this test"
        fi
    done
}

# Generate summary report
generate_summary() {
    print_section "Benchmark Summary"

    if [ ! -f "$RESULTS_FILE" ]; then
        print_warning "No results to summarize"
        return
    fi

    echo "Results saved to: $RESULTS_FILE"
    echo ""

    # Create CSV header
    sed -i '1i Implementation,Test Description,Dataset,K,Iterations,Avg Time (s),Successful Runs' "$RESULTS_FILE"

    # Display summary table
    echo "Performance Summary:"
    echo "────────────────────────────────────────────────────────────────"

    # Try to use column for nice formatting, fallback to cat if not available
    if command -v column &> /dev/null; then
        column -t -s',' "$RESULTS_FILE" 2>/dev/null || cat "$RESULTS_FILE"
    else
        cat "$RESULTS_FILE"
    fi
    echo ""

    # Calculate speedup if both sequential and parallel results exist
    if [ $RUN_SEQUENTIAL -eq 1 ] && [ $RUN_OMP_CPU -eq 1 ]; then
        echo ""
        echo "Speedup Analysis (Sequential vs OpenMP CPU):"
        echo "────────────────────────────────────────────────────────────────"

        # Extract unique test descriptions
        local tests=$(grep -v "^Implementation" "$RESULTS_FILE" | cut -d',' -f2 | sort -u)

        while IFS= read -r test_desc; do
            # Get sequential time
            local seq_time=$(grep "^Sequential," "$RESULTS_FILE" | grep ",$test_desc," | cut -d',' -f6)
            # Get OpenMP time
            local omp_time=$(grep "^OpenMP CPU" "$RESULTS_FILE" | grep ",$test_desc," | cut -d',' -f6)

            if [ -n "$seq_time" ] && [ -n "$omp_time" ]; then
                local speedup=$(echo "scale=2; $seq_time / $omp_time" | bc -l)
                printf "  %-30s: %.2fx\n" "$test_desc" "$speedup"
            fi
        done <<< "$tests"
        echo ""
    fi

    print_success "Benchmark complete!"
}

# Main execution
main() {
    # Start timing
    start_time=$(date +%s)

    print_header

    echo "Configuration:"
    echo "  Sequential:    $([ $RUN_SEQUENTIAL -eq 1 ] && echo -e "${GREEN}Enabled${NC}" || echo -e "${RED}Disabled${NC}")"
    echo "  OpenMP CPU:    $([ $RUN_OMP_CPU -eq 1 ] && echo -e "${GREEN}Enabled (${OMP_NUM_THREADS} threads)${NC}" || echo -e "${RED}Disabled${NC}")"
    echo "  OpenMP GPU:    $([ $RUN_OMP_GPU -eq 1 ] && echo -e "${GREEN}Enabled (${OMP_NUM_THREADS} threads)${NC}" || echo -e "${RED}Disabled${NC}")"
    echo "  CUDA:          $([ $RUN_CUDA -eq 1 ] && echo -e "${GREEN}Enabled${NC}" || echo -e "${RED}Disabled${NC}")"
    echo "  Runs per test: $NUM_RUNS"
    echo "  Total tests:   ${#TESTS[@]}"
    echo ""

    # Compile
    compile_implementations

    # Run benchmarks
    print_section "Running Benchmarks"

    if [ $RUN_SEQUENTIAL -eq 1 ]; then
        run_implementation_benchmark "Sequential" "bin/kmeans_seq" 0
    fi

    if [ $RUN_OMP_CPU -eq 1 ]; then
        run_implementation_benchmark "OpenMP CPU" "bin/kmeans_omp_cpu" 1
    fi

    if [ $RUN_OMP_GPU -eq 1 ]; then
        run_implementation_benchmark "OpenMP GPU" "bin/kmeans_omp_gpu" 1
    fi

    if [ $RUN_CUDA -eq 1 ]; then
        run_implementation_benchmark "CUDA" "bin/kmeans_cuda" 1
    fi

    # Generate summary
    generate_summary

    # End timing
    end_time=$(date +%s)
    total_time=$((end_time - start_time))

    echo ""
    echo "Total execution time: ${total_time}s"
}

# Run main
main
