//
// Created by elturpin on 03/12/2020.
//

#include "cuExpManager.h"

#include <cstdio>
#include <fstream>
#include <cassert>
#include <chrono>
#include <cmath>
#include <zlib.h>

#include "AeTime.h"

#include "cuIndividual.cuh"
#include "cuda_kernels.cuh"

using namespace std::chrono;
using namespace std;

#if !defined(_NDEBUG)
#define checkCuda(X) { \
    auto result = X; \
    if (result != cudaSuccess) { \
        fprintf(stderr, "CUDA Runtime Error: %s in file %s line %d\\n\n", \
                cudaGetErrorString(result), __FILE__, __LINE__); \
        assert(result == cudaSuccess); \
    } }

#define CHECK_KERNEL \
cudaDeviceSynchronize(); \
checkCuda(cudaGetLastError());
#else
#define checkCuda(X) X
#define CHECK_KERNEL
#endif

cuExpManager::cuExpManager(const ExpManager* cpu_exp) {
    grid_height_ = cpu_exp->grid_height_;
    grid_width_ = cpu_exp->grid_height_;

    mutation_rate_ = cpu_exp->mutation_rate_;

    backup_step_ = cpu_exp->backup_step_;

    nb_indivs_ = grid_height_ * grid_width_;

    genome_length_ = cpu_exp->internal_organisms_[0]->length();
    host_individuals_ = new char *[nb_indivs_];
    for (int i = 0; i < nb_indivs_; ++i) {
        host_individuals_[i] = new char[genome_length_];
        const auto& org = cpu_exp->internal_organisms_[i];
        memcpy(host_individuals_[i], org->dna_->seq_.data(), genome_length_ * sizeof(char));
    }

    target_ = new double[FUZZY_SAMPLING];
    memcpy(target_, cpu_exp->target, FUZZY_SAMPLING * sizeof(double));

    seed_ = cpu_exp->seed_;
    nb_counter_ = cpu_exp->rng_->counters().size();
    counters_ = new ctr_value_type[nb_counter_];
    for (int i = 0; i < nb_counter_; ++i) {
        counters_[i] = cpu_exp->rng_->counters()[i];
    }
}

cuExpManager::~cuExpManager() {
    device_data_destructor();
    delete[] counters_;
    delete[] target_;
    delete[] host_individuals_;
}

void cuExpManager::run_a_step() {
    // To time Kernel, take advantage of cuda streams with CUDA Events
    // cf: https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/

    auto threads_per_block = 32; // arbitrary : better if multiple of 32
    // Selection
    dim3 bloc_dim(threads_per_block / 2, threads_per_block / 2);
    auto grid_x = ceil((float) grid_width_ / (float) bloc_dim.x);
    auto grid_y = ceil((float) grid_height_ / (float) bloc_dim.y);
    dim3 grid_dim(grid_x, grid_y);
    selection<<<grid_dim, bloc_dim>>>(grid_height_,
                                      grid_width_,
                                      device_individuals_,
                                      rand_service_,
                                      reproducers_);
    CHECK_KERNEL;

    // Reproduction
    reproduction<<<nb_indivs_, threads_per_block>>>(nb_indivs_, device_individuals_, reproducers_, all_parent_genome_);
    CHECK_KERNEL;

    // Mutation
    auto grid_dim_1d = ceil((float)nb_indivs_ / (float)threads_per_block);
    do_mutation<<<grid_dim_1d, threads_per_block>>>(nb_indivs_, device_individuals_, mutation_rate_, rand_service_);
    CHECK_KERNEL;

    // Evaluation
    evaluate_population();

    // Swap genome information
    swap_parent_child_genome<<<grid_dim_1d, threads_per_block>>>(nb_indivs_, device_individuals_, all_parent_genome_);
    swap(all_parent_genome_, all_child_genome_);

    CHECK_KERNEL;
}

__global__
void evaluate_population_kernel1(int nb_indivs, cuIndividual* device_individuals)
{
    // Always one block per individuals
    search_patterns(nb_indivs, device_individuals);
    sparse_meta(nb_indivs, device_individuals);
    transcription(nb_indivs, device_individuals);
    find_gene_per_RNA(nb_indivs, device_individuals);
}


__global__
void evaluate_population_kernel2(int nb_indivs, cuIndividual* device_individuals, double* device_target)
{
    // Always one block per individuals
    translation(nb_indivs, device_individuals);
    compute_phenotype(nb_indivs, device_individuals);
    compute_fitness(nb_indivs, device_individuals, device_target);
}


void cuExpManager::evaluate_population() {
    dim3 my_blockDim(32); // keep a multiple of 32 (warp size)
    dim3 my_gridDim(nb_indivs_);
    dim3 one_indiv_by_thread_grid(ceil((float)nb_indivs_ / (float)my_blockDim.x));
    clean_metadata<<<one_indiv_by_thread_grid, my_blockDim>>>(nb_indivs_, device_individuals_);
    CHECK_KERNEL;
    evaluate_population_kernel1<<<my_gridDim, my_blockDim, (genome_length_ + PROM_SIZE) * sizeof(char)>>>(nb_indivs_, device_individuals_);
    CHECK_KERNEL;
    gather_genes<<<one_indiv_by_thread_grid, my_blockDim>>>(nb_indivs_, device_individuals_);
    CHECK_KERNEL;
    evaluate_population_kernel2<<<my_gridDim, my_blockDim, (genome_length_ + PROM_SIZE) * sizeof(char)>>>(nb_indivs_, device_individuals_, device_target_);
    CHECK_KERNEL;
}

void cuExpManager::run_evolution(int nb_gen) {
    const int MB_SIZE = 8;
    // Set a heap size of MB_SIZE megabytes.
    // cf: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#heap-memory-allocation
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, MB_SIZE * 1024 * 1024);
    // Default is 8 MB if not specified

    cout << "Transfer" << endl;
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    transfer_to_device();
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    auto duration_transfer_in = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
    cout << "Transfer done in " << duration_transfer_in << " ??s" << endl;

    // Evaluation of population at generation 0
    auto threads_per_block = 64;
    auto grid_dim_1d = ceil((float) nb_indivs_ / (float) threads_per_block);
    evaluate_population();
    swap_parent_child_genome<<<grid_dim_1d, threads_per_block>>>(nb_indivs_, device_individuals_, all_parent_genome_);
    CHECK_KERNEL
    swap(all_parent_genome_, all_child_genome_);
    check_result<<<1,1>>>(nb_indivs_, device_individuals_);
    CHECK_KERNEL

    printf("Running evolution GPU from %d to %d\n", AeTime::time(), AeTime::time() + nb_gen);
    for (int gen = 0; gen < nb_gen; gen++) {
        AeTime::plusplus();
        // printf("Generation %d : \n",AeTime::time());

        run_a_step();

        if (AeTime::time() % backup_step_ == 0) {
            save(AeTime::time());
            cout << "Backup for generation " << AeTime::time() << " done !" << endl;
        }
    }
    write_fitness_result();
    check_result<<<1,1>>>(nb_indivs_, device_individuals_);
    CHECK_KERNEL
}

void cuExpManager::save(int t) const {
    char exp_backup_file_name[255];
    sprintf(exp_backup_file_name, "backup/backup_%d.zae", t);

    // -------------------------------------------------------------------------
    // Open backup files
    // -------------------------------------------------------------------------
    gzFile exp_backup_file = gzopen(exp_backup_file_name, "w");

    // -------------------------------------------------------------------------
    // Check that files were correctly opened
    // -------------------------------------------------------------------------
    if (exp_backup_file == Z_NULL) {
        printf("Error: could not open backup file %s\n",
               exp_backup_file_name);
        exit(EXIT_FAILURE);
    }

    // -------------------------------------------------------------------------
    // Write the backup file
    // -------------------------------------------------------------------------
    gzwrite(exp_backup_file, &t, sizeof(t));

    gzwrite(exp_backup_file, &grid_height_, sizeof(grid_height_));
    gzwrite(exp_backup_file, &grid_width_, sizeof(grid_width_));
    gzwrite(exp_backup_file, &backup_step_, sizeof(backup_step_));
    gzwrite(exp_backup_file, &mutation_rate_, sizeof(mutation_rate_));


    for (int i = 0; i < FUZZY_SAMPLING; i++) {
        double tmp = target_[i];
        gzwrite(exp_backup_file, &tmp, sizeof(tmp));
    }

    transfer_to_host();

    for (int indiv_id = 0; indiv_id < nb_indivs_; indiv_id++) {
        gzwrite(exp_backup_file, &genome_length_, sizeof(genome_length_));
        gzwrite(exp_backup_file, host_individuals_[indiv_id], genome_length_ * sizeof(char));
    }

    gzwrite(exp_backup_file, &seed_, sizeof(seed_));
    gzwrite(exp_backup_file, counters_, nb_counter_ * sizeof(counters_[0]));

    if (gzclose(exp_backup_file) != Z_OK) {
        cerr << "Error while closing backup file" << endl;
    }
}

void cuExpManager::load(int t) {
    // Unused method because we load data from ExpManager
}


void cuExpManager::transfer_to_device() {
    // Allocate memory for individuals in device world
    checkCuda(cudaMalloc(&(device_individuals_), nb_indivs_ * sizeof(cuIndividual)));
    auto all_genomes_size = nb_indivs_ * genome_length_;
    // For each genome, we add a phantom space at the end.
    auto all_genomes_size_w_phantom = all_genomes_size + nb_indivs_ * PROM_SIZE;
    checkCuda(cudaMalloc(&(all_genome_char_), all_genomes_size_w_phantom * sizeof (char)));
    checkCuda(cudaMalloc(&(all_child_genome_), nb_indivs_ * sizeof(cuBitSet)));
    checkCuda(cudaMalloc(&(all_parent_genome_), nb_indivs_ * sizeof(cuBitSet)));

    auto nb_all_bit_in_bitsets = ceil((float) all_genomes_size_w_phantom / 8.0f);
    checkCuda(cudaMalloc(&(all_bits_in_bitset_), nb_all_bit_in_bitsets * sizeof(char)));
    checkCuda(cudaMalloc(&(all_parents_bits_in_bitset_), nb_all_bit_in_bitsets * sizeof(char)));


    uint8_t* all_promoters;
    uint* all_terminators;

    // New collection for tmp use in sparse methods
    uint* all_tmp_sparse_collection;

    // New value use in malloc to manage list_gene
    int list_max_size = 32;
    cuGene * list_all_cu_gene;
    checkCuda(cudaMalloc(&(list_all_cu_gene), nb_indivs_ * list_max_size * sizeof(cuGene)));

    cuProtein * list_all_cu_prot;
    checkCuda(cudaMalloc(&(list_all_cu_prot), nb_indivs_ * list_max_size * sizeof(cuProtein)));

    // We consider that there is maximum 32 rna per indiv and 32 gene per rna
    cuGene * list_all_rna_cu_gene;    
    checkCuda(cudaMalloc(&(list_all_rna_cu_gene),  nb_indivs_ * list_max_size * list_max_size * sizeof(cuGene)));

    uint* all_prot_start;
    cuRNA* all_rnas;
    checkCuda(cudaMalloc(&(all_promoters), all_genomes_size * sizeof(uint8_t)));
    checkCuda(cudaMalloc(&(all_terminators), all_genomes_size * sizeof(uint)));
    checkCuda(cudaMalloc(&(all_tmp_sparse_collection), all_genomes_size * sizeof(uint)));
    checkCuda(cudaMalloc(&(all_prot_start), all_genomes_size * sizeof(uint)));
    checkCuda(cudaMalloc(&(all_rnas), all_genomes_size * sizeof(cuRNA)));

    // Transfer data from individual to device
    for (int i = 0; i < nb_indivs_; ++i) {
        auto offset = genome_length_ + PROM_SIZE;
        auto indiv_genome_pointer = all_genome_char_ + (i * offset);
        auto indiv_genome_phantom_pointer = indiv_genome_pointer + genome_length_;
        checkCuda(cudaMemcpy(indiv_genome_pointer, host_individuals_[i], genome_length_, cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(indiv_genome_phantom_pointer, host_individuals_[i], PROM_SIZE, cudaMemcpyHostToDevice));
    }

    init_device_population<<<1, 1>>>(nb_indivs_, genome_length_, device_individuals_, all_child_genome_, all_bits_in_bitset_, all_parent_genome_, all_parents_bits_in_bitset_,
                                     all_genome_char_, all_promoters, all_terminators, all_tmp_sparse_collection, all_prot_start, all_rnas, list_all_cu_gene, list_all_rna_cu_gene, list_all_cu_prot, list_max_size);
    CHECK_KERNEL

    // Transfer phenotypic target
    checkCuda(cudaMalloc(&(device_target_), FUZZY_SAMPLING * sizeof(double)));
    checkCuda(cudaMemcpy(device_target_, target_, FUZZY_SAMPLING * sizeof(double), cudaMemcpyHostToDevice));

    // Allocate memory for reproduction data
    checkCuda(cudaMalloc(&(reproducers_), nb_indivs_ * sizeof(int)));

    // Initiate Random Number generator
    RandService tmp;
    checkCuda(cudaMalloc(&(tmp.rng_counters), nb_counter_ * sizeof(ctr_value_type)));
    checkCuda(cudaMemcpy(tmp.rng_counters, counters_, nb_counter_ * sizeof(ctr_value_type), cudaMemcpyHostToDevice));
    tmp.seed = {{0, seed_}};
    tmp.phase_size = nb_indivs_;
    assert(nb_counter_ == tmp.phase_size * NPHASES);

    checkCuda(cudaMalloc(&(rand_service_), sizeof(RandService)));
    checkCuda(cudaMemcpy(rand_service_, &tmp, sizeof(RandService), cudaMemcpyHostToDevice));


    //  Create a array to store the results fitness
    host_fitness_results_ = (double *) malloc(nb_indivs_*sizeof(double));

    checkCuda(cudaMalloc(&(device_fitness_results_), nb_indivs_*sizeof(double)));


//    check_rng<<<1, 1>>>(rand_service_);
//    check_target<<<1, 1>>>(device_target_);
//    CHECK_KERNEL
}

void cuExpManager::transfer_to_host() const {
    for (int i = 0; i < nb_indivs_; ++i) {
        auto offset = genome_length_ + PROM_SIZE;
        auto indiv_genome_pointer = all_parent_genome_ + (i * offset);
        checkCuda(cudaMemcpy(host_individuals_[i], indiv_genome_pointer, genome_length_, cudaMemcpyDeviceToHost));
    }

    RandService tmp;
    checkCuda(cudaMemcpy(&tmp, rand_service_, sizeof(RandService), cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy(counters_, tmp.rng_counters, nb_counter_ * sizeof(ctr_value_type), cudaMemcpyDeviceToHost));
}

void cuExpManager::device_data_destructor() {
    int thread_per_block = 64;
    int nbBlocks = ceil(nb_indivs_ / thread_per_block);
    clean_population_metadata<<<nbBlocks, thread_per_block>>>(nb_indivs_, device_individuals_);
    RandService tmp_rand;
    checkCuda(cudaMemcpy(&tmp_rand, rand_service_, sizeof(RandService), cudaMemcpyDeviceToHost));
    checkCuda(cudaFree(tmp_rand.rng_counters));
    checkCuda(cudaFree(rand_service_));

    checkCuda(cudaFree(reproducers_));

    checkCuda(cudaFree(device_target_));

    cuIndividual tmp;
    checkCuda(cudaMemcpy(&tmp, device_individuals_, sizeof(cuIndividual), cudaMemcpyDeviceToHost));
    checkCuda(cudaFree(tmp.promoters));
    checkCuda(cudaFree(tmp.terminators));
    checkCuda(cudaFree(tmp.prot_start));
    checkCuda(cudaFree(tmp.list_rnas));
    checkCuda(cudaFree(tmp.list_gene.ary));
    checkCuda(cudaFree(tmp.list_protein.ary));
    checkCuda(cudaFree(tmp.tmp_sparse_collection));
    checkCuda(cudaFree(tmp.list_gene_4_rna));
    checkCuda(cudaFree(all_bits_in_bitset_));
    checkCuda(cudaFree(all_parents_bits_in_bitset_));
    checkCuda(cudaFree(all_parent_genome_));
    checkCuda(cudaFree(all_child_genome_));
    checkCuda(cudaFree(device_individuals_));
    checkCuda(cudaFree(all_genome_char_));


    cudaDeviceReset();
}

void cuExpManager::write_fitness_result()
{
    int step_per_block = 32;
    int nbBlocks = ceil(nb_indivs_/step_per_block);
    write_fitness_result_device<<<nbBlocks, step_per_block>>>(nb_indivs_, device_fitness_results_, device_individuals_);
    cudaDeviceSynchronize();
    checkCuda(cudaMemcpy(host_fitness_results_, device_fitness_results_, nb_indivs_*sizeof(double), cudaMemcpyDeviceToHost));

    ofstream res("result_fitness_gpu.csv");

    for (int i = 0; i < nb_indivs_; i++)
    {
        res << host_fitness_results_[i] << endl;
    }

    res.close();
}


// __CUDA KERNELS__

// Evolution

__global__
void check_result(uint nb_indivs, cuIndividual* individuals) {
    // for (int indiv_idx = 0; indiv_idx < nb_indivs; ++indiv_idx) {
    //     const auto& indiv = individuals[indiv_idx];
    //     printf("%d: %1.10e | ", indiv_idx, indiv.fitness);
    // }
    // printf("\n");
}

__global__
void do_mutation(uint nb_indivs, cuIndividual* individuals, double mutation_rate, RandService* rand_service) {
    // One thread per individual
    auto indiv_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (indiv_idx >= nb_indivs)
        return;

    auto& indiv = individuals[indiv_idx];
    auto nb_switch = rand_service->binomial_random(indiv.size, mutation_rate, indiv_idx, MUTATION);

    for (int i = 0; i < nb_switch; ++i) {
        auto position = rand_service->gen_number_max(indiv.size, indiv_idx, MUTATION);
        // if (indiv.genome[position] == '0') indiv.genome[position] = '1';
        // else indiv.genome[position] = '0';
        indiv.genome->set(position);
        
        if (position < PROM_SIZE) // Do not forget phantom space
            // indiv.genome[position + indiv.size] = indiv.genome[position];
            indiv.genome->set(position + indiv.size, indiv.genome->get(position));

//        printf("indiv %u switch at %d\n", indiv_idx, position);
    }
}

__global__
void reproduction(uint nb_indivs, cuIndividual* individuals, const int* reproducers, cuBitSet* all_parent_genome) {
    // One block per individuals
    auto indiv_idx = blockIdx.x;
    auto idx = threadIdx.x;
    auto rr_width = blockDim.x;

    if (indiv_idx >= nb_indivs) {
        return;
    }

    __shared__ cuBitSet * parent_genome;
    __shared__ cuBitSet * child_genome;
    __shared__ uint size;

    if (threadIdx.x == 0) {
        auto& child = individuals[indiv_idx];
        // size do not change, what a chance !
        size = child.size;
        auto offset = reproducers[indiv_idx] * (child.size + PROM_SIZE); // Do not forget phantom space
        parent_genome = all_parent_genome + offset;
        child_genome = child.genome;

    }

    __syncthreads();

    for (int position = idx; position < size + PROM_SIZE; position += rr_width) { // Do not forget phantom space
        child_genome->set(position, parent_genome->get(position));
    }
}

__global__
void selection(uint grid_height, uint grid_width, const cuIndividual* individuals, RandService* rand_service,
               int* next_reproducers) {
    // One thread per grid cell
    int grid_x = threadIdx.x + blockIdx.x * blockDim.x;
    int grid_y = threadIdx.y + blockIdx.y * blockDim.y;

    if (grid_x >= grid_width || grid_y >= grid_height)
        return;

    double local_fit_array[NEIGHBORHOOD_SIZE];
    int count = 0;
    double sum_local_fit = 0.0;

    for (int8_t i = -1; i < NEIGHBORHOOD_WIDTH - 1; i++) {
        for (int8_t j = -1; j < NEIGHBORHOOD_HEIGHT - 1; j++) {
            // Toric topology
            int cur_x = (grid_x + i + grid_width) % grid_width;
            int cur_y = (grid_y + j + grid_height) % grid_height;

            local_fit_array[count] = individuals[cur_x * grid_width + cur_y].fitness;
            sum_local_fit += local_fit_array[count];

            count++;
        }
    }

    for(int8_t i = 0 ; i < NEIGHBORHOOD_SIZE ; i++) {
        local_fit_array[i] /= sum_local_fit;
    }

    uint grid_idx = grid_x * grid_width + grid_y;

    auto selected_cell = rand_service->random_roulette(local_fit_array, NEIGHBORHOOD_SIZE, grid_idx, SELECTION);

    int x_offset = (selected_cell / NEIGHBORHOOD_WIDTH) - 1;
    int y_offset = (selected_cell % NEIGHBORHOOD_HEIGHT) - 1;
    int selected_x = (grid_x + x_offset + grid_width) % grid_width;
    int selected_y = (grid_y + y_offset + grid_height) % grid_height;

    next_reproducers[grid_idx] = selected_x * grid_height + selected_y;
}

__global__
void swap_parent_child_genome(uint nb_indivs, cuIndividual* individuals, cuBitSet* all_parent_genome) {
    // One thread per individual
    auto indiv_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (indiv_idx >= nb_indivs)
        return;

    auto& indiv = individuals[indiv_idx];
    auto offset = indiv_idx * (indiv.size + PROM_SIZE); // Do not forget phantom space
    indiv.genome = all_parent_genome + offset;
}

// Evaluation process

__global__
void clean_metadata(uint nb_indivs, cuIndividual* individuals) {
    // One thread per individual
    auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (int i = idx; i < nb_indivs; i += stride)
        individuals[i].clean_metadata();
}

__device__
void search_patterns(uint nb_indivs, cuIndividual* individuals) {
    // One block per individual
    auto indiv_idx = blockIdx.x;
    if (indiv_idx < nb_indivs)
        individuals[indiv_idx].search_patterns();
}

__device__
void sparse_meta(uint nb_indivs, cuIndividual* individuals) {
    // One block per individual
    auto indiv_idx = blockIdx.x;
    if (indiv_idx < nb_indivs)
        individuals[indiv_idx].sparse_meta();
}

__device__
void transcription(uint nb_indivs, cuIndividual* individuals) {
    // One block per individual
    auto indiv_idx = blockIdx.x;
    if (indiv_idx < nb_indivs)
        individuals[indiv_idx].transcription();
}

__device__
void find_gene_per_RNA(uint nb_indivs, cuIndividual* individuals) {
    // One block per individual
    auto indiv_idx = blockIdx.x;
    if (indiv_idx < nb_indivs)
        individuals[indiv_idx].find_gene_per_RNA();
}

__global__
void gather_genes(uint nb_indivs, cuIndividual* individuals) {
    // One thread per individual
    auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (int i = idx; i < nb_indivs; i += stride)
        individuals[i].gather_genes();
}

__device__ void translation(uint size, cuIndividual* individuals) {
    // On block per individual
    auto indiv_idx = blockIdx.x;
    if (indiv_idx < size)
        individuals[indiv_idx].translation();
}

__device__ void compute_phenotype(uint size, cuIndividual* individuals) {
    // On block per individual
    auto indiv_idx = blockIdx.x;
    if (indiv_idx < size)
        individuals[indiv_idx].compute_phenotype();
}

__device__ void compute_fitness(uint size, cuIndividual* individuals, const double* target) {
    // On block per individual
    auto indiv_idx = blockIdx.x;
    if (indiv_idx < size)
        individuals[indiv_idx].compute_fitness(target);
}

// Interface Host | Device

__global__ void clean_population_metadata(uint nb_indivs, cuIndividual* individuals) {
    uint index = blockIdx.x *blockDim.x + threadIdx.x;
    if (index < nb_indivs) {
        individuals[index].clean_metadata();
    }
}

__global__
void check_target(double* target) {
    for (int i = 0; i < FUZZY_SAMPLING; ++i) {
        if (target[i] == 0.0){
            printf("0|");
        } else {
            printf("%f|", target[i]);
        }
    }
    printf("\n");
}

__global__
void check_rng(RandService* rand_service) {
    auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    auto number = rand_service->generator({{0, rand_service->rng_counters[rand_service->phase_size]}}, rand_service->seed);
    if (idx == 0) {
        printf("seed: %lu, counter[nb_indivs]: %lu, test: %lu\n",
               rand_service->seed[1],
               rand_service->rng_counters[rand_service->phase_size],
               number[0]);
    }
}

__global__
void init_device_population(int nb_indivs, int genome_length, cuIndividual* all_individuals, cuBitSet* all_genomes, char* all_bit_in_bitset, cuBitSet* all_parent_genome, 
                            char* all_parent_bit_in_bitset, char* all_genome_char, uint8_t* all_promoters, uint* all_terminators, uint* all_tmp_sparse_collection, uint* all_prot_start, cuRNA* all_rnas, cuGene * list_all_cu_gene, cuGene * list_all_rna_cu_gene, cuProtein * list_all_cu_prot, int list_max_size) {
    
    auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    auto rr_width = blockDim.x * gridDim.x;

    for (int i = idx; i < nb_indivs; i += rr_width) {
        auto& local_indiv = all_individuals[i];
        local_indiv.size = genome_length;
        auto offset = genome_length * i;
        local_indiv.genome = all_genomes + i;
        local_indiv.promoters = all_promoters + offset;
        local_indiv.terminators = all_terminators + offset;
        local_indiv.tmp_sparse_collection = all_tmp_sparse_collection + offset;
        local_indiv.prot_start = all_prot_start + offset;
        local_indiv.list_rnas = all_rnas + offset;
        local_indiv.nb_terminator = 0;
        local_indiv.nb_prot_start = 0;
        local_indiv.nb_rnas = 0;
        local_indiv.nb_gene = 0;
        local_indiv.list_gene.max_size = list_max_size;
        local_indiv.list_gene.ary = &(list_all_cu_gene[list_max_size*i]);
        local_indiv.list_protein.max_size = list_max_size;
        local_indiv.list_protein.ary = &(list_all_cu_prot[list_max_size*i]);
        local_indiv.fitness = 0.0;

        local_indiv.max_list_size = list_max_size;
        local_indiv.list_gene_4_rna = &(list_all_rna_cu_gene[list_max_size*list_max_size*i]);

        for (int j = 0; j < FUZZY_SAMPLING; ++j) {
            local_indiv.phenotype[j] = 0.0;
        }

        // Connect genome to corresponding bytes (for child)
        local_indiv.genome->bytes = all_bit_in_bitset + i *(genome_length + PROM_SIZE);

        // for parents
        (all_parent_genome + i)->bytes = all_parent_bit_in_bitset + + i *(genome_length + PROM_SIZE);

        // Genome copying and setting
        local_indiv.genome->init_size(genome_length + PROM_SIZE);
        for (int j = 0; j < genome_length + PROM_SIZE; j++)
        {
            if (all_genome_char[j+i*(genome_length + PROM_SIZE)] == '0')
            {
                local_indiv.genome->set(j, 0);
            }
            else {
                local_indiv.genome->set(j, 1);
            }
        }
    }
}

__global__
void write_fitness_result_device(int nb_indivs, double * device_fitness_results, cuIndividual* device_individuals)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < nb_indivs)
    {
        device_fitness_results[index] = device_individuals[index].fitness;
    }
}