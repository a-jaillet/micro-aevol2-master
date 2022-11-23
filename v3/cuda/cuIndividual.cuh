//
// Created by elturpin on 22/10/2020.
//

#pragma once

#include <cstdint>

#include "aevol_constants.h"
#include "cuProtein.cuh"

template <typename T> 
struct my_array {
    size_t max_size;
    T *ary; 
}; 

struct cuGene {
    // value function of the error of the RNA hosting the gene
    uint8_t concentration{};
    // position at which translation will start, after the START
    uint start{};
    // position of the terminators of the RNA
    uint length_limit{};
};

struct cuRNA {
    uint8_t errors{};
    uint start_transcription{};
    uint transcription_length{};

    uint nb_gene{};
    my_array<cuGene> list_gene;
};

struct cuIndividual {
    __device__ void search_patterns();
    __device__ void sparse_meta();
    __device__ void transcription();
    __device__ void find_gene_per_RNA();
    __device__ void translation();

    __device__ void prepare_rnas(uint* nbPerThreads, uint* tmp_sparse_collection);

    __device__ void compute_rna(uint rna_idx) const;

    __device__ void prepare_gene(uint rna_idx) const;

    __device__ void gather_genes();

    __device__ void translate_gene(uint gene_idx) const;

    __device__ void compute_phenotype();

    __device__ void compute_fitness(const double* target);

    __device__ void clean_metadata();

    inline __device__ uint get_distance(uint a, uint b) const {
        if (a > b)
            return (b + size) - a;
        return b - a;
    }

    inline __device__ uint get_distance_ori(uint a, uint b) const {
        if (a >= b)
            return (b + size) - a;
        return b - a;
    }
    
    // Printing
    __device__ void print_metadata_summary() const;

    __device__ void print_rnas() const;

    __device__ void print_gathered_genes() const;

    __device__ void print_proteins() const;

    __device__ void print_phenotype() const;

    uint size{};
    char *genome{};
    uint8_t *promoters{};
    uint nb_terminator{};
    uint *terminators{};
    uint nb_prot_start{};
    uint *prot_start{};

    // To not renew the list_gene of rnas
    uint max_list_size{};
    cuGene * list_gene_4_rna;

    // New struct for tmp calculations, usefull in sparse function
    uint *tmp_sparse_collection{};

    uint nb_rnas{};
    cuRNA *list_rnas{};

    uint nb_gene{};
    my_array<cuGene> list_gene;
    my_array<cuProtein> list_protein;

    double phenotype[FUZZY_SAMPLING]{};
    double fitness{};
};

__device__
void pseudo_new_my_array_gene(my_array<cuGene> * my_arr, size_t new_size);

__device__
void pseudo_new_my_array_prot(my_array<cuProtein> * my_arr, size_t new_size);