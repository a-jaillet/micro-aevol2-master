//
// Created by elturpin on 22/10/2020.
//

#include "cuIndividual.cuh"
#include "misc_functions.cuh"

#include <cstdio>
#include <cassert>

__device__ void cuIndividual::search_patterns() {
    // One block per individual
    uint idx = threadIdx.x;
    uint rr_width = blockDim.x;

    extern __shared__ char local_genome_PROM[];

    // First, we copy the genome in the shared memory
    genome->getBits(local_genome_PROM);

    __syncthreads();
    for (uint position = idx; position < size; position += rr_width) {
        const char *genome_at_pos = local_genome_PROM + position;
        promoters[position]   = is_promoter(genome_at_pos);
        terminators[position] = is_terminator(genome_at_pos);
        prot_start[position]  = is_prot_start(genome_at_pos);
    }

}


__device__ void cuIndividual::sparse_meta() {
    // Before: One block per individual
    // uint idx = threadIdx.x;
    
    // New version
    extern __shared__ uint nbPerThreads[];
    prepare_rnas(nbPerThreads, tmp_sparse_collection);
    sparse(size, terminators, tmp_sparse_collection, nbPerThreads, &nb_terminator);
    __syncthreads();
    sparse(size, prot_start, tmp_sparse_collection, nbPerThreads, &nb_prot_start);
}
__device__ void cuIndividual::transcription() {
    // One block per individual
    uint idx = threadIdx.x;
    uint rr_width = blockDim.x;

    for (uint rna_idx = idx; rna_idx < nb_rnas; rna_idx += rr_width) {
        compute_rna(rna_idx);
    }
}

__device__ void cuIndividual::find_gene_per_RNA() {
    // One block per individual
    uint idx = threadIdx.x;
    uint rr_width = blockDim.x;

    for (uint rna_idx = idx; rna_idx < nb_rnas; rna_idx += rr_width) {
        prepare_gene(rna_idx);
    }
}

__device__ void cuIndividual::translation() {
    // One block per individual
    uint idx = threadIdx.x;
    uint rr_width = blockDim.x;

    extern __shared__ char local_genome_PROM[];

    // First, we copy the genome in the shared memory
    genome->getBits(local_genome_PROM);

    for (uint gene_idx = idx; gene_idx < nb_gene; gene_idx += rr_width) {
        translate_gene(gene_idx, local_genome_PROM);
    }
}

__device__ void cuIndividual::clean_metadata() {
    // One thread working alone
    nb_terminator = 0;
    nb_prot_start = 0;
    nb_gene = 0;

    fitness = 0.0;
    
    // Because we change the memory management
    assert(nb_rnas < max_list_size);

    // for (int i = 0; i < nb_rnas; ++i) {
    //     // delete[] list_rnas[i].list_gene.ary;
    //     // list_rnas[i].list_gene.ary = nullptr;
    // }
    nb_rnas = 0;

    // delete[] list_gene.ary;
    // list_gene.ary = nullptr;
    // delete[] list_protein.ary;
    // list_protein.ary = nullptr;
}

__device__ void cuIndividual::prepare_rnas(uint * nbPerThreads, uint* tmp_sparse_collection) {
    // new    
    if (threadIdx.x == 0)
    {
        nbPerThreads[0] = 0;
    }

    nbPerThreads[threadIdx.x+1] = 0;
    uint range = (size / blockDim.x)+1;
    uint begin = threadIdx.x * range;
    uint nb_found = 0;
    uint read_position;
    uint8_t read_value;

    // for (read_position = threadIdx.x; read_position < size; read_position += blockDim.x) {
    //     read_value = promoters[read_position];
    //     if (read_value <= PROM_MAX_DIFF) {
    //         nbPerThreads[threadIdx.x+1]++;
    //         tmp_sparse_collection[threadIdx.x+nb_found*blockDim.x] = read_position;
    //         nb_found++;
    //     }
    // }

    for (read_position = begin; read_position < begin + range; ++read_position) {
        if (read_position < size)
        {
            read_value = promoters[read_position];
            if (read_value <= PROM_MAX_DIFF) {
                nbPerThreads[threadIdx.x+1]++;
                tmp_sparse_collection[begin+nb_found] = read_position;
                nb_found++;
            }
        }
        
    }
    __syncthreads();

    int insert_before = 0;
    uint i;
    for (i = 1; i < threadIdx.x+1; i++)
    {
        insert_before += nbPerThreads[i];
    }

    // for (i = 0; i < nb_found; i++)
    // {
    //     read_position = tmp_sparse_collection[threadIdx.x + i * blockDim.x];
    //     read_value = promoters[read_position];
    //     auto &rna = list_rnas[insert_before];
    //     rna.errors = read_value;
    //     rna.start_transcription = read_position + PROM_SIZE;
    //     if (rna.start_transcription >= size)
    //         rna.start_transcription -= size;
    //     insert_before++; 
    // }

    for (uint read_position = threadIdx.x * range; read_position < threadIdx.x * range + range; ++read_position) {
        if (read_position < size)
        {
            uint8_t read_value = promoters[read_position];
            if (read_value <= PROM_MAX_DIFF) {
                auto &rna = list_rnas[insert_before];
                rna.errors = read_value;
                rna.start_transcription = read_position + PROM_SIZE;
                if (rna.start_transcription >= size)
                    rna.start_transcription -= size;

                assert(insert_before < max_list_size);
                rna.list_gene.max_size = max_list_size;
                rna.list_gene.ary = &(list_gene_4_rna[insert_before*max_list_size]);
                insert_before++; 
            }
        }
    }

    atomicAdd(&nbPerThreads[0], nbPerThreads[threadIdx.x+1]);
    __syncthreads();

    if (threadIdx.x == 0)
    {
        nb_rnas = nbPerThreads[0];
    }
    // old

    // One thread working alone
    // int insert_position = 0;

    // for (uint read_position = 0; read_position < size; ++read_position) {
    //     uint8_t read_value = promoters[read_position];
    //     if (read_value <= PROM_MAX_DIFF) {
    //         auto &rna = list_rnas[insert_position];
    //         rna.errors = read_value;
    //         rna.start_transcription = read_position + PROM_SIZE;
    //         if (rna.start_transcription >= size)
    //             rna.start_transcription -= size;
    //         insert_position++;
    //     }
    // }

    // nb_rnas = insert_position;
}

__device__ void cuIndividual::compute_rna(uint rna_idx) const {
    // One thread
    auto &rna = list_rnas[rna_idx];
    uint start_transcript = rna.start_transcription;

    // get end of transcription
    // find the smallest element greater than start
    uint idx_term = find_smallest_greater(start_transcript, terminators, nb_terminator);
    uint term_position = terminators[idx_term];
    uint transcript_length = get_distance(start_transcript, term_position) + TERM_SIZE;
    rna.transcription_length = transcript_length;
    if (transcript_length < DO_TRANSLATION_LOOP) {
        rna.errors += 0b1000u;
    }
}

__device__ void cuIndividual::prepare_gene(uint rna_idx) const {
    // One thread
    auto &rna = list_rnas[rna_idx];
    rna.nb_gene = 0;
    // rna.list_gene.ary = nullptr;
    if (rna.errors > PROM_MAX_DIFF) {
        return;
    }

    uint nb_ps = nb_prot_start;
    if (not nb_ps) {
        return;
    }
    uint *list_ps = prot_start;
    uint local_nb_gene = 0;

    // Correctly setup the research
    uint max_distance = rna.transcription_length - DO_TRANSLATION_LOOP;
    uint first_next_ps = find_smallest_greater(rna.start_transcription, list_ps, nb_ps);
    uint ps_idx = first_next_ps;

    uint distance = get_distance(rna.start_transcription, list_ps[ps_idx]);
    while (distance <= max_distance) {
        local_nb_gene++;
        uint prev_idx = ps_idx++;
        if (ps_idx == nb_ps)
            ps_idx = 0;
        distance += get_distance_ori(list_ps[prev_idx], list_ps[ps_idx],2);
    }
    // all potential genes are counted
    // Let us put their position in a list

    rna.nb_gene = local_nb_gene;
    if (local_nb_gene > 0) {
        pseudo_new_my_array_gene(&(rna.list_gene), local_nb_gene);
        // rna.list_gene.ary = new cuGene[local_nb_gene]{};
        assert(rna.list_gene.ary != nullptr);
    }
    for (int i = 0; i < rna.nb_gene; ++i) {
        uint start = list_ps[first_next_ps] + SD_TO_START;
        if (start >= size) {
            start -= size;
        }

        rna.list_gene.ary[i].start = start;
        rna.list_gene.ary[i].concentration = PROM_MAX_DIFF + 1 - rna.errors;
        rna.list_gene.ary[i].length_limit = get_distance(rna.start_transcription, start);
        if (++first_next_ps >= nb_ps) {
            first_next_ps = 0;
        }
    }
}

__device__ void cuIndividual::gather_genes() {
    // One thread working alone
    nb_gene = 0;
    for (int idx_rna = 0; idx_rna < nb_rnas; ++idx_rna) {
        nb_gene += list_rnas[idx_rna].nb_gene;
    }
    if (nb_gene > 0) {
        // list_gene.ary = new cuGene[nb_gene]{};
        // new 
        pseudo_new_my_array_gene(&list_gene, nb_gene);

        // list_protein.ary = new cuProtein[nb_gene]{};
        pseudo_new_my_array_prot(&list_protein, nb_gene);

        assert(list_gene.ary != nullptr);
        assert(list_protein.ary != nullptr);
    }
    uint insert_idx = 0;

    for (int idx_rna = 0; idx_rna < nb_rnas; ++idx_rna) {
        const auto &rna = list_rnas[idx_rna];
        for (int i = 0; i < rna.nb_gene; ++i) {
            assert(insert_idx < nb_gene);
            list_gene.ary[insert_idx] = rna.list_gene.ary[i];
            // limit is difference between transcription_length and distance start_rna -> start_gen (computed in `prepare_gene`)
            list_gene.ary[insert_idx].length_limit = rna.transcription_length - list_gene.ary[insert_idx].length_limit;
            insert_idx++;
        }
    }
}

__device__ void cuIndividual::translate_gene(uint gene_idx, char * local_genome_PROM) const {
    // One thread
    const auto &gene = list_gene.ary[gene_idx];

    uint max_distance = gene.length_limit;

    auto next = [this](uint &it) -> void {
        it += 3;
        if (it >= this->size)
            it -= this->size;
    };

    uint it = gene.start;
    uint distance = 0;
    auto &new_protein = list_protein.ary[gene_idx];

    while (true) {
        uint8_t codon = translate_to_codon(local_genome_PROM + it);
        if (codon == CODON_STOP)
            break;
        distance += CODON_SIZE;
        if (distance > max_distance) {
            distance = false;
            break;
        }

        new_protein.add_codon(codon);
        next(it);
    }

    if (distance) {
        // Gene has been translated into a protein
        new_protein.concentration = (float) gene.concentration / (float) (PROM_MAX_DIFF + 1);
        new_protein.normalize();
    } else
        new_protein.concentration = 0.0;
}

__device__ void add_protein_to_phenotype(const cuProtein& protein,
                                         double* phenotype) {
    // One Thread
    double left_absc = protein.mean - protein.width;
    // mid point abscissa of triangle
    double mid_absc = protein.mean;
    // right point abscissa of triangle
    double right_absc = protein.mean + protein.width;

    // Interface between continuous world (up) and discrete world (down)
    int i_left_absc  = (int) (left_absc  * FUZZY_SAMPLING);
    int i_mid_absc   = (int) (mid_absc   * FUZZY_SAMPLING);
    int i_right_absc = (int) (right_absc * FUZZY_SAMPLING);

    // active contribution is positive and inhib is negative
    double height = protein.height * (double)protein.concentration;

    // Compute the first equation of the triangle
    double slope = height / (double)(i_mid_absc - i_left_absc);
    double y_intercept = -(double)i_left_absc * slope;

    // Updating value between left_absc and mid_absc
    for (int i = i_left_absc; i < i_mid_absc; i++) {
        if (i >= 0)
            atomicAdd_block(phenotype + i,
                            slope * (double)i + y_intercept);
    }

    // Compute the second equation of the triangle
    slope = height / (double)(i_mid_absc - i_right_absc);
    y_intercept = -(double)i_right_absc * slope;
    // Updating value between mid_absc and right_absc
    for (int i = i_mid_absc; i < i_right_absc; i++) {
        if (i < FUZZY_SAMPLING)
            atomicAdd_block(phenotype + i,
                            slope * (double)i + y_intercept);
    }
}

__device__ void cuIndividual::compute_phenotype() {
    // One block
    auto idx = threadIdx.x;
    auto rr_width = blockDim.x;

    __shared__ double phenotype_activ_inhib[2][FUZZY_SAMPLING]; // { activ_phenotype, inhib_phenotype }
    // Initialize activation and inhibition to zero
    for (int i = idx; i < FUZZY_SAMPLING; i += rr_width) {
        phenotype_activ_inhib[0][i] = 0.0;
        phenotype_activ_inhib[1][i] = 0.0;
    }
    __syncthreads();

    for (int protein_idx = idx; protein_idx < nb_gene; protein_idx += rr_width) {
        auto& protein = list_protein.ary[protein_idx];
        if (protein.is_functional()) {
            int8_t activ_inhib = protein.height > 0 ? 0 : 1;
            add_protein_to_phenotype(protein, phenotype_activ_inhib[activ_inhib]);
        }
    }
    __syncthreads();

    for (int fuzzy_idx = idx; fuzzy_idx < FUZZY_SAMPLING; fuzzy_idx += rr_width) {
        phenotype_activ_inhib[0][fuzzy_idx] = min(phenotype_activ_inhib[0][fuzzy_idx],  1.0);
        // (inhib_phenotype < 0)
        phenotype_activ_inhib[1][fuzzy_idx] = max(phenotype_activ_inhib[1][fuzzy_idx], -1.0);
        // phenotype = active_phenotype + inhib_phenotype
        phenotype[fuzzy_idx] = phenotype_activ_inhib[0][fuzzy_idx] + phenotype_activ_inhib[1][fuzzy_idx];
        phenotype[fuzzy_idx] = clamp(phenotype[fuzzy_idx], 0.0, 1.0);
    }
}

__device__ void cuIndividual::compute_fitness(const double* target) {
    // One block
    auto idx = threadIdx.x;
    auto rr_width = blockDim.x;

    __shared__ double delta[FUZZY_SAMPLING];

    for (int fuzzy_idx = idx; fuzzy_idx < FUZZY_SAMPLING; fuzzy_idx += rr_width) {
        delta[fuzzy_idx] = fabs(phenotype[fuzzy_idx] - target[fuzzy_idx]);
        delta[fuzzy_idx] /= (double)FUZZY_SAMPLING;
    }

    __syncthreads();
    if (idx == 0) {
        double meta_error = 0.0;
        for (int i = 0; i < FUZZY_SAMPLING; ++i) {
            meta_error += delta[i];
        }
        fitness = exp(-SELECTION_PRESSURE * meta_error);
    }
}

// ************ PRINTING
#define PRINT_HEADER(STRING)  printf("<%s>\n", STRING)
#define PRINT_CLOSING(STRING) printf("</%s>\n", STRING)

__device__ void cuIndividual::print_metadata_summary() const {
    __syncthreads();
    if (threadIdx.x == 0) {
        PRINT_HEADER("METADATA_SUMMARY");
        printf("proms: %d, terms: %d, prots: %d\n", nb_rnas, nb_terminator, nb_prot_start);
        PRINT_CLOSING("METADATA_SUMMARY");
    }
    __syncthreads();
}

__device__ void cuIndividual::print_rnas() const {
    __syncthreads();
    if (threadIdx.x == 0) {
        PRINT_HEADER("ARNS");
        uint nb_coding_rna = 0;
        for (int i = 0; i < nb_rnas; ++i) {
            const auto &rna = list_rnas[i];
            if (rna.errors <= PROM_MAX_DIFF) {
                nb_coding_rna++;
                uint start = rna.start_transcription;
                uint end = (start + rna.transcription_length) % size;
                printf("%d -> %d | %d\n", start, end, rna.errors);
            }
        }
        printf("Non coding: \n");
        for (int i = 0; i < nb_rnas; ++i) {
            const auto &rna = list_rnas[i];
            if (rna.errors > PROM_MAX_DIFF) {
                uint start = rna.start_transcription;
                uint end = (start + rna.transcription_length) % size;
                printf("%d -> %d | %d\n", start, end, (rna.errors - 0b1000u));
            }
        }

        printf("\nnumber of terminated rna: %u\n", nb_coding_rna);
        PRINT_CLOSING("ARNS");
    }
    __syncthreads();
}

__device__ void cuIndividual::print_gathered_genes() const {
    __syncthreads();
    if (threadIdx.x == 0) {
        PRINT_HEADER("GENES");
        uint local_nb_gene = 0;
        for (int i = 0; i < local_nb_gene; ++i) {
            local_nb_gene++;
            printf("\t%d: concentration: %d, limit: %d\n",
                   list_gene.ary[i].start, list_gene.ary[i].concentration, list_gene.ary[i].length_limit);
        }

        printf("\nnumber of potential gene: %u\n", local_nb_gene);
        PRINT_CLOSING("GENES");
    }
    __syncthreads();
}

__device__ void cuIndividual::print_proteins() const {
    __syncthreads();
    if (threadIdx.x == 0) {
        PRINT_HEADER("PROTEINS");
        uint nb_prot = 0;
        for (int i = 0; i < nb_gene; ++i) {
            const auto &prot = list_protein.ary[i];
            nb_prot++;
            printf("%d: %d %f %f %f %f\n",
                   list_gene.ary[i].start, prot.is_functional(),
                   prot.concentration, prot.width, prot.mean, prot.height);

        }

        printf("\nnumber of proteins: %u\n", nb_prot);
        PRINT_CLOSING("PROTEINS");
    }
    __syncthreads();
}

__device__ void cuIndividual::print_phenotype() const {
    __syncthreads();
    if (threadIdx.x == 0) {
        PRINT_HEADER("PHENOTYPE");
        for (int i = 0; i < FUZZY_SAMPLING; ++i) {
            if (phenotype[i] == 0.0){
                printf("0|");
            } else {
                printf("%f|", phenotype[i]);
            }
        }
        printf("\n");
        PRINT_CLOSING("PHENOTYPE");
    }
    __syncthreads();
}

__device__
void pseudo_new_my_array_gene(my_array<cuGene> * my_arr, size_t new_size)
{
    // if not, must increase the value list_max_size in th efile cuExpManager
    assert(new_size <= my_arr->max_size);
}

__device__
void pseudo_new_my_array_prot(my_array<cuProtein> * my_arr, size_t new_size)
{
    // if not, must increase the value list_max_size in th efile cuExpManager
    assert(new_size <= my_arr->max_size);

    // Then must reinit all the prots that we are going to use:

    for (int id = 0; id < my_arr->max_size; id++)
    {
        cuProtein * prot = &(my_arr->ary[id]);
        for (int i = 0; i < 3; i++)
        {
            prot->wmh[i] = 0;
            prot->wmh_nb[i] = 0;
        }

        prot->concentration = 0;
        prot->width = 0;
        prot->height = 0;
        prot->mean = 0;
    }
}