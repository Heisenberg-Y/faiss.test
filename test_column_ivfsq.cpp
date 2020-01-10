/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <unistd.h>

#include <iostream>
#include <Column.h>

#include "IndexIVF.h"
#include "IndexFlat.h"
#include "index_io.h"

#include "IndexPQ.h"
#include "index_factory.h"

#include "clone_index.h"
#include "IndexIVFSpectralHash.h"
#include "IndexSQHybrid.h"
#include "IndexBinaryHNSW.h"
#include "utils/distances.h"
#include "faiss/Column.h"

#include <faiss/gpu/GpuCloner.h>
#include <faiss/gpu/StandardGpuResources.h>

#define verbose 1

using namespace faiss;

int d = 10;                            // dimension
long nb = 100;
int nq = 2;                        // nb of queries
const char* index_description = "IVF2,SQ8";
int nprobe = 1;
int k = 5;
long size = d * nb;

const char* filename = "7.index";
const char *filecolumn = "7.column.index";
float *xb = nullptr;
int64_t *xids;

void
CpuExecutor(
        faiss::IndexComposition* index_composition,
        int nq,
        int nprobe,
        int k,
        float* xq,
        faiss::Index *cpu_index,
        ColumnInvertedSets *cis) {
    printf("CPU: \n");
    long *I = new long[k * nq];
    float *D = new float[k * nq];

    auto ivf_index =
            dynamic_cast<faiss::IndexIVF*>(cpu_index);
    ivf_index->nprobe = nprobe;
    double t4, t5;

    printf("--------------------------------------\n");

    ivf_index->make_direct_map(true);

    for (int i = 0; i < 1; ++i) {
        t4 = getmillisecs();
        ivf_index->search_with_ids(nq, xids, k, D, I);
        t5 = getmillisecs();
        printf("CPU execution time: %0.2f\n", t5 - t4);
    }

//    printf("--------------------------------------\n");
//
//    for (int i = 0; i < 5; ++i) {
//        t4 = getmillisecs();
//        ivf_index->search_conditional(nq, xq, k, D, I, cis);
//        t5 = getmillisecs();
//        printf("CPU execution time: %0.2f\n", t5 - t4);
//    }


#if verbose
    printf("\n");
    for (int m = 0; m < nq; ++m) {
        printf("query %d:\n", m);
        for (int j = 0; j < k; ++j) {
            long idx = I[m * k + j];
            printf("%2d. id = %2ld :", j, idx);
            for (int i = 0; i < d; ++i) {
                printf("%f ", xb[idx * d + i]);
            }
            printf(" distance = %f", D[m * k + j]);
            float true_distance = 0;
            for (int i = 0; i < d; ++i) {
                true_distance += (xq[m * d + i] - xb[idx * d + i]) * (xq[m * d + i] - xb[idx * d + i]);
            }
            printf("\n");
//            printf(" true = %f, true^2 = %f\n", sqrt(true_distance), true_distance);
        }
        printf("\n");
    }
#endif

    printf("CPU execution time: %0.2f\n", t5 - t4);




    delete [] I;
    delete [] D;
}

int main() {
    faiss::distance_compute_blas_threshold = 800;

    faiss::Index *cpu_index = nullptr;
    faiss::IndexIVF* cpu_ivf_index = nullptr;
    faiss::gpu::StandardGpuResources res;

    xb = new float[size];
    memset(xb, 0, size * sizeof(float));
    printf("size: %ld\n", (size * sizeof(float)) );

    for(long i = 0; i < nb; i++) {
        for(long j = 0; j < d; j++) {
            float rand = drand48();
            xb[d * i + j] = rand;
        }
    }

    float *xq = new float[d * nq];
    for(int i = 0; i < nq; i++) {
        for(int j = 0; j < d; j++) {
            xq[d * i + j] = drand48();
        }
    }

    xids = new int64_t[nb];
    for(int64_t i = 0; i < nb; i++){
        xids[i] = i;
    }

    if((access(filename,F_OK))==-1) {
        faiss::Index *ori_index = faiss::index_factory(d, index_description, faiss::METRIC_L2);

        auto device_index = faiss::gpu::index_cpu_to_gpu(&res, 0, ori_index);

        assert(!device_index->is_trained);
        device_index->train(nb, xb);
        assert(device_index->is_trained);
        device_index->add_with_ids(nb, xb, xids);  // add vectors to the index

        printf("is_trained = %s\n", ori_index->is_trained ? "true" : "false");
        printf("ntotal = %ld\n", ori_index->ntotal);

        cpu_index = faiss::gpu::index_gpu_to_cpu ((device_index));
//        faiss::write_index(cpu_index, filename);
//        printf("index.index is stored successfully.\n");

    } else {
        cpu_index = faiss::read_index(filename);
    }

#if verbose
//    printf("\nbase dataset: \n");
//    for (int l = 0; l < nb; ++l) {
//        printf("%2d: ", l);
//        for (int i = 0; i < d; ++i) {
//            printf("%f ", xb[l * d + i]);
//        }
//        printf("\n");
//    }
//    printf("\n");


    printf("\nquery dataset: \n");
    for (int l = 0; l < nq; ++l) {
        printf("%2d: ", l);
        for (int i = 0; i < d; ++i) {
            printf("%f ", xq[l * d + i]);
        }
        printf("\n");
    }
    printf("\n");
#endif

    cpu_ivf_index = dynamic_cast<faiss::IndexIVF*>(cpu_index);
//    if(cpu_ivf_index != nullptr) {
//        cpu_ivf_index->to_readonly();
//    }
    auto tmp = dynamic_cast<faiss::ReadOnlyArrayInvertedLists *>(cpu_ivf_index->invlists);
    if(tmp != nullptr){
        printf("success\n");
    }

    ColumnArrayInvertedSets cis((ArrayInvertedLists *)cpu_ivf_index->invlists, ColumnType::INT64, ColumnType::INT32, ColumnType::INT16, ColumnType::INT8, ColumnType::FLOAT, ColumnType::DOUBLE);

    faiss::IndexComposition index_composition0;
    index_composition0.index = cpu_index;
    index_composition0.quantizer = nullptr;
    index_composition0.mode = 1; // only quantizer

    CpuExecutor(&index_composition0, nq, nprobe, k, xq, cpu_index, (ColumnInvertedSets *)&cis);

    delete [] xq;
    delete [] xb;
    return 0;
}

