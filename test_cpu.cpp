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

#include "IndexIVF.h"
#include "IndexFlat.h"
#include "index_io.h"

#include "impl/FaissAssert.h"
#include "impl/AuxIndexStructures.h"

#include "IndexFlat.h"
#include "VectorTransform.h"
#include "IndexLSH.h"
#include "IndexPQ.h"
#include "index_factory.h"

#include "IndexIVFPQ.h"
#include "clone_index.h"
#include "IndexIVFFlat.h"
#include "IndexIVFSpectralHash.h"
#include "MetaIndexes.h"
#include "IndexSQHybrid.h"
#include "IndexHNSW.h"
#include "OnDiskInvertedLists.h"
#include "IndexBinaryFlat.h"
#include "IndexBinaryFromFloat.h"
#include "IndexBinaryHNSW.h"
#include "IndexBinaryIVF.h"
#include "utils/distances.h"

#define verbose 1

using namespace faiss;

int d = 10;                            // dimension
int nq = 2;                        // nb of queries
int nprobe = 2;
float *xb;
long nb = 100;
long size = d * nb;
const char* filename = "3.index";
int k = 5;
const char* index_description = "IVF4,Flat";


void
CpuExecutor(
        faiss::IndexComposition* index_composition,
        int nq,
        int nprobe,
        int k,
        float* xq,
        faiss::Index *cpu_index) {
    printf("CPU: \n");
    long *I = new long[k * nq];
    float *D = new float[k * nq];

    double t4 = getmillisecs();
    faiss::IndexIVF* ivf_index =
            dynamic_cast<faiss::IndexIVF*>(cpu_index);
    ivf_index->nprobe = nprobe;

//    faiss::gpu::GpuIndexFlat* is_gpu_flat_index = dynamic_cast<faiss::gpu::GpuIndexFlat*>(ivf_index->quantizer);
//    if(is_gpu_flat_index == nullptr) {
//        delete ivf_index->quantizer;
//        ivf_index->quantizer = index_composition->quantizer;
//    }

    cpu_index->search(nq, xq, k, D, I);


#if verbose
    printf("\n");
    for (int m = 0; m < nq; ++m) {
        printf("query %d:\n", m);
        for (int j = 0; j < k; ++j) {
            long idx = I[m * k + j];
            printf("%2d. id = %ld :", j, idx);
            for (int i = 0; i < d; ++i) {
                printf("%f ", xb[idx * d + i]);
            }
            printf(" distance = %f", D[m * k + j]);
            float true_distance = 0;
            for (int i = 0; i < d; ++i) {
                true_distance += (xq[m * d + i] - xb[idx * d + i]) * (xq[m * d + i] - xb[idx * d + i]);
            }
            printf(" true = %f, true^2 = %f\n", sqrt(true_distance), true_distance);
        }
        printf("\n");
    }
#endif

    double t5 = getmillisecs();
    printf("CPU execution time: %0.2f\n", t5 - t4);




    delete [] I;
    delete [] D;
}

int main() {
    faiss::distance_compute_blas_threshold = 800;


    faiss::Index *cpu_index = nullptr;
    faiss::IndexIVF* cpu_ivf_index = nullptr;

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

    if((access(filename,F_OK))==-1) {
        faiss::Index *ori_index = faiss::index_factory(d, index_description, faiss::METRIC_L2);

        assert(!ori_index->is_trained);
        ori_index->train(nb, xb);
        assert(ori_index->is_trained);
        ori_index->add(nb, xb);  // add vectors to the index

        printf("is_trained = %s\n", ori_index->is_trained ? "true" : "false");
        printf("ntotal = %ld\n", ori_index->ntotal);

        cpu_index = ori_index;
        faiss::write_index(cpu_index, filename);
        printf("index.index is stored successfully.\n");

    } else {
        cpu_index = faiss::read_index(filename);
    }

#if verbose
    printf("\nbase dataset: \n");
    for (int l = 0; l < nb; ++l) {
        printf("%2d: ", l);
        for (int i = 0; i < d; ++i) {
            printf("%f ", xb[l * d + i]);
        }
        printf("\n");
    }
    printf("\n");


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

//    cpu_ivf_index = dynamic_cast<faiss::IndexIVF*>(cpu_index);
//    if(cpu_ivf_index != nullptr) {
//        cpu_ivf_index->to_readonly();
//    }

//    auto tmp = dynamic_cast<faiss::ReadOnlyArrayInvertedLists *>(cpu_ivf_index->invlists);
//    if(tmp != nullptr){
//        printf("success\n");
//    }
//    faiss::gpu::GpuClonerOptions option0;
//
//    option0.allInGpu = true;

    faiss::IndexComposition index_composition0;
    index_composition0.index = cpu_index;
    index_composition0.quantizer = nullptr;
    index_composition0.mode = 1; // only quantizer

//    // Copy quantizer to GPU 0
//    auto index1 = faiss::gpu::index_cpu_to_gpu(&res, 0, &index_composition0, &option0);
//    delete index1;
//
//    index_composition0.mode = 2; // only data
//
//    index1 = faiss::gpu::index_cpu_to_gpu(&res, 0, &index_composition0, &option0);
//    delete index1;
//
//    for(long i = 0; i < 1; ++ i) {
//        std::shared_ptr<faiss::Index> gpu_index_ptr00;
//
//        GpuLoad(&res, 0, &option0, &index_composition0, std::ref(gpu_index_ptr00));
//
//        GpuExecutor(gpu_index_ptr00, res, 0, &option0, &index_composition0, nq, nprobe, k, xq);
//    }

    CpuExecutor(&index_composition0, nq, nprobe, k, xq, cpu_index);

    delete [] xq;
    delete [] xb;
    return 0;
}

