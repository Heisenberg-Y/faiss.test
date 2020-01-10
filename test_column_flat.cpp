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
#include <memory.h>
#include <cmath>

#include <faiss/utils/utils.h>
#include <faiss/IndexFlat.h>
#include <faiss/index_io.h>
#include <faiss/Column.h>
#include <faiss/utils/distances.h>
#include <faiss/MetaIndexes.h>

#define verbose 1

using namespace faiss;

int d = 8;                            // dimension
int nb = 20;                       // database size
int nq = 2;                        // nb of queries
int k = 5;
long size = d * nb;

const char* filename = "5.index";
const char *filecolumn = "5.column.index";
float *xb = nullptr;
int64_t *xids;

void
CpuExecutor(
        int nq,
        int k,
        float* xq,
        faiss::Index *cpu_index,
        ColumnSet *cs) {
    printf("CPU: \n");
    long *I = new long[k * nq];
    float *D = new float[k * nq];

    double t4, t5;

    printf("--------------------------------------\n");
    faiss::distance_compute_blas_threshold = 800;

    for (int i = 0; i < 1; ++i) {
        t4 = getmillisecs();
        cpu_index->search_conditional(nq, xq, k, D, I, (void *)(cs));
        t5 = getmillisecs();
        printf("CPU execution time: %0.2f\n", t5 - t4);
    }

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
            printf(" true = %f, true^2 = %f\n", sqrt(true_distance), true_distance);
        }
        printf("\n");
    }
#endif

    for (int i = 0; i < 1; ++i) {
        t4 = getmillisecs();
        cpu_index->search(nq, xq, k, D, I);
        t5 = getmillisecs();
        printf("CPU execution time: %0.2f\n", t5 - t4);
    }

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
            printf(" true = %f, true^2 = %f\n", sqrt(true_distance), true_distance);
        }
        printf("\n");
    }
#endif

    delete [] I;
    delete [] D;
}

int main() {
    faiss::ColumnSet *cs;
    if((access(filecolumn,F_OK))==-1){
        cs = new ColumnSet(nb, ColumnType::INT64, ColumnType::INT32, ColumnType::INT16, ColumnType::INT8, ColumnType::FLOAT, ColumnType::DOUBLE);
//        write_column_set(cs, filecolumn);
    } else{
        cs = read_column_set(filecolumn);
    }

//    cs->getColumn<bool >(0)->setValue(42, false);

    faiss::Index *cpu_index = nullptr;

    xb = new float[d * nb];
    memset(xb, 0, size * sizeof(float));
    printf("size: %ld\n", (size * sizeof(float)) );

    for(int i = 0; i < nb; i++) {
        for(int j = 0; j < d; j++)
            xb[d * i + j] = drand48();
    }

    float *xq = new float[d * nq];
    for(int i = 0; i < nq; i++) {
        for(int j = 0; j < d; j++)
            xq[d * i + j] = drand48();
    }

    xids = new int64_t[nb];
    for(int64_t i = 0; i < nb; i++){
        xids[i] = i;
    }

    if((access(filename, F_OK)) == -1){
        faiss::Index *index_flat = new faiss::IndexFlatL2(d);
        auto ori_index = new faiss::IndexIDMap(reinterpret_cast<Index*>(index_flat));

        assert(ori_index->is_trained);
        ori_index->add_with_ids(nb, xb, xids);  // add vectors to the index

        printf("is_trained = %s\n", ori_index->is_trained ? "true" : "false");
        printf("ntotal = %ld\n", ori_index->ntotal);

        cpu_index = ori_index;
//        faiss::write_index(cpu_index, filename);
        printf("index.index is stored successfully.\n");
    } else{
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

    int64_t ids[2] = {6, 9};
    cs->deleteByIds(2, ids, dynamic_cast<IndexIDMap *> (cpu_index));

    CpuExecutor(nq, k, xq, cpu_index, cs);

    delete [] xb;
    delete [] xq;

    return 0;
}
