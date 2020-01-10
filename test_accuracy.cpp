/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdio>
#include <cassert>
#include <unistd.h>
#include <hdf5.h>
#include <set>

#include <iostream>
#include <Column.h>

#include "IndexIVF.h"
#include "IndexFlat.h"

#include "index_factory.h"

#include "IndexIVFSpectralHash.h"
#include "IndexSQHybrid.h"
#include "IndexBinaryHNSW.h"
#include "utils/distances.h"
#include "faiss/Column.h"

#include <faiss/gpu/GpuCloner.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/index_io.h>

#define verbose 1

using namespace faiss;

size_t d;                            // dimension
size_t nb;
size_t nq;                        // nb of queries
const char* index_description = "IVF16384,Flat";
int nprobe = 32;
size_t k = 10;
size_t k2;

const char *filename = "sift-128-euclidean.hdf5";
float *xb = nullptr;
float *xq = nullptr;
faiss::Index::idx_t *gt = nullptr;
int64_t  *ids;

size_t
GetResultHitCount(const faiss::Index::idx_t* ground_index, const faiss::Index::idx_t* ids, size_t ground_k, size_t result_k) {
    assert(ground_k >= result_k);
    size_t hit = 0;
    for (size_t i = 0; i < nq; i++) {
        for (size_t j_c = 0; j_c < result_k; j_c++) {
            int r_c = ids[i * result_k + j_c];
            for (size_t j_g = 0; j_g < result_k; j_g++) {
                if (ground_index[i * ground_k + j_g] == r_c) {
                    hit++;
                    continue;
                }
            }
        }
    }
    return hit;
}

void
CpuExecutor(
        faiss::IndexComposition* index_composition,
        faiss::Index *cpu_index,
        faiss::Index *index_flat,
        ColumnInvertedSets *cis,
        ColumnSet *cs) {
    auto Is = new long[k * nq];
    auto I = new long[k * nq];
    auto D = new float[k * nq];

    auto ivf_index =
            dynamic_cast<faiss::IndexIVF*>(cpu_index);
    ivf_index->nprobe = nprobe;

    double t4, t5;

    printf("\n======================================\n");

    for (int i = 0; i < 1; ++i) {
        t4 = getmillisecs();
        index_flat->search_conditional(nq, xq, k, D, Is, cs);
        t5 = getmillisecs();
        printf("flat execution time: %0.2f\n", t5 - t4);
    }

    size_t standard_count = GetResultHitCount(gt, Is, k2, k);
    size_t ground_count = k * nq;
    size_t hit_count;
    printf("standard / ground:  %lf\n", double_t (standard_count) / (double)(ground_count));

    printf("--------------------------------------\n");

    for (int i = 0; i < 1; ++i) {
        t4 = getmillisecs();
        ivf_index->search_conditional(nq, xq, k, D, I, cis);
        t5 = getmillisecs();
        printf("ivf execution time: %0.2f\n", t5 - t4);
    }

    hit_count = GetResultHitCount(gt, I, k2, k);
    printf("hit / ground:  %lf\n", double_t (hit_count) / (double)(k * nq));
    hit_count = GetResultHitCount(Is, I, k, k);
    printf("hit / standard:  %lf\n", double_t (hit_count) / (double)(k * nq));

    printf("======================================\n\n");

//    for (int m = 0; m < 5; ++m) {
//        printf("query %d:\n", m);
//        for (int j = 0; j < k; ++j) {
//            long idx = I[m * k + j];
//            long gt_id = gt[m * k2 + j];
//            long flat_id = Is[m * k + j];
//            printf("%2d. id = %2ld gt_id = %ld flat_id = %ld", j, idx, gt_id, flat_id);
//            printf(" distance = %f", D[m * k + j]);
//            float true_distance = 0;
//            for (int i = 0; i < d; ++i) {
//                true_distance += (xq[m * d + i] - xb[idx * d + i]) * (xq[m * d + i] - xb[idx * d + i]);
//            }
//            printf(" true = %f, true^2 = %f\n", sqrt(true_distance), true_distance);
//        }
//        printf("\n");
//    }

    delete [] I;
    delete [] Is;
    delete [] D;
}

void*
hdf5_read(const std::string& file_name, const std::string& dataset_name, H5T_class_t dataset_class, size_t& d_out,
          size_t& n_out) {
    hid_t file, dataset, datatype, dataspace, memspace;
    H5T_class_t t_class;   /* data type class */
    hsize_t dimsm[3];      /* memory space dimensions */
    hsize_t dims_out[2];   /* dataset dimensions */
    hsize_t count[2];      /* size of the hyperslab in the file */
    hsize_t offset[2];     /* hyperslab offset in the file */
    hsize_t count_out[3];  /* size of the hyperslab in memory */
    hsize_t offset_out[3]; /* hyperslab offset in memory */
    void* data_out = nullptr;        /* output buffer */

    /* Open the file and the dataset. */
    file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    dataset = H5Dopen2(file, dataset_name.c_str(), H5P_DEFAULT);

    /* Get datatype and dataspace handles and then query
     * dataset class, order, size, rank and dimensions. */
    datatype = H5Dget_type(dataset); /* datatype handle */
    t_class = H5Tget_class(datatype);
    assert(t_class == dataset_class || !"Illegal dataset class type");

    dataspace = H5Dget_space(dataset); /* dataspace handle */
    H5Sget_simple_extent_dims(dataspace, dims_out, nullptr);
    n_out = dims_out[0];
    d_out = dims_out[1];

    /* Define hyperslab in the dataset. */
    offset[0] = offset[1] = 0;
    count[0] = dims_out[0];
    count[1] = dims_out[1];
    H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, offset, nullptr, count, nullptr);

    /* Define the memory dataspace. */
    dimsm[0] = dims_out[0];
    dimsm[1] = dims_out[1];
    dimsm[2] = 1;
    memspace = H5Screate_simple(3, dimsm, nullptr);

    /* Define memory hyperslab. */
    offset_out[0] = offset_out[1] = offset_out[2] = 0;
    count_out[0] = dims_out[0];
    count_out[1] = dims_out[1];
    count_out[2] = 1;
    H5Sselect_hyperslab(memspace, H5S_SELECT_SET, offset_out, nullptr, count_out, nullptr);

    /* Read data from hyperslab in the file into the hyperslab in memory and display. */
    switch (t_class) {
        case H5T_INTEGER:
            data_out = new int[dims_out[0] * dims_out[1]];
            H5Dread(dataset, H5T_NATIVE_INT, memspace, dataspace, H5P_DEFAULT, data_out);
            break;
        case H5T_FLOAT:
            data_out = new float[dims_out[0] * dims_out[1]];
            H5Dread(dataset, H5T_NATIVE_FLOAT, memspace, dataspace, H5P_DEFAULT, data_out);
            break;
        default:
            printf("Illegal dataset class type\n");
            break;
    }

    /* Close/release resources. */
    H5Tclose(datatype);
    H5Dclose(dataset);
    H5Sclose(dataspace);
    H5Sclose(memspace);
    H5Fclose(file);

    return data_out;
}

int main() {
    faiss::distance_compute_blas_threshold = 800;

    faiss::Index *cpu_index = nullptr;
    faiss::IndexIVF* cpu_ivf_index = nullptr;
    faiss::Index *index_flat = nullptr;
    faiss::gpu::StandardGpuResources res;

    //generate data
    xb = (float*)hdf5_read(filename, "train", H5T_FLOAT, d, nb);
    xq = (float*)hdf5_read(filename, "test", H5T_FLOAT, d, nq);
    int *gt_int = (int *)hdf5_read(filename, "neighbors", H5T_INTEGER, k2, nq);
    gt = new faiss::Index::idx_t[k2 * nq];
    for (size_t i = 0; i < k2 * nq; i++) {
        gt[i] = gt_int[i];
    }
    delete [] gt_int;
    std::vector<int64_t > ids(nb);
#pragma omp parallel for
    for (size_t j = 0; j < nb; ++j) {
        ids[j] = j;
    }
    printf("nb: %ld, nq: %ld, k: %ld, size: %ld\n", nb, nq, k2, (d * nb * sizeof(float)) );

    //generate index
    const char *index_ivf_file = "9.ivf.index";
    const char *index_flat_file = "9.flat.index";
    if((access(index_ivf_file,F_OK))==-1) {
        faiss::Index *ori_index = faiss::index_factory(d, index_description, faiss::METRIC_L2);

        auto device_index = faiss::gpu::index_cpu_to_gpu(&res, 0, ori_index);

        assert(!device_index->is_trained);
        device_index->train(nb, xb);
        assert(device_index->is_trained);
        device_index->add_with_ids(nb, xb, ids.data());  // add vectors to the index

        printf("is_trained = %s\n", device_index->is_trained ? "true" : "false");
        printf("ntotal = %ld\n", device_index->ntotal);

        cpu_index = faiss::gpu::index_gpu_to_cpu ((device_index));
        faiss::write_index(cpu_index, index_ivf_file);
    } else{
        cpu_index = faiss::read_index(index_ivf_file);
    }
    printf("generate ivf index complete.\n");

    if((access(index_flat_file,F_OK))==-1) {
        index_flat = new faiss::IndexFlatL2(d);

        auto ori_index = new faiss::IndexIDMap(reinterpret_cast<Index*>(index_flat));

        assert(ori_index->is_trained);
        ori_index->add_with_ids(nb, xb, ids.data());
        faiss::write_index(ori_index, index_flat_file);
        index_flat = ori_index;
    } else{
        index_flat = faiss::read_index(index_flat_file);
    }
    printf("generate flat index complete\n");

    cpu_ivf_index = dynamic_cast<faiss::IndexIVF*>(cpu_index);

    ColumnArrayInvertedSets cis((ArrayInvertedLists *)cpu_ivf_index->invlists, ColumnType::INT64, ColumnType::INT32, ColumnType::INT16, ColumnType::INT8, ColumnType::FLOAT, ColumnType::DOUBLE);
    auto cs = new ColumnSet(nb, ColumnType::INT64, ColumnType::INT32, ColumnType::INT16, ColumnType::INT8, ColumnType::FLOAT, ColumnType::DOUBLE);

    faiss::IndexComposition index_composition0;
    index_composition0.index = cpu_index;
    index_composition0.quantizer = nullptr;
    index_composition0.mode = 1; // only quantizer

    std::set<int64_t > ground_ids_set;
    int count = 0;
    for (size_t l = 0; l < nq; ++l) {
        for (size_t i = 0; i < k; ++i) {
            ground_ids_set.insert(gt[l * k2 + i]);
            count++;
        }
    }

    std::vector<int64_t > ground_ids = {};
    ground_ids.assign(ground_ids_set.begin(), ground_ids_set.end());

    printf("ground_ids_num: %ld\n", ground_ids.size());

    int64_t id_num = 0;

    CpuExecutor(&index_composition0, cpu_index, index_flat, (ColumnInvertedSets *)&cis, cs);

    id_num = 1000;
    cs->deleteByIds(id_num, ground_ids.data(), dynamic_cast<IndexIDMap *>(index_flat));
    cis.deleteByIds(id_num, ground_ids.data(), (ArrayInvertedLists *)cpu_ivf_index->invlists);
    printf("delete vectors complete\n");
    CpuExecutor(&index_composition0, cpu_index, index_flat, (ColumnInvertedSets *)&cis, cs);

    id_num = 5000;
    cs->deleteByIds(id_num, ground_ids.data(), dynamic_cast<IndexIDMap *>(index_flat));
    cis.deleteByIds(id_num, ground_ids.data(), (ArrayInvertedLists *)cpu_ivf_index->invlists);
    printf("delete vectors complete\n");
    CpuExecutor(&index_composition0, cpu_index, index_flat, (ColumnInvertedSets *)&cis, cs);

    id_num = 10000;
    cs->deleteByIds(id_num, ground_ids.data(), dynamic_cast<IndexIDMap *>(index_flat));
    cis.deleteByIds(id_num, ground_ids.data(), (ArrayInvertedLists *)cpu_ivf_index->invlists);
    printf("delete vectors complete\n");
    CpuExecutor(&index_composition0, cpu_index, index_flat, (ColumnInvertedSets *)&cis, cs);

    id_num = 20000;
    cs->deleteByIds(id_num, ground_ids.data(), dynamic_cast<IndexIDMap *>(index_flat));
    cis.deleteByIds(id_num, ground_ids.data(), (ArrayInvertedLists *)cpu_ivf_index->invlists);
    printf("delete vectors complete\n");
    CpuExecutor(&index_composition0, cpu_index, index_flat, (ColumnInvertedSets *)&cis, cs);

    delete [] xq;
    delete [] xb;
    delete [] gt;
    delete cs;
    return 0;
}

