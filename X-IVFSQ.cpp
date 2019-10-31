#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <fstream>

#include <IndexFlat.h>
#include <IndexIVF.h>
#include <gpu/StandardGpuResources.h>
#include <gpu/GpuAutoTune.h>

constexpr long DIMENSION = 128;
constexpr long TOPK = 100;

using namespace std;

void ivecs_read(char* fname, int *data, int &d, int &num){
    int count;
    ifstream input(fname, ios::in | ios::binary);
    if(!input.is_open()){
        printf("File not found\n");
        exit(-1);
    }
    int *p = data;
    num = 0;
    while(input.read((char *)&count, sizeof(int))){
        num++;
        d = count;
        input.read((char *)p, sizeof(int) * d);
        p = p + count;
    }
    input.close();
    return;
}

void fvecs_read(char* fname, float *data, int &d, int &num){
    int count;
    ifstream input(fname, ios::in | ios::binary);
    if(!input){
        printf("File not found\n");
        exit(-1);
    }
    float *p = data;
    num = 0;
    while(input.read((char *)&count, sizeof(int))){
        num++;
        d = count;
        input.read((char *)p, sizeof(float) * d);
        p = p + count;
    }
    input.close();
    return;
}

void load_sift1M(float *xb, float *xq, float *xt, int *gt, int &d1, int &d2, int &nxt, int &nb, int &nq, int &ngt){
    printf("Loading sift1M...");
    fvecs_read((char*)("../sift1M/sift_learn.fvecs"), xt, d1, nxt);
    fvecs_read((char*)("../sift1M/sift_base.fvecs"), xb, d1, nb);
    fvecs_read((char*)("../sift1M/sift_query.fvecs"), xq, d1, nq);
    ivecs_read((char*)("../sift1M/sift_groundtruth.ivecs"), gt, d2, ngt);
    printf("done\n");
}

void evaluate(faiss::Index& index, float *xq, int *gt, long int k, float &t, float *r, int nq){
    long *I2 = new long[k * nq];
    float *D2 = new float[k * nq];

    float t0, t1;
    t0 = (float)(clock())/CLOCKS_PER_SEC;
    index.search(nq, xq, k, D2, I2);  // noqa: E741
    t1 = (float)(clock())/CLOCKS_PER_SEC;

    float recalls[1024];
    long i = 1, sum = 0;

    while(i <= k){
        sum = 0;
        for(int j = 0; j < nq; j++){
            for( int l = 0; l < i; l++){
                if(I2[ j * k + l ] == gt[ j * TOPK ])
                    sum++;
            }
        }
        recalls[i] = (float(sum)) / float(nq);
        r[i] = recalls[i];
        i *= 10;

    }

    t = (float)((t1 - t0) * 1000.0 / nq);

    delete [] I2;
    delete [] D2;
}

int main() {
    int nb, nq, nxt, ngt, d1, d2;
    long int nprobe = 0;
    float t;
    float *r = new float[1024];
    float *xt = new float[DIMENSION * 100000];
    float *xb = new float[DIMENSION * 1000000];
    float *xq = new float[DIMENSION * 10000];
    int *gt = new int[TOPK * 10000];
    printf("load data\n");
    load_sift1M(xb, xq, xt, gt, d1, d2, nxt, nb, nq, ngt);

    // we need only a StandardGpuResources per GPU
    faiss::gpu::StandardGpuResources res;

    const char* index_description = "IVF16384,SQ8";
    faiss::Index *ori_index = faiss::index_factory(DIMENSION, index_description, faiss::METRIC_L2);

    auto device_index = faiss::gpu::index_cpu_to_gpu(&res, 0, ori_index);

    auto gpu_index_ivf_ptr = std::shared_ptr<faiss::Index>(device_index);

    assert(!device_index->is_trained);
    device_index->train(nxt, xt);
    assert(device_index->is_trained);
    device_index->add(nb, xb);  // add vectors to the index

    printf("sq8cpu: is_trained = %s\n", device_index->is_trained ? "true" : "false");
    printf("sq8cpu: ntotal = %ld\n", device_index->ntotal);

    auto cpu_index = faiss::gpu::index_gpu_to_cpu ((device_index));
    faiss::IndexIVF* ivf_index =
            dynamic_cast<faiss::IndexIVF*>(cpu_index);

    printf("sq8cpu: benchmark\n");
    for(int lk = 0; lk < 10; lk++){
        nprobe = 1 << lk;
        ivf_index->nprobe = nprobe;
        evaluate(*cpu_index, xq, gt, 100, t, r, nq);

        // the recall should be 1 at all times
        printf("sq8cpu: nprobe=%ld %.3f ms, R@1 %.4f\n", nprobe, t, r[1]);
    }

    return 0;
}