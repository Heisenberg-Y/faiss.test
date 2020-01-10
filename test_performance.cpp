//
// Created by heisenberg on 23/12/19.
//
#include <iostream>
#include <faiss/utils/utils.h>
#include <vector>
#include <fstream>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

int main(){
//    int64_t nb = 10000000;
//    std::vector<int64_t > vec_ids(nb);
//    std::vector<int64_t > doc_ids(nb);
//    std::vector<int64_t > ids(nb);
//    std::vector<int64_t > skip_lists(nb);
//
//    double t4, t5;
//#pragma omp parallel for
//    for (int64_t i = 0; i < nb; ++i) {
//        vec_ids[i] = i;
//        doc_ids[i] = nb - i - 1;
//        skip_lists[i] = 0;
//    }
//
//
//    t4 = faiss::getmillisecs();
//    for (int64_t j = 0; j < nb; ++j) {
//        ids[j] = doc_ids[vec_ids[j] + skip_lists[j]];
//    }
//    t5 = faiss::getmillisecs();
//    printf("time: %0.2f\n", t5 - t4);

#if 1

    const size_t buffer_size = 7 * 1024 * 1024;
    char buffer[buffer_size];
    memset(buffer, 0, buffer_size);


    double t4 = faiss::getmillisecs();
    for(long i = 0; i < 1000; ++ i) {
        buffer[i] = 1;
        int fd = open("/tmp/temp", O_WRONLY|O_CREAT, 0777);
        write(fd, buffer, buffer_size);
        close(fd);
    }
    double t5 = faiss::getmillisecs();
    printf("cost: %0.2f\n", t5 - t4);
#else
    const size_t buffer_size = 7 * 1024 * 1024;
    char buffer[buffer_size];
    memset(buffer, 0, buffer_size);

    FILE *fptr;
    if ((fptr = fopen("/home/heisenberg/temp","wb")) == NULL){
        printf("Error! opening file");
        // Program exits if the file pointer returns NULL.
        exit(1);
    }

    double t4 = faiss::getmillisecs();
    for(int n = 0; n < 100; ++n)
    {
        buffer[n] = 1;
        fwrite(buffer, 1, buffer_size, fptr);
    }
    double t5 = faiss::getmillisecs();
    printf("cost: %0.2f\n", t5 - t4);

    fclose(fptr);

    return 0;
#endif
    return 0;
}