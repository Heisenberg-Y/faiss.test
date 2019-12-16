#include <iostream>

#include <cstdint>
#include <string>
#include <cstring>
#include <cpuid.h>

struct cpu_x86 {
    //  Vendor
    bool Vendor_AMD;
    bool Vendor_Intel;

    //  OS Features
    bool OS_AVX;
    bool OS_AVX512;

    //  Misc.
    bool HW_MMX;
    bool HW_x64;
    bool HW_ABM;
    bool HW_RDRAND;
    bool HW_BMI1;
    bool HW_BMI2;
    bool HW_ADX;
    bool HW_PREFETCHWT1;
    bool HW_MPX;

    //  SIMD: 128-bit
    bool HW_SSE;
    bool HW_SSE2;
    bool HW_SSE3;
    bool HW_SSSE3;
    bool HW_SSE41;
    bool HW_SSE42;
    bool HW_SSE4a;
    bool HW_AES;
    bool HW_SHA;

    //  SIMD: 256-bit
    bool HW_AVX;
    bool HW_XOP;
    bool HW_FMA3;
    bool HW_FMA4;
    bool HW_AVX2;

    //  SIMD: 512-bit
    bool HW_AVX512_F;
    bool HW_AVX512_PF;
    bool HW_AVX512_ER;
    bool HW_AVX512_CD;
    bool HW_AVX512_VL;
    bool HW_AVX512_BW;
    bool HW_AVX512_DQ;
    bool HW_AVX512_IFMA;
    bool HW_AVX512_VBMI;

    cpu_x86() = default;

    void detect_host(){
        //  OS Features
        OS_AVX = detect_OS_AVX();
        OS_AVX512 = detect_OS_AVX512();

        //  Vendor
        std::string vendor(get_vendor_string());
        if (vendor == "GenuineIntel"){
            Vendor_Intel = true;
        }else if (vendor == "AuthenticAMD"){
            Vendor_AMD = true;
        }

        uint32_t info[4];
        cpuid(info, 0);
        int nIds = info[0];

        cpuid(info, 0x80000000);
        uint32_t nExIds = info[0];

        //  Detect Features
        if (nIds >= 0x00000001){
            cpuid(info, 0x00000001);
            HW_MMX    = (info[3] & ((uint32_t)1 << 23u)) != 0;
            HW_SSE    = (info[3] & ((uint32_t)1 << 25u)) != 0;
            HW_SSE2   = (info[3] & ((uint32_t)1 << 26u)) != 0;
            HW_SSE3   = (info[2] & ((uint32_t)1 <<  0u)) != 0;

            HW_SSSE3  = (info[2] & ((uint32_t)1 <<  9u)) != 0;
            HW_SSE41  = (info[2] & ((uint32_t)1 << 19u)) != 0;
            HW_SSE42  = (info[2] & ((uint32_t)1 << 20u)) != 0;
            HW_AES    = (info[2] & ((uint32_t)1 << 25u)) != 0;

            HW_AVX    = (info[2] & ((uint32_t)1 << 28u)) != 0;
            HW_FMA3   = (info[2] & ((uint32_t)1 << 12u)) != 0;

            HW_RDRAND = (info[2] & ((uint32_t)1 << 30u)) != 0;
        }
        if (nIds >= 0x00000007){
            cpuid(info, 0x00000007);
            HW_AVX2         = (info[1] & ((uint32_t)1 <<  5u)) != 0;

            HW_BMI1         = (info[1] & ((uint32_t)1 <<  3u)) != 0;
            HW_BMI2         = (info[1] & ((uint32_t)1 <<  8u)) != 0;
            HW_ADX          = (info[1] & ((uint32_t)1 << 19u)) != 0;
            HW_MPX          = (info[1] & ((uint32_t)1 << 14u)) != 0;
            HW_SHA          = (info[1] & ((uint32_t)1 << 29u)) != 0;
            HW_PREFETCHWT1  = (info[2] & ((uint32_t)1 <<  0u)) != 0;

            HW_AVX512_F     = (info[1] & ((uint32_t)1 << 16u)) != 0;
            HW_AVX512_CD    = (info[1] & ((uint32_t)1 << 28u)) != 0;
            HW_AVX512_PF    = (info[1] & ((uint32_t)1 << 26u)) != 0;
            HW_AVX512_ER    = (info[1] & ((uint32_t)1 << 27u)) != 0;
            HW_AVX512_VL    = (info[1] & ((uint32_t)1 << 31u)) != 0;
            HW_AVX512_BW    = (info[1] & ((uint32_t)1 << 30u)) != 0;
            HW_AVX512_DQ    = (info[1] & ((uint32_t)1 << 17u)) != 0;
            HW_AVX512_IFMA  = (info[1] & ((uint32_t)1 << 21u)) != 0;
            HW_AVX512_VBMI  = (info[2] & ((uint32_t)1 <<  1u)) != 0;
        }
        if (nExIds >= 0x80000001){
            cpuid(info, 0x80000001);
            HW_x64   = (info[3] & ((uint32_t)1 << 29u)) != 0;
            HW_ABM   = (info[2] & ((uint32_t)1 <<  5u)) != 0;
            HW_SSE4a = (info[2] & ((uint32_t)1 <<  6u)) != 0;
            HW_FMA4  = (info[2] & ((uint32_t)1 << 16u)) != 0;
            HW_XOP   = (info[2] & ((uint32_t)1 << 11u)) != 0;
        }
    }

    void print() const{
        std::cout << "CPU Vendor:" << std::endl;
        print("    AMD         = ", Vendor_AMD);
        print("    Intel       = ", Vendor_Intel);
        std::cout << std::endl;

        std::cout << "OS Features:" << std::endl;

        print("    OS AVX      = ", OS_AVX);
        print("    OS AVX512   = ", OS_AVX512);
        std::cout << std::endl;

        std::cout << "Hardware Features:" << std::endl;
        print("    MMX         = ", HW_MMX);
        print("    x64         = ", HW_x64);
        print("    ABM         = ", HW_ABM);
        print("    RDRAND      = ", HW_RDRAND);
        print("    BMI1        = ", HW_BMI1);
        print("    BMI2        = ", HW_BMI2);
        print("    ADX         = ", HW_ADX);
        print("    MPX         = ", HW_MPX);
        print("    PREFETCHWT1 = ", HW_PREFETCHWT1);
        std::cout << std::endl;

        std::cout << "SIMD: 128-bit" << std::endl;
        print("    SSE         = ", HW_SSE);
        print("    SSE2        = ", HW_SSE2);
        print("    SSE3        = ", HW_SSE3);
        print("    SSSE3       = ", HW_SSSE3);
        print("    SSE4a       = ", HW_SSE4a);
        print("    SSE4.1      = ", HW_SSE41);
        print("    SSE4.2      = ", HW_SSE42);
        print("    AES-NI      = ", HW_AES);
        print("    SHA         = ", HW_SHA);
        std::cout << std::endl;

        std::cout << "SIMD: 256-bit" << std::endl;
        print("    AVX         = ", HW_AVX);
        print("    XOP         = ", HW_XOP);
        print("    FMA3        = ", HW_FMA3);
        print("    FMA4        = ", HW_FMA4);
        print("    AVX2        = ", HW_AVX2);
        std::cout << std::endl;

        std::cout << "SIMD: 512-bit" << std::endl;
        print("    AVX512-F    = ", HW_AVX512_F);
        print("    AVX512-CD   = ", HW_AVX512_CD);
        print("    AVX512-PF   = ", HW_AVX512_PF);
        print("    AVX512-ER   = ", HW_AVX512_ER);
        print("    AVX512-VL   = ", HW_AVX512_VL);
        print("    AVX512-BW   = ", HW_AVX512_BW);
        print("    AVX512-DQ   = ", HW_AVX512_DQ);
        print("    AVX512-IFMA = ", HW_AVX512_IFMA);
        print("    AVX512-VBMI = ", HW_AVX512_VBMI);
        std::cout << std::endl;

        std::cout << "Summary:" << std::endl;
        print("    Safe to use AVX:     ", HW_AVX && OS_AVX);
        print("    Safe to use AVX512:  ", HW_AVX512_F && OS_AVX512);
        std::cout << std::endl;
    }

    void static print_host(){
        cpu_x86 features = cpu_x86();
        features.detect_host();
        features.print();
    }

    void static cpuid(uint32_t out[4], int32_t x){
        int outx[4];
        memcpy(outx, out, sizeof(int32_t) * 4);
        __cpuid_count(x, 0, out[0], out[1], out[2], out[3]);
    }

    uint64_t static xgetbv(unsigned int index){
        uint32_t eax, edx;
        __asm__ __volatile__("xgetbv" : "=a"(eax), "=d"(edx) : "c"(index));
        return ((uint64_t)edx << 32u) | eax;
    }

#define _XCR_XFEATURE_ENABLED_MASK  0

    static std::string get_vendor_string(){
        uint32_t CPUInfo[4];
        char name[13];

        cpuid(CPUInfo, 0);
        memcpy(name + 0, &CPUInfo[1], 4);
        memcpy(name + 4, &CPUInfo[3], 4);
        memcpy(name + 8, &CPUInfo[2], 4);
        name[12] = '\0';

        return name;
    }

    static void print(const char *label, bool yes){
        std::cout << label;
        std::cout << (yes ? "Yes" : "No") << std::endl;
    }

    bool static detect_OS_AVX(){
        //  Copied from: http://stackoverflow.com/a/22521619/922184

        bool avxSupported = false;

        uint32_t cpuInfo[4];
        cpuid(cpuInfo, 1);

        bool osUsesXSAVE_XRSTORE = (cpuInfo[2] & ((uint32_t)1 << 27u)) != 0;
        bool cpuAVXSuport = (cpuInfo[2] & ((uint32_t)1 << 28u)) != 0;

        if (osUsesXSAVE_XRSTORE && cpuAVXSuport)
        {
            uint64_t xcrFeatureMask = xgetbv(_XCR_XFEATURE_ENABLED_MASK);
            avxSupported = (xcrFeatureMask & 0x6u) == 0x6u;
        }

        return avxSupported;
    }

    bool static detect_OS_AVX512(){
        if (!detect_OS_AVX())
            return false;

        uint64_t xcrFeatureMask = xgetbv(_XCR_XFEATURE_ENABLED_MASK);
        return (xcrFeatureMask & 0xe6u) == 0xe6u;
    }
};

int main(){
    std::cout << "CPU Vendor String: " << cpu_x86::get_vendor_string() << std::endl << std::endl;
    cpu_x86::print_host();
}