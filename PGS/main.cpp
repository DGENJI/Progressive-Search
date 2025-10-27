#include <unordered_set>
#include <mutex>
#include <iostream>
#include <fstream>
#include <random>
#include <algorithm>
#include <vector>
#include <boost/timer/timer.hpp>
#include <boost/dynamic_bitset.hpp>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/moment.hpp>
#include "boost/smart_ptr/detail/spinlock.hpp"
#include <ctime>
#include "heap.h"
#include "../hnswlib/hnswlib/hnswlib.h"

using namespace std;

struct NeighborX
{
    uint32_t id;
    float dist;
    bool flag;
    uint16_t m;
    uint16_t M;
    bool checked;
    NeighborX(unsigned i, float d) : id(i), dist(d), flag(true), m(1), M(0), checked(false)
    {
    }
    NeighborX() : id(-1), dist(-1), flag(true), m(1), M(0), checked(false)
    {
    }
};

template <typename RNG>
static void GenRandom(RNG &rng, unsigned *addr, unsigned size, unsigned N)
{
    if (N == size)
    {
        for (unsigned i = 0; i < size; ++i)
        {
            addr[i] = i;
        }
        return;
    }
    for (unsigned i = 0; i < size; ++i)
    {
        addr[i] = rng() % (N - size);
    }
    sort(addr, addr + size);
    for (unsigned i = 1; i < size; ++i)
    {
        if (addr[i] <= addr[i - 1])
        {
            addr[i] = addr[i - 1] + 1;
        }
    }
    unsigned off = rng() % N;
    for (unsigned i = 0; i < size; ++i)
    {
        addr[i] = (addr[i] + off) % N;
    }
}

static inline bool operator<(NeighborX const &n1, NeighborX const &n2)
{
    return n1.dist < n2.dist;
}

static inline bool operator>(NeighborX const &n1, NeighborX const &n2)
{
    return n1.dist > n2.dist;
}

static inline bool operator==(NeighborX const &n1, NeighborX const &n2)
{
    return n1.id == n2.id;
}

unsigned UpdateKnnList(NeighborX *addr, unsigned K, NeighborX nn)
{
    // find the location to insert
    unsigned j;
    unsigned i = K;
    while (i > 0)
    {
        j = i - 1;
        if (addr[j].dist >= nn.dist)
            break;
        i = j;
    }
    // check for equal ID
    unsigned l = i;
    while (l > 0)
    {
        j = l - 1;
        if (addr[j].dist > nn.dist)
            break;
        if (addr[j].id == nn.id)
            return K + 1;
        l = j;
    }
    // i <= K-1
    j = K;
    while (j > i)
    {
        addr[j] = addr[j - 1];
        --j;
    }
    addr[i] = nn;
    return i;
}

int search_hnsw(hnswlib::HierarchicalNSW<float> &hnsw, float *query, float *data, int *&ids, float *&dists, int N, int d, int kn, int kr, float thres, int ef)
{
    int count = 0;
    bool next_level = false;
    vector<NeighborX> knn(N);
    vector<NeighborX> results(kr);
    // flags access is totally random, so use small block to avoid
    // extra memory access
    boost::dynamic_bitset<> flags(N, false);

    unsigned seed = time(NULL);
    mt19937 rng(seed);
    unsigned n_comps = 0;

    unsigned L = 0; // generate random starting points
    hnswlib::tableint currObj = hnsw.enterpoint_node_;
    float curdist = hnsw.fstdistfunc_(query, hnsw.getDataByInternalId(hnsw.enterpoint_node_), hnsw.dist_func_param_);

    for (int level = hnsw.maxlevel_; level > 0; level--)
    {
        bool changed = true;
        while (changed)
        {
            changed = false;
            unsigned int *data;

            data = (unsigned int *)hnsw.get_linklist(currObj, level);
            int size = hnsw.getListCount(data);

            hnswlib::tableint *datal = (hnswlib::tableint *)(data + 1);
            for (int i = 0; i < size; i++)
            {
                hnswlib::tableint cand = datal[i];
                if (cand < 0 || cand > hnsw.max_elements_)
                    throw std::runtime_error("cand error");
                float d = hnsw.fstdistfunc_(query, hnsw.getDataByInternalId(cand), hnsw.dist_func_param_);

                if (d < curdist)
                {
                    curdist = d;
                    currObj = cand;
                    changed = true;
                }
            }
        }
    }
    int k = 0;
    L = 1;
    int got = 0;
    knn[0].id = currObj;
    flags[knn[0].id] = true;
    knn[0].flag = true;
    float dist_init = 0;
    for (int j = 0; j < d; j++)
    {
        dist_init += data[currObj * d + j] * query[j];
        // dist_init += (data[currObj * d + j] - query[j]) * (data[currObj * d + j] - query[j]);
    }
    // dist_init = 1 - sqrt(dist_init);
    knn[0].dist = dist_init;
    int now_stop = ef * kr;
    while (k < L)
    {
        auto &e = knn[k];
        if (!e.flag)
        { // all neighbors of this node checked
            ++k;
            continue;
        }
        int *data_list = (int *)hnsw.get_linklist0(e.id);
        unsigned beginM = 1;
        unsigned endM = hnsw.getListCount((hnswlib::linklistsizeint *)data_list); // check this many entries
        e.flag = false;
        // all modification to knn[k] must have been done now,
        // as we might be relocating knn[k] in the loop below
        for (unsigned m = beginM; m < endM; ++m)
        {
            unsigned id = *(data_list + m);
            // BOOST_VERIFY(id < graph.size());
            if (flags[id])
            {
                continue;
            }
            flags[id] = true;
            float dist = 0;
            for (int j = 0; j < d; j++)
            {
                dist += data[id * d + j] * query[j];
                // dist += (data[id * d + j] - query[j]) * (data[id * d + j] - query[j]);
            }
            // dist = 1 - sqrt(dist);
            NeighborX nn(id, dist);
            unsigned r = UpdateKnnList(&knn[0], L, nn);
            // if (r > L) continue;
            if (L + 1 < knn.size())
                ++L;
            if (r < L)
            {
                if (r < k)
                {
                    k = r;
                }
            }
        }
        if (k >= now_stop)
        {
            if (got == 0)
            {
                got = 1;
                results[0].dist = knn[0].dist;
                results[0].id = knn[0].id;
                knn[0].checked = true;
            }
            for (int i = 0; i < k / ef; i++)
            {
                if (!knn[i].checked)
                {
                    knn[i].checked = true;
                    bool div = true;
                    for (int j = 0; j < got; j++)
                    {
                        float dist = 0;
                        for (int k = 0; k < d; k++)
                        {
                            dist += data[knn[i].id * d + k] * data[results[j].id * d + k];
                            // dist += (data[knn[i].id * d + k] - data[results[j].id * d + k]) * (data[knn[i].id * d + k] - data[results[j].id * d + k]);
                        }
                        // dist = sqrt(dist);
                        if (dist > thres)
                        {
                            div = false;
                            break;
                        }
                    }
                    if (div)
                    {
                        results[got].dist = knn[i].dist;
                        results[got].id = knn[i].id;
                        results[got].checked = true;
                        got++;
                    }
                }
                if (got == kr)
                    break;
            }
            if (got == kr)
            {
                break;
            }
            else
                now_stop += ef * kr;
        }
    }
    L = results.size();
    /*
    if (!(L <= params.K)) {
        cerr << L << ' ' << params.K << endl;
    }
    */
    // check epsilon
    ids = new int[L];
    for (unsigned k = 0; k < L; ++k)
    {
        ids[k] = results[k].id;
    }

    dists = new float[L];
    for (unsigned k = 0; k < L; ++k)
    {
        dists[k] = results[k].dist;
    }
    return k;
}

inline void read_fvecs(const char *data_name, int &dim, int &n, float *&data)
{
    std::ifstream in(data_name, std::ios::binary); // 以二进制的方式打开文件
    in.read((char *)&dim, 4);                      // 读取向量维度
    in.seekg(0, std::ios::end);                    // 光标定位到文件末尾
    std::ios::pos_type ss = in.tellg();            // 获取文件大小（多少字节）
    size_t fsize = (size_t)ss;
    n = (unsigned)(fsize / (dim + 1) / 4); // 数据的个数
    data = new float[(size_t)n * (size_t)dim];
    in.seekg(0, std::ios::beg); // 光标定位到起始处
    for (size_t i = 0; i < n; i++)
    {
        in.seekg(4, std::ios::cur);                 // 光标向右移动4个字节
        in.read((char *)(data + i * dim), dim * 4); // 读取数据到一维数据data中
    }
    in.close();
}

inline void read_ivecs(const char *data_name, int &dim, int &n, int *&data)
{
    std::ifstream in(data_name, std::ios::binary); // 以二进制的方式打开文件
    in.read((char *)&dim, 4);                      // 读取向量维度
    in.seekg(0, std::ios::end);                    // 光标定位到文件末尾
    std::ios::pos_type ss = in.tellg();            // 获取文件大小（多少字节）
    size_t fsize = (size_t)ss;
    n = (unsigned)(fsize / (dim + 1) / 4); // 数据的个数
    data = new int[(size_t)n * (size_t)dim];
    in.seekg(0, std::ios::beg); // 光标定位到起始处
    for (size_t i = 0; i < n; i++)
    {
        in.seekg(4, std::ios::cur);                 // 光标向右移动4个字节
        in.read((char *)(data + i * dim), dim * 4); // 读取数据到一维数据data中
    }
    in.close();
}

inline void read_fbin(const char *data_name, int &dim, int &n, float *&data)
{
    std::ifstream in(data_name, std::ios::binary);
    in.read((char *)&n, 4);
    in.read((char *)&dim, 4);
    data = new float[(size_t)n * (size_t)dim];
    in.read((char *)data, dim * n);
    in.close();
}

int main()
{
    int N;
    int d;
    int kn;
    float *data;
    int *ids;
    float *dists;
    printf("Loading...\n");
    cout << CLOCKS_PER_SEC << endl;
    read_fvecs("../laion/laion-art_norm.fvecs", d, N, data);
    cout << "data done" << endl;
    std::string index_path = "../laion_200_16_ip.bin";
    hnswlib::InnerProductSpace space(d);
    hnswlib::HierarchicalNSW<float> hnsw(&space, index_path, N);

    float *query_deep;
    int query_n;
    read_fvecs("../laion/query_300.fvecs", d, query_n, query_deep);
    cout << "query done" << endl;

    for (int ef = 10; ef < 101; ef += 10)
    {

        char log_path[1024];
        sprintf(log_path, "../log/greedy_hnsw/laion/test/test_100_1000_%d.log", ef);
        ofstream logFile(log_path, std::ios::out);

        printf("Searching...\n");
        for (float sim = 0.9; sim >= 0.78; sim = sim -= 0.02)
        {
            for (int kr = 5; kr < 25; kr = kr + 5)
            {
                float period = 0;
                float value = 0;
                int query_num = 1000;
                char single_log_path[1024];
                sprintf(single_log_path, "../log/greedy_hnsw/laion/laion_100_100_%.2f_%d_%d.log", sim, kr, ef);
                ofstream single_logFile(single_log_path, std::ios::out);
                for (int qn = 0; qn < query_num; qn++)
                {
                    float *qq = query_deep + qn * d;
                    int t = clock();
                    int L_hnsw = search_hnsw(hnsw, qq, data, ids, dists, N, d, kn, kr, sim, ef);
                    t = clock() - t;
                    period += t;
                    float value_single = 0;
                    for (int i = 0; i < kr; i++)
                    {
                        value_single += dists[i];
                    }
                    value += value_single;
                    for (int i = 0; i < kr; i++)
                    {
                        single_logFile << ids[i] << " ";
                    }
                    single_logFile << t << ' ' << value_single << endl;
                    cout << sim << ' ' << kr << ' ' << qn << ' ' << t << endl;
                    delete[] ids;
                    delete[] dists;
                }
                single_logFile.close();
                logFile << sim << ' ' << kr << ' ' << period / query_num << ' ' << value / query_num << endl;
            }
        }
        logFile.close();
    }

    delete[] data;
    delete[] query_deep;

    return 0;
}