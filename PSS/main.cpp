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
#include <immintrin.h>
#include <cmath>
#include <cstddef>
#include "../hnswlib/hnswlib/hnswlib.h"

using namespace std;

float l2_distance(const float *a, const float *b, int dim)
{
    float result = 0;
    for (int i = 0; i < dim; i++)
    {
        result += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return std::sqrt(result);
}

float inner_product(const float *a, const float *b, int dim)
{
    float result = 0;
    for (int i = 0; i < dim; i++)
    {
        result += a[i] * b[i];
    }
    return result;
}

struct NeighborX
{
    unsigned id;
    float dist;
    bool flag;
    uint16_t m;
    uint16_t M;
    bool checked;
    int size;
    vector<int> div;
    int round;
    NeighborX(unsigned i, float d) : id(i), dist(d), flag(true), m(1), M(0), checked(false), size(0), round(0)
    {
    }
    NeighborX() : id(-1), dist(-1), flag(true), m(1), M(0), checked(false), size(0), round(0)
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

struct A_star_result
{
    int topk;
    float *bscore;
    float *score;
    int **sol;

    A_star_result(int topk) : topk(topk)
    {
        bscore = new float[topk + 1];
        score = new float[topk + 1];
        sol = new int *[topk + 1];
        memset(bscore, 0, sizeof(float) * (topk + 1));
        memset(score, 0, sizeof(float) * (topk + 1));
        memset(sol, 0, sizeof(int *) * (topk + 1));
    }

    ~A_star_result()
    {
        delete[] bscore;
        delete[] score;
        for (int i = 0; i < topk + 1; ++i)
            if (sol[i])
                delete[] sol[i];
        delete[] sol;
    }
};

void update_result(HeapNode *now, int now_topk, int size, A_star_result *tk, int *invert, vector<NeighborX> &knn)
{
    vector<int> v_nod;
    boost::dynamic_bitset<> used(size, false);

    HeapNode *nod = now;
    for (int i = 0; i < now->n; ++i)
    {
        used[nod->last_pos] = true;
        v_nod.push_back(nod->last_pos);
        nod = nod->prev;
    }
    for (int i = 0; i < now->n / 2; ++i)
        swap(v_nod[i], v_nod[now->n - i - 1]);

    int nowk = now->n;
    float nowscore = now->score;
    for (int i = now->last_pos + 1; i < size && nowk < now_topk; ++i)
    {
        bool succ = true;
        for (int j = 0; j < knn[i].div.size(); ++j)
        {
            if (used[invert[knn[i].div[j]]])
            {
                succ = false;
                break;
            }
        }
        if (succ)
        {
            ++nowk;
            used[i] = true;
            v_nod.push_back(i);
            nowscore += knn[i].dist;
            if (nowscore > tk->bscore[nowk])
                tk->bscore[nowk] = nowscore;
            if (nowscore > tk->score[nowk])
            {
                tk->score[nowk] = nowscore;
                if (tk->sol[nowk] == NULL)
                    tk->sol[nowk] = new int[nowk];
                for (int i = 0; i < nowk; ++i)
                    tk->sol[nowk][i] = v_nod[i];
            }
        }
    }

    if (now->n > 0)
    {
        if (now->score > tk->score[now->n])
        {
            tk->score[now->n] = now->score;
            nod = now;

            if (tk->sol[now->n] == NULL)
                tk->sol[now->n] = new int[now->n];

            for (int i = 0; i < now->n; ++i)
            {
                tk->sol[now->n][now->n - i - 1] = nod->last_pos;
                nod = nod->prev;
            }

            if (nod->last_pos != -1)
                printf("[NLNN]");

            if (now->score > tk->bscore[now->n])
                tk->bscore[now->n] = now->score;

            for (int i = now->n + 1; i <= now_topk; ++i)
            {
                if (tk->bscore[i - 1] > tk->bscore[i])
                    tk->bscore[i] = tk->bscore[i - 1];
            }
        }
    }
}

bool update_bound(int *mark, int nowmark, HeapNode *nod, int now_topk, int *invert, A_star_result *tk, int size, vector<NeighborX> &knn)
{
    bool succ = false;
    nod->bound = nod->score;
    int pos = nod->last_pos + 1;
    for (int i = nod->n; i < now_topk && pos < size; ++pos)
    {
        bool flag = true;
        for (int j = 0; j < knn[pos].div.size(); ++j)
            if (mark[invert[knn[pos].div[j]]] == nowmark)
            {
                flag = false;
                break;
            }
        if (flag)
        {
            nod->bound += knn[pos].dist;
            ++i;
            if (nod->bound > tk->bscore[i] - 1e-6f)
                succ = true;
        }
    }
    return succ;
}

A_star_result *A_star(int *invert, vector<NeighborX> &knn, int size, int topk)
{
    A_star_result *tk = new A_star_result(topk);

    Heap *h = new Heap();
    HeapNode *nod = new HeapNode();
    nod->bound = 1e+10f;
    h->add(nod);
    int nowmark = 0;
    int *mark = new int[size];
    memset(mark, -1, sizeof(int) * size);

    int now_topk = topk;

    HeapNode *pre = NULL, *now = NULL;

    while (!h->is_empty())
    {
        pre = now;
        now = h->get_top();

        if (now->bound <= tk->score[now_topk])
        {
            if (now_topk == 1)
                break;
            else
            {
                --now_topk;

                int tmpn = 0;
                for (int i = 1; i <= h->n; ++i)
                    if (h->heap[i]->n <= now_topk)
                    {
                        ++nowmark;
                        for (nod = h->heap[i]; nod->last_pos >= 0; nod = nod->prev)
                            mark[nod->last_pos] = nowmark;
                        if (update_bound(mark, nowmark, h->heap[i], now_topk, invert, tk, size, knn))
                            h->heap[++tmpn] = h->heap[i];
                    }
                h->n = tmpn;

                h->reconstruct();
                continue;
            }
        }

        h->remove_top();

        ++nowmark;
        for (nod = now; nod->last_pos >= 0; nod = nod->prev)
            mark[nod->last_pos] = nowmark;

        if (now->n == now_topk || now->last_pos == size - 1)
            continue;

        for (int p = now->last_pos + 1; p < size; ++p)
        {
            bool succ = true;

            for (int i = 0; i < knn[p].div.size(); ++i)
            {
                int q = invert[knn[p].div[i]];
                if (mark[q] == nowmark)
                {
                    succ = false;
                    break;
                }
            }
            if (!succ)
                continue;
            mark[p] = nowmark;

            nod = new HeapNode();
            nod->last_pos = p;
            nod->n = now->n + 1;
            nod->prev = now;
            nod->score = now->score;
            nod->score += knn[p].dist;

            if (!update_bound(mark, nowmark, nod, now_topk, invert, tk, size, knn))
                if (nod->bound <= tk->bscore[nod->n])
                {
                    delete nod;
                    nod = NULL;
                }

            if (nod)
            {
                update_result(nod, now_topk, size, tk, invert, knn);
                if (nod->n < now_topk)
                    h->add(nod);
                else
                    delete nod;
            }

            mark[p] = -1;
        }
    }
    delete h;
    delete[] mark;

    return tk;
}

A_star_result *search_hnsw(hnswlib::HierarchicalNSW<float> &hnsw, float *query, float *data, int N, int d, int kn, int kr, float thres, int ef)
{
    int count = 0;
    bool next_level = false;
    vector<NeighborX> knn(N);
    vector<NeighborX> helper(kr);
    // flags access is totally random, so use small block to avoid
    // extra memory access
    boost::dynamic_bitset<> flags(N, false);

    unsigned seed = time(NULL);
    mt19937 rng(seed);
    unsigned n_comps = 0;
    int current_round = 0;
    float cutline = -100;

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
    int L = 1;
    int got = 0;
    knn[0].id = currObj;
    flags[knn[0].id] = true;
    knn[0].flag = true;
    knn[0].dist = inner_product(query, data + currObj * d, d);
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
            float dist = inner_product(query, data + id * d, d);
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
        if (k >= now_stop || (k / ef >= kr && knn[k / ef - 1].dist < cutline))
        {
            for (int i = 0; i < got; i++)
            {
                bool in_k = false;
                for (int j = 0; j < k / ef; j++)
                {
                    if (helper[i].id == knn[j].id)
                    {
                        in_k = true;
                        break;
                    }
                }
                if (!in_k)
                {
                    for (int j = i; j < got - 1; j++)
                    {
                        helper[j] = helper[j + 1];
                    }
                    got--;
                    i--;
                }
            }
            if (got == 0)
            {
                got = 1;
                helper[0].dist = knn[0].dist;
                helper[0].id = knn[0].id;
                knn[0].checked = true;
            }
            if (got < kr)
            {
                for (int i = 0; i < k / ef; i++)
                {
                    if (!knn[i].checked)
                    {
                        knn[i].checked = true;
                        bool div_check = true;
                        for (int j = 0; j < got; j++)
                        {
                            float dist = inner_product(data + helper[j].id * d, data + knn[i].id * d, d);
                            if (dist > thres)
                            {
                                div_check = false;
                                break;
                            }
                        }
                        if (div_check)
                        {
                            helper[got].dist = knn[i].dist;
                            helper[got].id = knn[i].id;
                            helper[got].checked = true;
                            got++;
                        }
                    }
                    if (got == kr)
                        break;
                }
            }
            if (got == kr)
            {
                cutline = 100;
                current_round++;
                if (current_round == 1)
                {
                    for (int i = 0; i < k / ef; i++)
                    {
                        knn[i].round = current_round;
                        for (int j = i + 1; j < k / ef; j++)
                        {
                            float dist = inner_product(data + knn[j].id * d, data + knn[i].id * d, d);
                            if (dist > thres)
                            {
                                knn[i].size++;
                                knn[j].size++;
                                knn[i].div.push_back(knn[j].id);
                                knn[j].div.push_back(knn[i].id);
                            }
                        }
                    }
                }
                else
                {
                    for (int i = 0; i < k / ef; i++)
                    {
                        for (int j = i + 1; j < k / ef; j++)
                        {
                            if (knn[i].round == knn[j].round)
                                continue;
                            float dist = inner_product(data + knn[j].id * d, data + knn[i].id * d, d);
                            if (dist > thres)
                            {
                                knn[i].size++;
                                knn[j].size++;
                                knn[i].div.push_back(knn[j].id);
                                knn[j].div.push_back(knn[i].id);
                            }
                        }
                        knn[i].round = current_round;
                    }
                }
                int *invert = new int[N];
                for (int i = 0; i < k / ef; i++)
                {
                    invert[knn[i].id] = i;
                }
                A_star_result *tk = A_star(invert, knn, k / ef, kr);
                bool tt = true;
                for (int i = 1; i < kr; ++i)
                {
                    float judge = (tk->score[kr] - tk->score[i]) / (kr - i);
                    if (judge < knn[k / ef - 1].dist)
                    {
                        tt = false;
                        if (judge < cutline)
                            cutline = judge;
                    }
                }
                delete[] invert;
                if (tt)
                {
                    for (int j = 0; j < kr; j++)
                    {
                        tk->sol[kr][j] = knn[tk->sol[kr][j]].id;
                    }
                    return tk;
                }
                else
                {
                    delete tk;
                    now_stop = N;
                }
            }
            else
                now_stop += ef * kr;
        }
    }
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
        sprintf(log_path, "../log/v2/laion/test/100_1000_%d.log", ef);
        ofstream logFile(log_path, std::ios::out);

        printf("Searching...\n");

        for (float sim = 0.9; sim >= 0.779; sim -= 0.01)
        {
            for (int kr = 5; kr < 25; kr = kr + 5)
            {
                float period = 0;
                float value = 0;
                int query_num = 1000;
                char single_log_path[1024];
                sprintf(single_log_path, "../log/v2/laion/100_1000_%.2f_%d_%d.log", sim, kr, ef);
                ofstream single_logFile(single_log_path, std::ios::out);
                for (int qn = 0; qn < query_num; qn++)
                {
                    float *qq = query_deep + qn * d;
                    int t = clock();
                    A_star_result *rs = search_hnsw(hnsw, qq, data, N, d, kn, kr, sim, ef);
                    t = clock() - t;
                    period += t;
                    value += rs->score[kr];
                    for (int i = 0; i < kr; i++)
                    {
                        single_logFile << rs->sol[kr][i] << " ";
                    }
                    single_logFile << t << ' ' << rs->score[kr] << endl;
                    cout << sim << ' ' << kr << ' ' << qn << ' ' << t << endl;
                    delete rs;
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