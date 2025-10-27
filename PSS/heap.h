#pragma once

#pragma warning(disable : 4996)

#include <vector>
#include <map>
#include <list>
#include <set>
#include <queue>
#include <stack>
#include <algorithm>
#include <functional>
#include <sstream>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <ctime>

using namespace std;

class HeapNode
{
public:
    HeapNode *prev;
    int last_pos;
    int n;
    float score;
    float bound;

    HeapNode();
};

class Heap
{
public:
    vector<HeapNode *> heap;
    vector<HeapNode *> pool;
    int n;

    Heap();
    ~Heap();
    void add(HeapNode *nod);
    HeapNode *remove_top();
    bool is_empty();
    HeapNode *get_top();

private:
    void up(int p);
    void down(int p);

public:
    void reconstruct();
};
