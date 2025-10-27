#include "heap.h"

bool cmp_hn(HeapNode *n1, HeapNode *n2)
{
    if (n1->bound > n2->bound + 1e-6f)
        return true;
    if (n1->bound < n2->bound - 1e-6f)
        return false;
    if (n1->n > n2->n)
        return true;
    if (n1->n < n2->n)
        return false;
    if (n1->last_pos > n2->last_pos)
        return true;
    return false;
}

HeapNode::HeapNode()
{
    prev = NULL;
    last_pos = -1;
    n = 0;
    score = 0.0f;
    bound = 0.0f;
}

Heap::Heap()
{
    heap.push_back(NULL);
    n = 0;
}

Heap::~Heap()
{
    for (int i = 0; i < (int)pool.size(); ++i)
        delete pool[i];
}

void Heap::up(int p)
{
    HeapNode *tmp = heap[p];

    for (int i = p >> 1; i >= 1; i >>= 1)
    {
        if (cmp_hn(heap[i], tmp))
            break;
        heap[p] = heap[i];
        p = i;
    }

    heap[p] = tmp;
}

void Heap::down(int p)
{
    HeapNode *tmp = heap[p];

    for (int i = p << 1; i <= n; i <<= 1)
    {
        if (i + 1 <= n && cmp_hn(heap[i + 1], heap[i]))
            ++i;
        if (cmp_hn(tmp, heap[i]))
            break;
        heap[p] = heap[i];
        p = i;
    }

    heap[p] = tmp;
}

void Heap::add(HeapNode *nod)
{
    pool.push_back(nod);
    if (n == heap.size() - 1)
        heap.push_back(nod);
    else
        heap[n + 1] = nod;
    ++n;
    up(n);
}

HeapNode *Heap::remove_top()
{
    if (n == 0)
        return NULL;
    HeapNode *tmp = heap[1];
    heap[1] = heap[n];
    --n;
    down(1);
    return tmp;
}

HeapNode *Heap::get_top()
{
    if (n == 0)
        return NULL;
    return heap[1];
}

bool Heap::is_empty()
{
    return n == 0;
}

void Heap::reconstruct()
{
    for (int i = 1; i <= n; ++i)
        up(i);
}