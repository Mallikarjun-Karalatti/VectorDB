#include "engine_c_api.h"
#include "hnswlib.h"

#include <mutex>
#include <string>
#include <queue>
#include <vector>
#include <iostream>

struct vektor_index {
    hnswlib::SpaceInterface<float>* space;
    hnswlib::HierarchicalNSW<float>* appr_alg;
    int dim;
    int max_elements;
    int metric;
    std::mutex add_mutex;
};

extern "C" {

vektor_index_t* vektor_init(int dim, int max_elements, int M, int ef_construction, int metric) {
    if (dim <= 0 || max_elements <= 0) {
        return nullptr;
    }

    try {
        vektor_index_t* idx = new vektor_index_t();
        idx->dim = dim;
        idx->max_elements = max_elements;
        idx->metric = metric;

        if (metric == VEKTOR_METRIC_COSINE) {
            idx->space = new hnswlib::InnerProductSpace(static_cast<size_t>(dim));
        } else {
            idx->space = new hnswlib::L2Space(static_cast<size_t>(dim));
        }

        size_t m_val = (M > 0) ? static_cast<size_t>(M) : 16;
        size_t ef_c = (ef_construction > 0) ? static_cast<size_t>(ef_construction) : 200;

        idx->appr_alg = new hnswlib::HierarchicalNSW<float>(
            idx->space,
            static_cast<size_t>(max_elements),
            m_val,
            ef_c
        );

        return idx;
    } catch (const std::exception& e) {
        std::cerr << "vektor_init error: " << e.what() << std::endl;
        return nullptr;
    } catch (...) {
        return nullptr;
    }
}

int vektor_add(vektor_index_t* index, uint64_t id, const float* vec) {
    if (!index || !index->appr_alg || !vec) {
        return -1;
    }

    try {
        std::lock_guard<std::mutex> lock(index->add_mutex);
        index->appr_alg->addPoint(vec, static_cast<hnswlib::labeltype>(id));
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "vektor_add error: " << e.what() << std::endl;
        return -1;
    } catch (...) {
        return -1;
    }
}

int vektor_search(vektor_index_t* index, const float* query, int k, int ef_search, uint64_t* out_ids, float* out_distances) {
    if (!index || !index->appr_alg || !query || k <= 0 || !out_ids || !out_distances) {
        return -1;
    }

    try {
        if (ef_search > 0) {
            index->appr_alg->setEf(static_cast<size_t>(ef_search));
        }

        std::priority_queue<std::pair<float, hnswlib::labeltype>> result =
            index->appr_alg->searchKnn(query, static_cast<size_t>(k));

        int count = static_cast<int>(result.size());
        // searchKnn returns max-heap (largest distance at top).
        // Populate backwards so output is sorted nearest to farthest (ascending distance).
        for (int i = count - 1; i >= 0; --i) {
            auto item = result.top();
            result.pop();
            out_ids[i] = static_cast<uint64_t>(item.second);
            out_distances[i] = item.first;
        }

        return count;
    } catch (const std::exception& e) {
        std::cerr << "vektor_search error: " << e.what() << std::endl;
        return -1;
    } catch (...) {
        return -1;
    }
}

void vektor_save(vektor_index_t* index, const char* path) {
    if (!index || !index->appr_alg || !path) {
        return;
    }

    try {
        std::lock_guard<std::mutex> lock(index->add_mutex);
        index->appr_alg->saveIndex(std::string(path));
    } catch (const std::exception& e) {
        std::cerr << "vektor_save error: " << e.what() << std::endl;
    }
}

vektor_index_t* vektor_load(const char* path, int dim, int max_elements, int metric) {
    if (!path || dim <= 0 || max_elements <= 0) {
        return nullptr;
    }

    try {
        vektor_index_t* idx = new vektor_index_t();
        idx->dim = dim;
        idx->max_elements = max_elements;
        idx->metric = metric;

        if (metric == VEKTOR_METRIC_COSINE) {
            idx->space = new hnswlib::InnerProductSpace(static_cast<size_t>(dim));
        } else {
            idx->space = new hnswlib::L2Space(static_cast<size_t>(dim));
        }

        idx->appr_alg = new hnswlib::HierarchicalNSW<float>(
            idx->space,
            std::string(path),
            false,
            static_cast<size_t>(max_elements)
        );

        return idx;
    } catch (const std::exception& e) {
        std::cerr << "vektor_load error: " << e.what() << std::endl;
        return nullptr;
    } catch (...) {
        return nullptr;
    }
}

void vektor_free(vektor_index_t* index) {
    if (!index) return;

    try {
        if (index->appr_alg) {
            delete index->appr_alg;
            index->appr_alg = nullptr;
        }
        if (index->space) {
            delete index->space;
            index->space = nullptr;
        }
        delete index;
    } catch (...) {
    }
}

size_t vektor_count(vektor_index_t* index) {
    if (!index || !index->appr_alg) return 0;
    try {
        return index->appr_alg->cur_element_count;
    } catch (...) {
        return 0;
    }
}

int vektor_dim(vektor_index_t* index) {
    if (!index) return 0;
    return index->dim;
}

} // extern "C"
