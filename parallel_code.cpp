#include <vector>
#include <queue>
#include <limits>
#include <algorithm>
#include <omp.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <string>
#include <random>
#include <mpi.h>
#include <metis.h>

using namespace std;
using namespace std::chrono;

// Structure to represent an edge
struct Edge {
    int u, v;
    double weight;
    Edge(int _u, int _v, double _w) : u(_u), v(_v), weight(_w) {}
};

// Graph representation using CSR with METIS
class Graph {
public:
    int num_vertices;
    int local_num_vertices;  
    vector<int> row_ptr;     
    vector<int> col_idx;     
    vector<double> weights;   
    vector<int> parent;       
    vector<double> dist;      
    vector<bool> affected;   
    vector<bool> affected_del;
    vector<int> part;         
    vector<int> global_to_local; 
    vector<int> local_to_global; 
    int rank, size;          
    Graph(int n, int _rank, int _size) : num_vertices(n), rank(_rank), size(_size) {
        parent.resize(n, -1);
        dist.resize(n, numeric_limits<double>::infinity());
        affected.resize(n, false);
        affected_del.resize(n, false);
        global_to_local.resize(n, -1);
        row_ptr.push_back(0);
    }

    // Partition graph using METIS
    void partitionGraph(int nparts) {
        idx_t nvtxs = num_vertices;
        idx_t ncon = 1;
        vector<idx_t> xadj(row_ptr.begin(), row_ptr.end());
        vector<idx_t> adjncy(col_idx.begin(), col_idx.end());
        vector<idx_t> part_vec(num_vertices);
        idx_t objval;

        // Call METIS to partition the graph
        int ret = METIS_PartGraphKway(&nvtxs, &ncon, xadj.data(), adjncy.data(), NULL, NULL, NULL,
                                    &nparts, NULL, NULL, NULL, &objval, part_vec.data());
        if (ret != METIS_OK) {
            cerr << "METIS partitioning failed" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        part.assign(part_vec.begin(), part_vec.end());

        // Create global-to-local and local-to-global mappings
        int local_count = 0;
        global_to_local.clear();
        global_to_local.resize(num_vertices, -1);
        local_to_global.clear();
        for (int i = 0; i < num_vertices; ++i) {
            if (part[i] == rank) {
                global_to_local[i] = local_count++;
                local_to_global.push_back(i);
            }
        }
        local_num_vertices = local_count;

        // Rebuild CSR for local vertices
        vector<int> new_row_ptr = {0};
        vector<int> new_col_idx;
        vector<double> new_weights;
        for (int u : local_to_global) {
            int local_u = global_to_local[u];
            int start = row_ptr[u];
            int end = (u + 1 < row_ptr.size()) ? row_ptr[u + 1] : col_idx.size();
            for (int i = start; i < end; ++i) {
                new_col_idx.push_back(col_idx[i]);
                new_weights.push_back(weights[i]);
            }
            new_row_ptr.push_back(new_col_idx.size());
        }
        row_ptr = new_row_ptr;
        col_idx = new_col_idx;
        weights = new_weights;
    }

    void addEdge(int u, int v, double w) {
        if (u >= num_vertices || v >= num_vertices || u < 0 || v < 0) {
            cerr << "Rank " << rank << ": Invalid edge (" << u << ", " << v << ")" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (!isfinite(w) || w < 0) {
            cerr << "Rank " << rank << ": Invalid weight " << w << " for edge (" << u << ", " << v << ")" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        col_idx.push_back(v);
        weights.push_back(w);
        while (row_ptr.size() <= u + 1) {
            row_ptr.push_back(col_idx.size());
        }
    }

    long long insertEdgesFromFile(const string& filename) {
        auto start = high_resolution_clock::now();
        ifstream file(filename);
        if (!file.is_open()) {
            cerr << "Rank " << rank << ": Error opening file: " << filename << endl;
            return -1;
        }

        string line;
        while (getline(file, line)) {
            istringstream iss(line);
            int u, v;
            double w;
            if (iss >> u >> v >> w) {
                if (part[u] == rank) {
                    // Add to CSR
                    col_idx.push_back(v);
                    weights.push_back(w);
                    for (size_t i = u + 1; i < row_ptr.size(); ++i) {
                        row_ptr[i]++;
                    }
                    if (row_ptr.size() <= u + 1) {
                        row_ptr.push_back(col_idx.size());
                    }

                    int x = (dist[u] > dist[v]) ? v : u;
                    int y = (x == u) ? v : u;
                    if (dist[y] > dist[x] + w) {
                        dist[y] = dist[x] + w;
                        parent[y] = x;
                        affected[y] = true;
                    }
                }
            }
        }
        file.close();

        auto end = high_resolution_clock::now();
        return duration_cast<microseconds>(end - start).count();
    }

    long long deleteEdgesFromFile(const string& filename) {
        auto start = high_resolution_clock::now();
        ifstream file(filename);
        if (!file.is_open()) {
            cerr << "Rank " << rank << ": Error opening file: " << filename << endl;
            return -1;
        }

        string line;
        while (getline(file, line)) {
            istringstream iss(line);
            int u, v;
            if (iss >> u >> v) {
                if (part[u] == rank) {
                    int start_idx = row_ptr[u];
                    int end_idx = (u + 1 < row_ptr.size()) ? row_ptr[u + 1] : col_idx.size();
                    for (int i = start_idx; i < end_idx; ++i) {
                        if (col_idx[i] == v) {
                            // Remove edge by shifting elements
                            col_idx.erase(col_idx.begin() + i);
                            weights.erase(weights.begin() + i);
                            for (size_t j = u + 1; j < row_ptr.size(); ++j) {
                                row_ptr[j]--;
                            }
                            if (parent[v] == u || parent[u] == v) {
                                int y = (dist[u] > dist[v]) ? u : v;
                                dist[y] = numeric_limits<double>::infinity();
                                affected_del[y] = true;
                                affected[y] = true;
                                parent[y] = -1;
                            }
                            break;
                        }
                    }
                }
            }
        }
        file.close();

        auto end = high_resolution_clock::now();
        return duration_cast<microseconds>(end - start).count();
    }

    void synchronizeDistances() {
        vector<double> send_dist(num_vertices);
        vector<double> recv_dist(num_vertices);
        
        for (int i = 0; i < num_vertices; ++i) {
            send_dist[i] = (part[i] == rank) ? dist[i] : numeric_limits<double>::infinity();
        }

        MPI_Allreduce(send_dist.data(), recv_dist.data(), num_vertices, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

        for (int i = 0; i < num_vertices; ++i) {
            if (part[i] == rank) {
                dist[i] = recv_dist[i];
            }
        }
    }
};

void initializeSSSP(Graph& G, int source) {
    G.dist[source] = 0;
    priority_queue<pair<double, int>, vector<pair<double, int>>, greater<>> pq;
    pq.push({0, source});

    while (!pq.empty()) {
        int u = pq.top().second;
        double d = pq.top().first;
        pq.pop();

        if (d > G.dist[u]) continue;

        int start = G.row_ptr[u];
        int end = (u + 1 < G.row_ptr.size()) ? G.row_ptr[u + 1] : G.col_idx.size();
        for (int i = start; i < end; ++i) {
            int v = G.col_idx[i];
            double w = G.weights[i];

            if (G.dist[u] + w < G.dist[v]) {
                G.dist[v] = G.dist[u] + w;
                G.parent[v] = u;
                pq.push({G.dist[v], v});
            }
        }
    }
    G.synchronizeDistances();
}

void processChangedEdges(Graph& G, const vector<Edge>& deletions, const vector<Edge>& insertions) {
    // Process deletions
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < deletions.size(); ++i) {
        int u = deletions[i].u;
        int v = deletions[i].v;
        if (G.part[u] != G.rank) 
        continue;  

        bool in_tree = (G.parent[v] == u || G.parent[u] == v);
        if (in_tree) {
            int y = (G.dist[u] > G.dist[v]) ? u : v;

            #pragma omp critical
            {
                G.dist[y] = numeric_limits<double>::infinity();
                G.affected_del[y] = true;
                G.affected[y] = true;
                G.parent[y] = -1;
            }
        }
    }

    // Process insertions
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < insertions.size(); ++i) {
        int u = insertions[i].u;
        int v = insertions[i].v;
        double w = insertions[i].weight;
        if (G.part[u] != G.rank)
         continue;  

        int x = (G.dist[u] > G.dist[v]) ? v : u;
        int y = (x == u) ? v : u;

        if (G.dist[y] > G.dist[x] + w) {
            #pragma omp critical
            {
                G.dist[y] = G.dist[x] + w;
                G.parent[y] = x;
                G.affected[y] = true;
            }
        }
    }
    G.synchronizeDistances();
}

// Algorithm 4: Asynchronous Update of SSSP
void asynchronousUpdate(Graph& G, int async_level) {
    bool change = true;
    while (change) {
        change = false;
        // Process deletion-affected vertices
        #pragma omp parallel for schedule(dynamic)
        for (int v = 0; v < G.num_vertices; ++v) {
            if (G.part[v] != G.rank) 
            continue; 
            if (G.affected_del[v]) {
                queue<int> Q;
                Q.push(v);
                int level = 0;

                while (!Q.empty() && level <= async_level) {
                    int x = Q.front();
                    Q.pop();

                    // Find children in SSSP tree
                    for (int i = 0; i < G.num_vertices; ++i) {
                        if (G.parent[i] == x) {
                            #pragma omp critical
                            {
                                G.affected[i] = true;
                                G.affected_del[i] = true;
                                G.dist[i] = numeric_limits<double>::infinity();
                                G.parent[i] = -1;
                                change = true;
                            }
                            if (level < async_level) {
                                Q.push(i);
                            }
                        }
                    }
                    level++;
                }
            }
        }


        change = true;
        while (change) {
            change = false;
            #pragma omp parallel for schedule(dynamic)
            for (int v = 0; v < G.num_vertices; ++v) {
                if (G.part[v] != G.rank) continue;
                if (G.affected[v]) {
                    G.affected[v] = false;
                    queue<int> Q;
                    Q.push(v);
                    int level = 0;

                    while (!Q.empty() && level <= async_level) {
                        int x = Q.front();
                        Q.pop();
                        int start = G.row_ptr[x];
                        int end = (x + 1 < G.row_ptr.size()) ? G.row_ptr[x + 1] : G.col_idx.size();
                        for (int i = start; i < end; ++i) {
                            int n = G.col_idx[i];
                            double w = G.weights[i];

                            // Update distances
                            if (G.dist[x] > G.dist[n] + w) {
                                #pragma omp critical
                                {
                                    G.dist[x] = G.dist[n] + w;
                                    G.parent[x] = n;
                                    G.affected[x] = true;
                                    change = true;
                                }
                                if (level < async_level) {
                                    Q.push(x);
                                }
                            }
                            if (G.dist[n] > G.dist[x] + w) {
                                #pragma omp critical
                                {
                                    G.dist[n] = G.dist[x] + w;
                                    G.parent[n] = x;
                                    G.affected[n] = true;

                                    change = true;
                                }
                                if (level < async_level) {
                                    Q.push(n);
                                }
                            }
                        }
                        level++;
                    }
                }
            }
            G.synchronizeDistances();
        }
    }
}

// Process changes in batches
long long updateSSSP(Graph& G, const vector<Edge>& deletions, const vector<Edge>& insertions, int batch_size, int async_level) {
    auto start = high_resolution_clock::now();

    for (size_t i = 0; i < max(deletions.size(), insertions.size()); i += batch_size) {
        vector<Edge> del_batch;
        vector<Edge> ins_batch;

        // Create batches
        for (size_t j = i; j < min(i + batch_size, deletions.size()); ++j) {
            del_batch.push_back(deletions[j]);
        }
        for (size_t j = i; j < min(i + batch_size, insertions.size()); ++j) {
            ins_batch.push_back(insertions[j]);
        }

        // Process batch
        processChangedEdges(G, del_batch, ins_batch);
        asynchronousUpdate(G, async_level);
    }

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    return duration.count();
}

// Print SSSP tree
void printSSSP(const Graph& G) {
    if (G.rank != 0) return;  // Only print from rank 0
    
    cout << "SSSP Tree:\n";
    for (int i = 0; i < G.num_vertices; ++i) {
        cout << "Vertex " << i << ": Distance = " 
             << (G.dist[i] == numeric_limits<double>::infinity() ? "INF" : to_string(G.dist[i]))
             << ", Parent = " << G.parent[i] << endl;
    }
}

// Function to read graph from file and collect edges
int readGraphFromFile(const string& filename, Graph& G, vector<Edge>& all_edges) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        return -1;
    }

    string line;
    int max_vertex = 0;

    getline(file, line); 
    getline(file, line); 

    while (getline(file, line)) {
        istringstream iss(line);
        int u, v;
        double w;
        if (iss >> u >> v >> w) {
            all_edges.emplace_back(u, v, w);
            max_vertex = max({max_vertex, u, v});
        }
    }
    file.close();

    G.num_vertices = max_vertex + 1;
    G.parent.resize(G.num_vertices, -1);
    G.dist.resize(G.num_vertices, numeric_limits<double>::infinity());
    G.affected.resize(G.num_vertices, false);
    G.affected_del.resize(G.num_vertices, false);
    G.part.resize(G.num_vertices, -1);

    for (const auto& edge : all_edges) {
        G.addEdge(edge.u, edge.v, edge.weight);
    }

    while (G.row_ptr.size() < G.num_vertices + 1) {
        G.row_ptr.push_back(G.col_idx.size());
    }

    return 0;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
  


    random_device rd;
    mt19937 gen(rd() + rank);

    Graph G(0, rank, size);
    vector<Edge> all_edges;

    if (readGraphFromFile("weighted_edge_list.txt", G, all_edges) != 0) {
        if (rank == 0) {
            cerr << "Failed to read graph\n";
        }
        MPI_Finalize();
        return 1;
    }

    G.partitionGraph(size);

    initializeSSSP(G, 0);

    long long total_delete_time = 0;
    total_delete_time = G.deleteEdgesFromFile("deletions.txt");
    if (rank == 0) {
        cout << "Total deletion time: " << total_delete_time << " microseconds\n";
    }

    long long total_insert_time = 0;
    total_insert_time = G.insertEdgesFromFile("insertions.txt");
    if (rank == 0) {
        cout << "Total insertion time: " << total_insert_time << " microseconds\n";
    }

    MPI_Finalize();
    return 0;
}
