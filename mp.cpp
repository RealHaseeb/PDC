#include <iostream>
#include <vector>
#include <queue>
#include <limits>
#include <algorithm>
#include <omp.h>
#include <mpi.h>
#include<metis.h>

using namespace std;

// Structure for an edge
struct Edge {
    int to;
    double weight;
};

// Structure for an edge change (insertion or deletion)
struct EdgeChange {
    int u, v;
    double weight;
    bool isInsertion;
};

// Graph class to manage adjacency list and SSSP tree
class Graph {
public:
    int n;                     // Number of vertices
    vector<vector<Edge>> adj;  // Adjacency list
    vector<double> dist;       // Distance from source
    vector<int> parent;        // Parent in SSSP tree

    Graph(int vertices) : n(vertices), adj(vertices), dist(vertices, numeric_limits<double>::infinity()), parent(vertices, -1) {}

    void addEdge(int u, int v, double weight) {
        adj[u].push_back({v, weight});
        adj[v].push_back({u, weight}); // Undirected graph
    }

    // Compute initial SSSP using Dijkstra's algorithm
    void computeInitialSSSP(int source) {
        dist[source] = 0;
        priority_queue<pair<double, int>, vector<pair<double, int>>, greater<>> pq;
        pq.emplace(0, source);

        while (!pq.empty()) {
            double d = pq.top().first;
            int u = pq.top().second;
            pq.pop();
            if (d > dist[u]) continue;
            #pragma omp parallel for
            for (size_t i = 0; i < adj[u].size(); ++i) {
                int v = adj[u][i].to;
                double w = adj[u][i].weight;
                double newDist = dist[u] + w;
                #pragma omp critical
                if (newDist < dist[v]) {
                    dist[v] = newDist;
                    parent[v] = u;
                    pq.emplace(newDist, v);
                }
            }
        }
    }
};

// Partition the graph using METIS
void partitionGraphWithMETIS(const Graph& graph, int nparts, vector<int>& partition) {
    idx_t nvtxs = graph.n;
    idx_t ncon = 1;  // Number of constraints
    idx_t objval;    // Edge-cut value returned by METIS
    vector<idx_t> xadj(nvtxs + 1);
    vector<idx_t> adjncy;

    // Build CSR-like structure for METIS
    for (int u = 0; u < nvtxs; ++u) {
        xadj[u] = adjncy.size();
        for (const auto& edge : graph.adj[u]) {
            adjncy.push_back(edge.to);
        }
    }
    xadj[nvtxs] = adjncy.size();

    partition.resize(nvtxs);
    METIS_PartGraphKway(&nvtxs, &ncon, xadj.data(), adjncy.data(), NULL, NULL, NULL, &nparts, NULL, NULL, NULL, &objval, partition.data());
}

// Distribute subgraph to an MPI process
Graph distributeSubgraph(int rank, int size, const Graph& globalGraph, const vector<int>& partition) {
    vector<vector<Edge>> localAdj(globalGraph.n);
    for (int u = 0; u < globalGraph.n; ++u) {
        if (partition[u] == rank) {
            localAdj[u] = globalGraph.adj[u];
        }
    }
    Graph localGraph(globalGraph.n);
    localGraph.adj = localAdj;
    localGraph.dist = globalGraph.dist;
    localGraph.parent = globalGraph.parent;
    return localGraph;
}

// Parallel SSSP update algorithm
void parallelUpdateSSSP(int rank, int size, Graph& localGraph, const vector<EdgeChange>& changes) {
    vector<bool> affected(localGraph.n, false);

    // Step 1: Identify affected vertices in parallel
    #pragma omp parallel for
    for (size_t i = 0; i < changes.size(); ++i) {
        const auto& change = changes[i];
        int u = change.u, v = change.v;
        double w = change.weight;

        if (change.isInsertion) {
            if (localGraph.dist[u] + w < localGraph.dist[v]) {
                #pragma omp critical
                {
                    localGraph.dist[v] = localGraph.dist[u] + w;
                    localGraph.parent[v] = u;
                    affected[v] = true;
                }
            }
            if (localGraph.dist[v] + w < localGraph.dist[u]) {
                #pragma omp critical
                {
                    localGraph.dist[u] = localGraph.dist[v] + w;
                    localGraph.parent[u] = v;
                    affected[u] = true;
                }
            }
        } else { // Deletion
            if (localGraph.parent[v] == u) {
                #pragma omp critical
                {
                    localGraph.dist[v] = numeric_limits<double>::infinity();
                    localGraph.parent[v] = -1;
                    affected[v] = true;
                }
            }
        }
    }

    // Step 2: Update SSSP tree iteratively
    bool globalUpdated = true;
    while (globalUpdated) {
        bool localUpdated = false;
        #pragma omp parallel for
        for (int u = 0; u < localGraph.n; ++u) {
            if (affected[u]) {
                for (const auto& edge : localGraph.adj[u]) {
                    int v = edge.to;
                    double w = edge.weight;
                    double newDist = localGraph.dist[u] + w;
                    #pragma omp critical
                    if (newDist < localGraph.dist[v]) {
                        localGraph.dist[v] = newDist;
                        localGraph.parent[v] = u;
                        affected[v] = true;
                        localUpdated = true;
                    }
                }
            }
        }
        // Synchronize across MPI processes
        MPI_Allreduce(&localUpdated, &globalUpdated, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);
        if (!globalUpdated) break;
    }
}

// Main function
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Step 1: Initialize global graph (example with 6 vertices)
    Graph globalGraph(6);
    if (rank == 0) {
        globalGraph.addEdge(0, 1, 4.0);
        globalGraph.addEdge(0, 2, 2.0);
        globalGraph.addEdge(1, 2, 1.0);
        globalGraph.addEdge(1, 3, 5.0);
        globalGraph.addEdge(2, 3, 8.0);
        globalGraph.addEdge(2, 4, 10.0);
        globalGraph.addEdge(3, 4, 2.0);
        globalGraph.addEdge(3, 5, 6.0);
        globalGraph.addEdge(4, 5, 3.0);

        // Compute initial SSSP
        globalGraph.computeInitialSSSP(0);
    }

    // Broadcast initial distances and parents
    MPI_Bcast(globalGraph.dist.data(), globalGraph.n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(globalGraph.parent.data(), globalGraph.n, MPI_INT, 0, MPI_COMM_WORLD);

    // Step 2: Partition the graph using METIS
    vector<int> partition;
    if (rank == 0) {
        partitionGraphWithMETIS(globalGraph, size, partition);
    }
    partition.resize(globalGraph.n);
    MPI_Bcast(partition.data(), globalGraph.n, MPI_INT, 0, MPI_COMM_WORLD);

    // Step 3: Distribute subgraphs
    Graph localGraph = distributeSubgraph(rank, size, globalGraph, partition);

    // Step 4: Simulate edge changes (example batch)
    vector<EdgeChange> changes;
    if (rank == 0) {
        changes.push_back({1, 3, 2.0, true});  // Insertion
        changes.push_back({2, 3, 0.0, false}); // Deletion
    }
    int changeSize = changes.size();
    MPI_Bcast(&changeSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) changes.resize(changeSize);
    MPI_Bcast(changes.data(), changeSize * sizeof(EdgeChange), MPI_BYTE, 0, MPI_COMM_WORLD);

    // Step 5: Update SSSP in parallel
    parallelUpdateSSSP(rank, size, localGraph, changes);

    // Step 6: Gather results (for verification)
    vector<double> globalDist(globalGraph.n);
    vector<int> globalParent(globalGraph.n);
    MPI_Reduce(localGraph.dist.data(), globalDist.data(), globalGraph.n, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(localGraph.parent.data(), globalParent.data(), globalGraph.n, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

    // Output results on rank 0
    if (rank == 0) {
        cout << "Updated SSSP Distances:\n";
        for (int i = 0; i < globalGraph.n; ++i) {
            cout << "Vertex " << i << ": Distance = " << globalDist[i] << ", Parent = " << globalParent[i] << "\n";
        }
    }

    MPI_Finalize();
    return 0;
}