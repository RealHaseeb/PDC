#include <mpi.h>
#include <omp.h>
#include <metis.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <queue>
#include <limits>
#include <algorithm>

#define INF std::numeric_limits<int>::max()

using namespace std;

typedef unordered_map<int, vector<int>> Graph;

Graph readGraphFromCustomFormat(const string& filename, int& numVertices) {
    ifstream infile(filename);
    Graph graph;
    string line;
    numVertices = 0;

    while (getline(infile, line)) {
        if (line.empty()) continue;
        istringstream iss(line);
        int u;
        char colon;
        iss >> u >> colon;
        string neighbors;
        getline(iss, neighbors);

        istringstream neighborStream(neighbors);
        string token;
        while (getline(neighborStream, token, ',')) {
            int v = stoi(token);
            graph[u].push_back(v);
            graph[v]; // Ensure v exists
            numVertices = max({numVertices, u, v});
        }
    }

    numVertices += 1; // Since nodes are 0-based
    return graph;
}

void convertGraphToCSR(const Graph& graph, vector<idx_t>& xadj, vector<idx_t>& adjncy) {
    int n = graph.size();
    xadj.resize(n + 1);
    int edgeCount = 0;

    for (int i = 0; i < n; ++i) {
        xadj[i] = edgeCount;
        if (graph.find(i) != graph.end()) {
            const auto& neighbors = graph.at(i);
            adjncy.insert(adjncy.end(), neighbors.begin(), neighbors.end());
            edgeCount += neighbors.size();
        }
    }
    xadj[n] = edgeCount;
}

vector<int> partitionGraphWithMetis(const Graph& graph, int nparts) {
    idx_t n = graph.size();
    vector<idx_t> xadj, adjncy;
    convertGraphToCSR(graph, xadj, adjncy);

    vector<idx_t> part(n);
    idx_t objval;
    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);

    // Call METIS_PartGraphKway to partition the graph
    int metisStatus = METIS_PartGraphKway(&n, &xadj[0], &adjncy[0], nullptr, nullptr,
                                          nullptr, nullptr, &nparts, nullptr, nullptr,
                                          options, &objval, &part[0]);

    // Check if METIS_PartGraphKway was successful
    if (metisStatus != METIS_OK) {
        cerr << "METIS partitioning failed!" << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    return vector<int>(part.begin(), part.end());
}

Graph extractSubgraph(const Graph& graph, const vector<int>& part, int rank) {
    Graph subgraph;
    for (const auto& [u, neighbors] : graph) {
        if (part[u] == rank) {
            for (int v : neighbors) {
                if (part[v] == rank) {
                    subgraph[u].push_back(v);
                }
            }
        }
    }
    return subgraph;
}

unordered_map<int, int> parallelUpdateSSSP(const Graph& subgraph, int source) {
    unordered_map<int, int> dist;
    for (const auto& [node, _] : subgraph)
        dist[node] = INF;
    dist[source] = 0;

    queue<int> q;
    q.push(source);

    while (!q.empty()) {
        int u = q.front();
        q.pop();

        for (int v : subgraph.at(u)) {
            if (dist[v] > dist[u] + 1) {
                dist[v] = dist[u] + 1;
                q.push(v);
            }
        }
    }

    return dist;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int numVertices;
    Graph graph;
    vector<int> partition;

    // Rank 0 reads the graph
    if (rank == 0) {
        graph = readGraphFromCustomFormat("graph.txt", numVertices);
        partition = partitionGraphWithMetis(graph, size);
    }

    // Broadcast the number of vertices to all ranks
    MPI_Bcast(&numVertices, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Resize partition vector after broadcasting numVertices
    partition.resize(numVertices);

    // Broadcast the partition to all ranks
    MPI_Bcast(partition.data(), numVertices, MPI_INT, 0, MPI_COMM_WORLD);

    // Rank 0 will read the graph, while other ranks get the partitioned data
    if (rank != 0) {
        graph = readGraphFromCustomFormat("graph.txt", numVertices);
    }

    // Extract subgraph for this rank
    Graph subgraph = extractSubgraph(graph, partition, rank);
    int source = 1; // change as needed

    // Run parallel SSSP on the subgraph
    unordered_map<int, int> localDistances = parallelUpdateSSSP(subgraph, source);

    // Output the results from each rank
    for (const auto& [node, dist] : localDistances) {
        cout << "[Rank " << rank << "] Distance to node " << node << ": " << dist << endl;
    }

    MPI_Finalize();
    return 0;
}

