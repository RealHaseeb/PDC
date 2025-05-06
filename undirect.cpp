#include <vector>
#include <queue>
#include <limits>
#include <algorithm>
#include <omp.h>
#include <iostream>
#include <chrono>
#include <fstream>
#include <sstream>
#include <string>

using namespace std;
using namespace std::chrono;

// Structure to represent an edge
struct Edge {
    int u, v;
    double weight;
    Edge(int _u, int _v, double _w) : u(_u), v(_v), weight(_w) {}
};

// Graph representation using CSR
class Graph {
public:
    int num_vertices;
    vector<int> row_ptr;      // CSR row pointers
    vector<int> col_idx;      // CSR column indices
    vector<double> weights;   // Edge weights
    vector<int> parent;       // Parent array for SSSP tree
    vector<double> dist;      // Distance from source
    vector<bool> affected;    // Affected vertices
    vector<bool> affected_del;// Affected by deletion

    Graph(int n) : num_vertices(n) {
        parent.resize(n, -1);
        dist.resize(n, numeric_limits<double>::infinity());
        affected.resize(n, false);
        affected_del.resize(n, false);
        row_ptr.push_back(0);
    }

    // Add edge to CSR (initial graph construction, undirected)
    void addEdge(int u, int v, double w) {
        // Add (u, v, w)
        col_idx.push_back(v);
        weights.push_back(w);
        while (row_ptr.size() <= u + 1) {
            row_ptr.push_back(col_idx.size());
        }
        // Add (v, u, w) for undirected graph
        col_idx.push_back(u);
        weights.push_back(w);
        while (row_ptr.size() <= v + 1) {
            row_ptr.push_back(col_idx.size());
        }
    }

    // Insert a new edge and update affected vertices, return execution time in microseconds
    long long insertEdge(int u, int v, double w) {
        auto start = high_resolution_clock::now();

        // Add (u, v, w)
        col_idx.push_back(v);
        weights.push_back(w);
        for (size_t i = u + 1; i < row_ptr.size(); ++i) {
            row_ptr[i]++;
        }
        if (row_ptr.size() <= u + 1) {
            row_ptr.push_back(col_idx.size());
        }
        // Add (v, u, w) for undirected graph
        col_idx.push_back(u);
        weights.push_back(w);
        for (size_t i = v + 1; i < row_ptr.size(); ++i) {
            row_ptr[i]++;
        }
        if (row_ptr.size() <= v + 1) {
            row_ptr.push_back(col_idx.size());
        }

        // Check if edge affects SSSP tree
        int x = (dist[u] > dist[v]) ? v : u;
        int y = (x == u) ? v : u;
        if (dist[y] > dist[x] + w) {
            dist[y] = dist[x] + w;
            parent[y] = x;
            affected[y] = true;
        }
        // Check reverse direction
        if (dist[u] > dist[v] + w) {
            dist[u] = dist[v] + w;
            parent[u] = v;
            affected[u] = true;
        }

        auto end = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end - start);
        return duration.count();
    }

    // Delete an edge and update affected vertices, return execution time in microseconds
    long long deleteEdge(int u, int v) {
        auto start = high_resolution_clock::now();

        bool found = false;
        // Delete (u, v)
        int start_idx = row_ptr[u];
        int end_idx = (u + 1 < row_ptr.size()) ? row_ptr[u + 1] : col_idx.size();
        for (int i = start_idx; i < end_idx; ++i) {
            if (col_idx[i] == v) {
                col_idx.erase(col_idx.begin() + i);
                weights.erase(weights.begin() + i);
                for (size_t j = u + 1; j < row_ptr.size(); ++j) {
                    row_ptr[j]--;
                }
                found = true;
                break;
            }
        }
        // Delete (v, u)
        start_idx = row_ptr[v];
        end_idx = (v + 1 < row_ptr.size()) ? row_ptr[v + 1] : col_idx.size();
        for (int i = start_idx; i < end_idx; ++i) {
            if (col_idx[i] == u) {
                col_idx.erase(col_idx.begin() + i);
                weights.erase(weights.begin() + i);
                for (size_t j = v + 1; j < row_ptr.size(); ++j) {
                    row_ptr[j]--;
                }
                found = true;
                break;
            }
        }

        if (found) {
            // Check if edge was in SSSP tree
            if (parent[v] == u || parent[u] == v) {
                int y = (dist[u] > dist[v]) ? u : v;
                dist[y] = numeric_limits<double>::infinity();
                affected_del[y] = true;
                affected[y] = true;
                parent[y] = -1;
            }
        }

        auto end = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end - start);
        return duration.count();
    }
};

// Load graph from edge list file
bool loadGraphFromFile(Graph& G, const string& filename, int& num_vertices) {
    ifstream file(filename);
    if (!file.is_open()) {
        cout << "Error: Could not open file " << filename << endl;
        return false;
    }

    string line;
    vector<Edge> edges;
    num_vertices = 0;

    // Read file
    while (getline(file, line)) {
        // Skip comments or empty lines
        if (line.empty() || line[0] == '%' || line[0] == '#') {
            continue;
        }

        istringstream iss(line);
        int u, v;
        double w = 1.0; // Default weight if not provided
        if (!(iss >> u >> v)) {
            continue; // Skip malformed lines
        }
        iss >> w; // Read weight if present

        // Adjust for 1-based indexing (convert to 0-based)
        u--;
        v--;
        num_vertices = max(num_vertices, max(u, v) + 1);
        edges.emplace_back(u, v, w);
    }
    file.close();

    // Resize graph if necessary
    if (num_vertices > G.num_vertices) {
        G.num_vertices = num_vertices;
        G.parent.resize(num_vertices, -1);
        G.dist.resize(num_vertices, numeric_limits<double>::infinity());
        G.affected.resize(num_vertices, false);
        G.affected_del.resize(num_vertices, false);
    }

    // Add edges (undirected)
    for (const auto& e : edges) {
        G.addEdge(e.u, e.v, e.weight);
    }

    return true;
}

// Function to initialize SSSP tree using Dijkstra's algorithm
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
}

// Algorithm 2: Identify Affected Vertices
void processChangedEdges(Graph& G, const vector<Edge>& deletions, const vector<Edge>& insertions) {
    // Process deletions
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < deletions.size(); ++i) {
        int u = deletions[i].u;
        int v = deletions[i].v;
        // Check if edge is in SSSP tree
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
}

// Algorithm 4: Asynchronous Update of SSSP
void asynchronousUpdate(Graph& G, int async_level) {
    bool change = true;
    while (change) {
        change = false;
        // Process deletion-affected vertices
        #pragma omp parallel for schedule(dynamic)
        for (int v = 0; v < G.num_vertices; ++v) {
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

        // Process affected vertices
        change = true;
        while (change) {
            change = false;
            #pragma omp parallel for schedule(dynamic)
            for (int v = 0; v < G.num_vertices; ++v) {
                if (G.affected[v]) {
                    G.affected[v] = false;
                    queue<int> Q;
                    Q.push(v);
                    int level = 0;

                    while (!Q.empty() && level <= async_level) {
                        int x = Q.front();
                        Q.pop();

                        // Check neighbors
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
        }
    }
}

// Process changes in batches
void updateSSSP(Graph& G, const vector<Edge>& deletions, const vector<Edge>& insertions, int batch_size, int async_level) {
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
}

// Print SSSP tree
void printSSSP(const Graph& G) {
    cout << "SSSP Tree:\n";
    for (int i = 0; i < G.num_vertices; ++i) {
        cout << "Vertex " << i << ": Distance = " << (G.dist[i] == numeric_limits<double>::infinity() ? "INF" : to_string(G.dist[i]))
             << ", Parent = " << G.parent[i] << endl;
    }
}

// Main function for testing
int main() {
    string filename = "path/to/your/dataset.txt"; 
    int num_vertices = 0;

    Graph G(1000); // Adjust based on expected size

    // Load graph from file
    cout << "Loading graph from " << filename << "...\n";
    if (!loadGraphFromFile(G, filename, num_vertices)) {
        return 1;
    }
    cout << "Graph loaded: " << num_vertices << " vertices\n";

    // Initialize SSSP tree from source vertex 0
    cout << "Initializing SSSP tree...\n";
    initializeSSSP(G, 0);
    printSSSP(G);

    // Define changes (example insertions and deletions)
    vector<Edge> deletions, insertions;
    // Example insertions: Add new edges (adjust based on your graph)
    insertions.push_back(Edge(0, 1, 2.0));  // Example: Shorter path
    insertions.push_back(Edge(2, 3, 3.0));  // Example: New connection
    // Example deletions: Remove existing edges (ensure they exist in the dataset)
    deletions.push_back(Edge(1, 2, 1.0));   // Example: Remove edge
    deletions.push_back(Edge(3, 4, 1.0));   // Example: Remove edge

    // Apply changes directly using insertEdge and deleteEdge
    long long total_insert_time = 0;
    cout << "\nApplying insertions...\n";
    for (const auto& e : insertions) {
        cout << "Inserting edge (" << e.u << ", " << e.v << ", " << e.weight << ")";
        long long time_taken = G.insertEdge(e.u, e.v, e.weight);
        total_insert_time += time_taken;
        cout << " - Time: " << time_taken << " microseconds\n";
    }
    cout << "Total insertion time: " << total_insert_time << " microseconds\n";

    long long total_delete_time = 0;
    cout << "\nApplying deletions...\n";
    for (const auto& e : deletions) {
        cout << "Deleting edge (" << e.u << ", " << e.v << ")";
        long long time_taken = G.deleteEdge(e.u, e.v);
        total_delete_time += time_taken;
        if (time_taken >= 0) {
            cout << " - Time: " << time_taken << " microseconds\n";
        } else {
            cout << " - Edge not found\n";
        }
    }
    cout << "Total deletion time: " << total_delete_time << " microseconds\n";

    // Update SSSP with batch size 10 and async level 2
    cout << "\nUpdating SSSP tree...\n";
    updateSSSP(G, deletions, insertions, 10, 2);
    printSSSP(G);

    return 0;
}