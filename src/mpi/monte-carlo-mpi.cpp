#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <mpi.h>
#include <omp.h>

#ifdef __x86_64__
#include <x86intrin.h>
static inline uint64_t read_cycles() { return __rdtsc(); }
#else
static inline uint64_t read_cycles() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
#endif
static inline uint64_t budget_to_cycles(int megacycles) { return (uint64_t)megacycles * 1'000'000ULL; }

#include "../standard/common.hpp"
#include "../standard/evaluate.hpp"


static std::atomic<bool> exceeded_budget{false};
static uint64_t          g_cycle_start  = 0;
static uint64_t          g_cycle_budget = 0;
static int               g_rank         = 0;
static int               g_nranks       = 1;


double rollout(chess::Board state, int max_depth = 5) {
    static thread_local std::mt19937 rng(std::random_device{}());
    const chess::Color root_color = state.sideToMove();

    for (int depth = 0; depth < max_depth; ++depth) {
        if (state.isHalfMoveDraw() || state.isRepetition(2)) return 0.5;
        chess::Movelist moves;
        chess::movegen::legalmoves(moves, state);

        if (moves.empty()) {
            if (!state.inCheck()) return 0.5;
            else return (state.sideToMove() == root_color) ? 0.0 : 1.0;
        }

        state.makeMove(moves[rng() % moves.size()]);
    }

    int eval = engine_evaluate(state);
    if (state.sideToMove() != root_color) eval = -eval;
    constexpr double k = 0.003;
    return 1.0 / (1.0 + std::exp(-k * eval));
}


struct alignas(64) MCTSNode {
    chess::Move move;
    std::atomic<double> wins{0.0};
    std::atomic<int> visits{0};
    std::atomic<int> virtual_loss{0};

    std::vector<chess::Move> untried_moves;
    std::vector<MCTSNode*> children;

    std::atomic<size_t> next_untried_idx{0};
    std::atomic<bool> is_stable{false};

    MCTSNode* parent = nullptr;

    explicit MCTSNode(const chess::Board& b, chess::Move m = chess::Move::NO_MOVE, MCTSNode* p = nullptr)
        : move(m), parent(p) {
        chess::Movelist ml;
        chess::movegen::legalmoves(ml, b);

        untried_moves.assign(ml.begin(), ml.end());
        children.resize(untried_moves.size(), nullptr);

        if (untried_moves.empty())
            is_stable.store(true, std::memory_order_release);
    }

    MCTSNode* best_child(double c = 1.414) const {
        MCTSNode* best = nullptr;
        double best_score = -1e18;

        int p_v = visits.load(std::memory_order_relaxed) +
                  virtual_loss.load(std::memory_order_relaxed);
        double log_v = std::log(std::max(1.0, (double)p_v));

        for (const auto& child : children) {
            if (!child) continue;
            int cv = child->visits.load(std::memory_order_relaxed) + child->virtual_loss.load(std::memory_order_relaxed);
            if (cv == 0) return child;

            double q = 1.0 - (child->wins.load(std::memory_order_relaxed) / cv);
            double u = c * std::sqrt(log_v / cv);
            double score = q + u;

            if (score > best_score) {
                best_score = score;
                best = child;
            }
        }
        return best;
    }
};


struct NodePool {
    static constexpr size_t CHUNK_SIZE = 1 << 16;

    struct Chunk {
        alignas(64) MCTSNode* data;
        std::atomic<size_t> used;

        Chunk() : used(0) {
            // SAFE aligned allocation (C++-correct)
            data = static_cast<MCTSNode*>(
                ::operator new(
                    CHUNK_SIZE * sizeof(MCTSNode),
                    std::align_val_t(64)
                )
            );
        }

        ~Chunk() {
            ::operator delete(
                data,
                std::align_val_t(64)
            );
        }
    };

    std::vector<Chunk*> chunks;
    std::atomic<size_t> current{0};
    std::mutex grow_mutex;

    NodePool() {
        chunks.reserve(64);
        chunks.push_back(new Chunk());
    }

    MCTSNode* allocate(const chess::Board& b, chess::Move m, MCTSNode* parent) {
        size_t idx_chunk = current.load(std::memory_order_relaxed);
        Chunk* c = chunks[idx_chunk];

        size_t idx = c->used.fetch_add(1, std::memory_order_relaxed);

        if (idx < CHUNK_SIZE) {
            return new (&c->data[idx]) MCTSNode(b, m, parent);
        }

        std::lock_guard<std::mutex> lock(grow_mutex);

        idx_chunk = current.load(std::memory_order_relaxed);

        if (idx_chunk + 1 >= chunks.size()) {
            chunks.push_back(new Chunk());
        }

        current.store(idx_chunk + 1, std::memory_order_relaxed);
        c = chunks[idx_chunk + 1];

        size_t new_idx = c->used.fetch_add(1, std::memory_order_relaxed);

        return new (&c->data[new_idx]) MCTSNode(b, m, parent);
    }

    void reset() {
        for (Chunk* c : chunks) {
            c->used.store(0, std::memory_order_relaxed);
        }
        current.store(0, std::memory_order_relaxed);
    }

    ~NodePool() {
        for (Chunk* c : chunks) {
            delete c;
        }
    }
};


static MCTSNode* tree_descend(MCTSNode* node, chess::Board& board, NodePool& pool) {
    while (true) {
        node->virtual_loss.fetch_add(1, std::memory_order_relaxed);

        if (node->is_stable.load(std::memory_order_acquire)) {
            if (node->children.empty()) return node;
            MCTSNode* best = node->best_child();
            board.makeMove(best->move);
            node = best;
            continue;
        }

        size_t idx = node->next_untried_idx.fetch_add(1, std::memory_order_relaxed);

        if (idx < node->untried_moves.size()) {
            chess::Move m = node->untried_moves[idx];
            board.makeMove(m);
            MCTSNode* child = pool.allocate(board, m, node);
            node->children[idx] = child;
            if (idx + 1 == node->untried_moves.size()) {
                node->is_stable.store(true, std::memory_order_release);
            }

            child->virtual_loss.fetch_add(1, std::memory_order_relaxed);
            return child;
        }
    }
}


static void backprop(MCTSNode* node, double result) {
    while (node != nullptr) {
        node->visits.fetch_add(1, std::memory_order_relaxed);
        double old = node->wins.load(std::memory_order_relaxed);
        while (!node->wins.compare_exchange_weak(old, old + result, std::memory_order_relaxed));
        node->virtual_loss.fetch_sub(1, std::memory_order_relaxed);
        result = 1.0 - result;
        node = node->parent;
    }
}


static void run_simulations(MCTSNode& root, const chess::Board& board, int max_depth, int num_simulations, NodePool& pool) {
    #pragma omp parallel for schedule(static, 16)
    for (int i = 0; i < num_simulations; ++i) {
        chess::Board local_board = board;
        MCTSNode* leaf = tree_descend(&root, local_board, pool);
        double result = rollout(local_board, max_depth);
        backprop(leaf, result);
    }
}


static void run_simulations_persistent(MCTSNode& root, const chess::Board& board, int max_depth, NodePool& pool) {
    #pragma omp parallel
    {
        while (!exceeded_budget.load(std::memory_order_relaxed)) {
            chess::Board local_board = board;
            MCTSNode* leaf = tree_descend(&root, local_board, pool);
            double result = rollout(local_board, max_depth);
            backprop(leaf, result);

            if (g_cycle_budget && read_cycles() - g_cycle_start >= g_cycle_budget)
                exceeded_budget = true;
        }
    }
}


// Aggregate root-child visit/win counts across all ranks then pick best by visits.
// children[i] corresponds to untried_moves[i] (same deterministic move ordering on all ranks).
static chess::Move pick_best_mpi(MCTSNode& root) {
    int n = (int)root.children.size();

    std::vector<int>    local_visits(n, 0);
    std::vector<double> local_wins(n, 0.0);
    for (int i = 0; i < n; i++) {
        if (root.children[i]) {
            local_visits[i] = root.children[i]->visits.load(std::memory_order_relaxed);
            local_wins[i]   = root.children[i]->wins.load(std::memory_order_relaxed);
        }
    }

    std::vector<int>    total_visits(n, 0);
    std::vector<double> total_wins(n, 0.0);
    MPI_Allreduce(local_visits.data(), total_visits.data(), n, MPI_INT,    MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(local_wins.data(),   total_wins.data(),   n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    int    best_v        = -1;
    int    best_i        = 0;
    double best_win_rate = 0.0;
    for (int i = 0; i < n; i++) {
        if (total_visits[i] > best_v) {
            best_v        = total_visits[i];
            best_i        = i;
            best_win_rate = total_visits[i] > 0
                            ? 1.0 - (total_wins[i] / total_visits[i]) : 0.0;
        }
    }

    if (g_rank == 0) {
        printf("MCTS prefers move %s with %s rate %.2f%% (%d visits)\n",
               chess::uci::moveToUci(root.untried_moves[best_i]).c_str(),
               best_win_rate >= 0.5 ? "win" : "loss",
               std::max(best_win_rate, 1.0 - best_win_rate) * 100.0,
               best_v);
    }
    return root.untried_moves[best_i];
}


// Each rank runs depth as max rollout depth, splitting total simulations evenly.
static std::string search_depth(const chess::Board& board, int depth) {
    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);
    if (moves.empty()) return "";
    if (moves.size() == 1) return chess::uci::moveToUci(moves[0]);

    static thread_local NodePool pool;
    MCTSNode root(board);

    constexpr int TOTAL_SIMS = 10000;
    int base    = TOTAL_SIMS / g_nranks;
    int my_sims = base + (g_rank < TOTAL_SIMS % g_nranks ? 1 : 0);

    run_simulations(root, board, depth, my_sims, pool);

    chess::Move best = pick_best_mpi(root);
    return chess::uci::moveToUci(best);
}


// Each rank runs until its own timer fires, then all aggregate.
static std::string search_time(const chess::Board& board, int time_ms) {
    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);
    if (moves.empty()) return "";
    if (moves.size() == 1) return chess::uci::moveToUci(moves[0]);

    exceeded_budget.store(false, std::memory_order_relaxed);

    using clock = std::chrono::steady_clock;
    auto deadline = clock::now() + std::chrono::milliseconds(time_ms);
    std::thread timer([deadline]() {
        std::this_thread::sleep_until(deadline);
        exceeded_budget.store(true, std::memory_order_relaxed);
    });

    constexpr int SEARCH_DEPTH = 6;
    static thread_local NodePool pool;
    MCTSNode root(board);
    run_simulations_persistent(root, board, SEARCH_DEPTH, pool);
    timer.join();

    chess::Move best = pick_best_mpi(root);
    return chess::uci::moveToUci(best);
}


// Each rank runs until its own cycle budget fires, then all aggregate.
static std::string search_cycles(const chess::Board& board, int megacycle_budget) {
    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);
    if (moves.empty()) return "";
    if (moves.size() == 1) return chess::uci::moveToUci(moves[0]);

    exceeded_budget.store(false, std::memory_order_relaxed);
    g_cycle_start  = read_cycles();
    g_cycle_budget = budget_to_cycles(megacycle_budget);

    constexpr int SEARCH_DEPTH = 6;
    static thread_local NodePool pool;
    MCTSNode root(board);
    run_simulations_persistent(root, board, SEARCH_DEPTH, pool);
    g_cycle_budget = 0;

    chess::Move best = pick_best_mpi(root);
    return chess::uci::moveToUci(best);
}


int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_nranks);

    if (g_rank == 0) {
        fprintf(stderr, "[mpi-monte-carlo] %d ranks x %d OMP threads each\n",
                g_nranks, omp_get_max_threads());
    }

    constexpr int BUF_SIZE = 4096;
    char buf[BUF_SIZE];

    while (true) {
        // Rank 0 reads the next command and broadcasts to all ranks.
        int len = 0;
        if (g_rank == 0) {
            std::string line;
            if (!std::getline(std::cin, line) || line == "quit") {
                len = -1;
            } else {
                std::strncpy(buf, line.c_str(), BUF_SIZE - 1);
                buf[BUF_SIZE - 1] = '\0';
                len = (int)strlen(buf);
            }
        }

        MPI_Bcast(&len, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (len < 0) break; // All ranks need to quit so after broadcast
        MPI_Bcast(buf, BUF_SIZE, MPI_CHAR, 0, MPI_COMM_WORLD);

        // Parse: "<mode> <budget> <fen...>"
        std::string line_str(buf);
        std::istringstream ss(line_str);
        std::string mode;
        int         budget;
        std::string fen;

        ss >> mode >> budget;
        std::getline(ss, fen);
        if (!fen.empty() && fen[0] == ' ') fen = fen.substr(1);

        chess::Board board;
        if (!board.setFen(fen)) {
            if (g_rank == 0) std::cout << "error\n" << std::flush;
            continue;
        }

        std::string move;
        if      (mode == "depth")  move = search_depth(board, budget);
        else if (mode == "time")   move = search_time(board, budget);
        else if (mode == "cycles") move = search_cycles(board, budget);
        else { if (g_rank == 0) std::cout << "error\n" << std::flush; continue; }

        if (g_rank == 0) std::cout << move << "\n" << std::flush;
    }

    MPI_Finalize();
    return 0;
}
