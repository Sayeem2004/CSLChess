#include <cstring>
#include <memory>
#include <random>
#include <atomic>
#include <mutex>
#include <omp.h>
#include <thread>
#include <pthread.h>

#include "common.hpp"
#include "evaluate.hpp"

#ifdef USE_PAPI
#include <papi.h>
static std::once_flag papi_init_flag;
static void init_papi() { PAPI_library_init(PAPI_VER_CURRENT); }
#endif
static std::atomic<bool> exceeded_budget{false};

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
        chunks.reserve(32);
        chunks.push_back(new Chunk());
    }

    MCTSNode* allocate(const chess::Board& b,
                       chess::Move m,
                       MCTSNode* parent)
    {
        size_t idx_chunk = current.load(std::memory_order_relaxed);
        Chunk* c = chunks[idx_chunk];

        // fast path: atomic bump inside chunk
        size_t idx = c->used.fetch_add(1, std::memory_order_relaxed);

        if (idx < CHUNK_SIZE) {
            return new (&c->data[idx]) MCTSNode(b, m, parent);
        }

        // slow path: need new chunk
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
        }
    }
}

static chess::Move pick_best(MCTSNode& root) {
    int most_visits = -1;
    chess::Move best_move = chess::Move::NO_MOVE;
    double expected_win_rate = 0.0;

    for (const auto& child : root.children) {
        int child_v = child->visits.load();
        if (child_v > most_visits) {
            most_visits = child_v;
            best_move = child->move;
            expected_win_rate = 1.0 - (child->wins.load() / child_v);
        }
    }

    // printf("MCTS prefers move %s with %s rate %.2f%% (%d visits)\n",
    //        chess::uci::moveToUci(best_move).c_str(),
    //        expected_win_rate >= 0.5 ? "win" : "loss",
    //        std::max(expected_win_rate, 1.0 - expected_win_rate) * 100.0,
    //        most_visits);
    return best_move;
}

chess::Move monte_carlo_search(const chess::Board& board, int max_depth, int num_simulations) {
    MCTSNode root(board);
    static thread_local NodePool pool;
    run_simulations(root, board, max_depth, num_simulations, pool);
    return pick_best(root);
}

int best_move_monte_carlo_depth(const char* fen, int depth, char* out_move, int out_len) {
    chess::Board board;
    if (!board.setFen(fen)) return -2;
    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);
    if (moves.empty()) return -1;

    static thread_local NodePool pool;
    MCTSNode root(board);

    run_simulations(root, board, depth, 100000, pool);

    chess::Move best = pick_best(root);
    std::strncpy(out_move, chess::uci::moveToUci(best).c_str(), out_len);
    out_move[out_len-1] = '\0';
    return 0;
}

int best_move_monte_carlo_time(const char* fen, int time_ms, char* out_move, int out_len) {
    chess::Board board;
    if (!board.setFen(fen)) return -2;
    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);
    if (moves.empty()) return -1;

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

    chess::Move best = pick_best(root);
    std::strncpy(out_move, chess::uci::moveToUci(best).c_str(), out_len);
    out_move[out_len - 1] = '\0';
    return 0;
}

int best_move_monte_carlo_cycles(const char* fen, int megacycle_budget, char* out_move, int out_len) {
    chess::Board board;
    if (!board.setFen(fen)) return -2;
    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);
    if (moves.empty()) return -1;

    exceeded_budget.store(false, std::memory_order_relaxed);

    #ifdef USE_PAPI
    std::call_once(papi_init_flag, init_papi);
    #endif

    long long time_ms = std::max(1LL, (long long)megacycle_budget);
    using clock = std::chrono::steady_clock;
    auto deadline = clock::now() + std::chrono::milliseconds(time_ms);
    std::thread timer([deadline]() {
        while (!exceeded_budget && clock::now() < deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        exceeded_budget = true;
    });

    constexpr int SEARCH_DEPTH = 6;
    constexpr int PAPI_CHECK_SIMS = 64;

    static thread_local NodePool pool;
    MCTSNode root(board);

    #pragma omp parallel
    {
    #ifdef USE_PAPI
        PAPI_thread_init(pthread_self);

        int my_event_set    = PAPI_NULL;
        long long my_cycle_budget = (long long)megacycle_budget * 1'000'000LL;
        int my_sim_count    = 0;

        bool papi_ok = (PAPI_create_eventset(&my_event_set) == PAPI_OK) &&
                       (PAPI_add_event(my_event_set, PAPI_TOT_CYC) == PAPI_OK) &&
                       (PAPI_start(my_event_set) == PAPI_OK);

        if (!papi_ok && my_event_set != PAPI_NULL) {
            PAPI_cleanup_eventset(my_event_set);
            PAPI_destroy_eventset(&my_event_set);
            my_event_set = PAPI_NULL;
        }
    #endif

        while (!exceeded_budget.load(std::memory_order_relaxed)) {
            chess::Board local_board = board;
            MCTSNode* leaf = tree_descend(&root, local_board, pool);
            double result  = rollout(local_board, SEARCH_DEPTH);
            backprop(leaf, result);

    #ifdef USE_PAPI
            if (my_event_set != PAPI_NULL && ++my_sim_count >= PAPI_CHECK_SIMS) {
                my_sim_count = 0;
                long long cycles_used;
                if (PAPI_read(my_event_set, &cycles_used) == PAPI_OK)
                    if (cycles_used >= my_cycle_budget)
                        exceeded_budget.store(true, std::memory_order_relaxed);
            }
    #endif
        }

    #ifdef USE_PAPI
        if (my_event_set != PAPI_NULL) {
            long long cycles_used;
            PAPI_stop(my_event_set, &cycles_used);
            PAPI_cleanup_eventset(my_event_set);
            PAPI_destroy_eventset(&my_event_set);
            my_event_set = PAPI_NULL;
        }
        PAPI_unregister_thread();
    #endif
    }

    exceeded_budget = true;
    timer.join();

    chess::Move best = pick_best(root);
    std::strncpy(out_move, chess::uci::moveToUci(best).c_str(), out_len);
    out_move[out_len - 1] = '\0';
    return 0;
}