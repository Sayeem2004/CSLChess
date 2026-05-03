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
    std::vector<std::unique_ptr<MCTSNode>> children;
    MCTSNode* parent = nullptr;

    std::mutex mtx;

    explicit MCTSNode(const chess::Board& b, chess::Move m = chess::Move::NO_MOVE, MCTSNode* p = nullptr)
        : move(m), parent(p) {
        chess::Movelist ml;
        chess::movegen::legalmoves(ml, b);
        for (int i = 0; i < (int)ml.size(); ++i)
            untried_moves.push_back(ml[i]);
        static thread_local std::mt19937 rng(std::random_device{}());
        std::shuffle(untried_moves.begin(), untried_moves.end(), rng);
    }

    MCTSNode(const MCTSNode&) = delete;
    MCTSNode& operator=(const MCTSNode&) = delete;

    bool is_terminal() const { return untried_moves.empty() && children.empty(); }
    bool fully_expanded() const { return untried_moves.empty(); }

    MCTSNode* best_child(double c = std::sqrt(2.0)) const {
        MCTSNode* best = nullptr;
        double best_score = -1e18;

        int parent_visits = visits.load() + virtual_loss.load();
        double log_visits = std::log(std::max(1.0, (double)parent_visits));

        for (const auto& child : children) {
            int child_v = child->visits.load() + child->virtual_loss.load();
            if (child_v == 0) return child.get();
            double child_w = child->wins.load();
            double q_s_a = 1.0 - (child_w / child_v);
            double n_s_a = c * std::sqrt(log_visits / child_v);
            double score = q_s_a + n_s_a;
            if (score > best_score) {
                best_score = score;
                best = child.get();
            }
        }
        return best;
    }
};

static MCTSNode* tree_descend(MCTSNode* node, chess::Board& board) {
    node->virtual_loss++;

    while (true) {
        std::unique_lock<std::mutex> lock(node->mtx);

        if (node->is_terminal()) return node;

        if (!node->fully_expanded()) {
            chess::Move m = node->untried_moves.back();
            node->untried_moves.pop_back();
            board.makeMove(m);
            node->children.push_back(std::make_unique<MCTSNode>(board, m, node));
            MCTSNode* new_child = node->children.back().get();
            new_child->virtual_loss++;
            return new_child;
        }

        MCTSNode* best = node->best_child();
        lock.unlock();

        if (best == nullptr) return node;

        best->virtual_loss++;
        node = best;
        board.makeMove(node->move);
    }
}

static void backprop(MCTSNode* node, double result) {
    while (node != nullptr) {
        node->visits++;
        double old = node->wins.load(std::memory_order_relaxed);
        while (!node->wins.compare_exchange_weak(old, old + result, std::memory_order_relaxed));
        node->virtual_loss--;
        result = 1.0 - result;
        node = node->parent;
    }
}

static void run_simulations(MCTSNode& root, const chess::Board& board, int max_depth, int num_simulations) {
    #pragma omp parallel for schedule(static, 16)
    for (int i = 0; i < num_simulations; ++i) {
        chess::Board local_board = board;
        MCTSNode* leaf = tree_descend(&root, local_board);
        double result = rollout(local_board, max_depth);
        backprop(leaf, result);
    }
}

static void run_simulations_persistent(MCTSNode& root, const chess::Board& board, int max_depth) {
    #pragma omp parallel
    {
        while (!exceeded_budget.load(std::memory_order_relaxed)) {
            chess::Board local_board = board;
            MCTSNode* leaf = tree_descend(&root, local_board);
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

    printf("MCTS prefers move %s with %s rate %.2f%% (%d visits)\n",
           chess::uci::moveToUci(best_move).c_str(),
           expected_win_rate >= 0.5 ? "win" : "loss",
           std::max(expected_win_rate, 1.0 - expected_win_rate) * 100.0,
           most_visits);
    return best_move;
}

chess::Move monte_carlo_search(const chess::Board& board, int max_depth, int num_simulations) {
    MCTSNode root(board);
    run_simulations(root, board, max_depth, num_simulations);
    return pick_best(root);
}

int best_move_monte_carlo_depth(const char* fen, int depth, char* out_move, int out_len) {
    chess::Board board;
    if (!board.setFen(fen)) return -2;
    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);
    if (moves.empty()) return -1;

    MCTSNode root(board);
    run_simulations(root, board, depth, 1000000);

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
    MCTSNode root(board);
    run_simulations_persistent(root, board, SEARCH_DEPTH);
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

    constexpr int SEARCH_DEPTH    = 6;
    constexpr int PAPI_CHECK_SIMS = 64;

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
            MCTSNode* leaf = tree_descend(&root, local_board);
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