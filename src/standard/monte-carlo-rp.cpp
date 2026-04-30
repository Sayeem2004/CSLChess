#include <cstring>
#include <memory>
#include <random>
#include <omp.h>
#include <atomic>
#include <mutex>
#include <thread>

#include "common.hpp"
#include "evaluate.hpp"

#ifdef USE_PAPI
#include <papi.h>
static std::once_flag papi_init_flag;
static void init_papi() { PAPI_library_init(PAPI_VER_CURRENT); }
#endif
static std::atomic<bool> exceeded_budget{false};


// rollout() simulates a random playout from the given state until a terminal state is reached or a maximum depth is exceeded.
// Returns 1.0 (win), 0.0 (loss), or 0.5 (draw) from the perspective of the side that was to move when rollout() was first called.
double rollout(chess::Board state, int max_depth = 5) {
    static thread_local std::mt19937 rng(std::random_device{}());
    const chess::Color root_color = state.sideToMove();

    for (int depth = 0; depth < max_depth; ++depth) {
        // Check 50-50 / repetition draws before generating moves
        if (state.isHalfMoveDraw() || state.isRepetition(2)) return 0.5;
        chess::Movelist moves;
        chess::movegen::legalmoves(moves, state);

        if (moves.empty()) {
            if (!state.inCheck()) return 0.5; // Stalemate
            else return (state.sideToMove() == root_color) ? 0.0 : 1.0; // Loss if root_color is to move, win if opponent is to move
        }

        // Apply/continue with a random legal move
        std::uniform_int_distribution<int> dist(0, (int)moves.size() - 1);
        state.makeMove(moves[dist(rng)]);
    }

    // If rollout truncated, we use evaluate function as a proxy
    int eval = engine_evaluate(state);
    if (state.sideToMove() != root_color) eval = -eval;
    constexpr double k = 0.003; 
    return 1.0 / (1.0 + std::exp(-k * eval));
}


struct MCTSNode {
    chess::Move move;
    double wins = 0.0; int visits = 0;
    std::vector<chess::Move> untried_moves;
    std::vector<std::unique_ptr<MCTSNode>> children;
    MCTSNode* parent = nullptr;

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

    MCTSNode* best_child(double c = std::sqrt(0.0)) const {
        MCTSNode* best    = nullptr;
        double best_score = -1e18;
        double log_visits = std::log((double)visits);

        for (const auto& child : children) {
            double q_s_a = 1.0 - (child->wins / child->visits);
            double n_s_a = c * std::sqrt(log_visits / child->visits);
            double score = q_s_a + n_s_a;
            if (score > best_score) {
                best_score = score;
                best = child.get();
            }
        }
        return best;
    }
};


// tree_descend() descends from the given node until it finds a node that is not fully expanded or is terminal,
// expanding a new child if possible, and then returns the node that was reached
static MCTSNode* tree_descend(MCTSNode* node, chess::Board& board) {
    while (!node->is_terminal()) {
        if (!node->fully_expanded()) {
            chess::Move m = node->untried_moves.back();
            node->untried_moves.pop_back();

            board.makeMove(m);

            node->children.push_back(std::make_unique<MCTSNode>(board, m, node));
            return node->children.back().get();
        }
        node = node->best_child();
        board.makeMove(node->move); // play move to keep node in sync
    }
    return node; // Return a terminal node
}


static void backprop(MCTSNode* node, double result) {
    while (node != nullptr) {
        node->visits++;
        node->wins += result;
        result = 1.0 - result;
        node = node->parent;
    }
}

static void run_simulations(MCTSNode& root, const chess::Board& board, int max_depth, int num_simulations) {
    for (int i = 0; i < num_simulations; ++i) {
        chess::Board local_board = board;
        MCTSNode* leaf = tree_descend(&root, local_board);
        double result = rollout(local_board, max_depth);
        backprop(leaf, result);
    }
}

static chess::Move pick_best(std::vector<std::unique_ptr<MCTSNode>>& roots) {
    std::unordered_map<uint16_t, std::pair<int,double>> stats;

    for (auto& root : roots) {
        for (const auto& child : root->children) {
            auto& s = stats[child->move.move()];
            s.first  += child->visits;
            s.second += child->wins;
        }
    }

    int best_visits = -1;
    chess::Move best_move = chess::Move::NO_MOVE;
    double expected_win_rate = 0.0;

    for (const auto& [key, val] : stats) {
        if (val.first > best_visits) {
            best_visits = val.first;
            best_move = chess::Move(key);
            expected_win_rate = 1.0 - (val.second / val.first);
        }
    }

    /*printf("MCTS prefers move %s with %s rate %.2f%% (%d visits)\n",
           chess::uci::moveToUci(best_move).c_str(),
           expected_win_rate >= 0.5 ? "win" : "loss",
           std::max(expected_win_rate, 1.0 - expected_win_rate) * 100.0,
           best_visits);*/

    return best_move;
}

chess::Move monte_carlo_search(const chess::Board& board, int max_depth = 5, int num_simulations = 1000) {
    int num_threads = omp_get_max_threads();

    std::vector<std::unique_ptr<MCTSNode>> roots(num_threads);
    for (int t = 0; t < num_threads; ++t)
        roots[t] = std::make_unique<MCTSNode>(board);

    int base = num_simulations / num_threads;
    int extra = num_simulations % num_threads;

    #pragma omp parallel for schedule(static,1)
    for (int t = 0; t < num_threads; ++t) {
        int sims = base + (t < extra ? 1 : 0);
        run_simulations(*roots[t], board, max_depth, sims);
    }

    return pick_best(roots);
}

int best_move_monte_carlo_depth(const char* fen, int depth, char* out_move, int out_len) {
    chess::Board board;
    if (!board.setFen(fen)) return -2;

    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);
    if (moves.empty()) return -1;

    constexpr int SIMS = 100000;

    chess::Move best = monte_carlo_search(board, depth, SIMS);

    std::strncpy(out_move, chess::uci::moveToUci(best).c_str(), out_len);
    out_move[out_len - 1] = '\0';
    return 0;
}

int best_move_monte_carlo_time(const char* fen, int time_ms,
                               char* out_move, int out_len) {
    chess::Board board;
    if (!board.setFen(fen)) return -2;

    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);
    if (moves.empty()) return -1;

    exceeded_budget = false;

    static bool printed = false;
    if (!printed) {
        printed = true;
        fprintf(stderr, "[monte-carlo] threads: %d\n", omp_get_max_threads());
    }

    using clock = std::chrono::steady_clock;
    auto deadline = clock::now() + std::chrono::milliseconds(time_ms);

    std::thread timer([deadline]() {
        while (clock::now() < deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        exceeded_budget = true;
    });

    int num_threads = omp_get_max_threads();
    std::vector<std::unique_ptr<MCTSNode>> roots(num_threads);

    for (int t = 0; t < num_threads; ++t)
        roots[t] = std::make_unique<MCTSNode>(board);

    constexpr int SEARCH_DEPTH = 6;

    #pragma omp parallel for
    for (int t = 0; t < num_threads; ++t) {
        while (!exceeded_budget) {
            chess::Board local_board = board;
            MCTSNode* leaf = tree_descend(roots[t].get(), local_board);
            if (exceeded_budget) break;          // ← check before expensive rollout
            double result = rollout(local_board, SEARCH_DEPTH);
            if (exceeded_budget) break;          // ← check before backprop
            backprop(leaf, result);
        }
    }

    timer.join();

    chess::Move best = pick_best(roots);

    std::strncpy(out_move, chess::uci::moveToUci(best).c_str(), out_len);
    out_move[out_len - 1] = '\0';

    return 0;
}

int best_move_monte_carlo_cycles(const char* fen, int megacycle_budget,
                                 char* out_move, int out_len) {
    chess::Board board;
    if (!board.setFen(fen)) return -2;

    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);
    if (moves.empty()) return -1;

    exceeded_budget = false;

    static bool printed = false;
    if (!printed) {
        printed = true;
        fprintf(stderr, "[monte-carlo] threads: %d\n", omp_get_max_threads());
    }

    #ifdef USE_PAPI
    std::call_once(papi_init_flag, init_papi);

    int event_set = PAPI_NULL;
    PAPI_create_eventset(&event_set);
    PAPI_add_event(event_set, PAPI_TOT_CYC);
    PAPI_start(event_set);

    long long cycle_budget = (long long)megacycle_budget * 1'000'000LL;
    #endif

    int num_threads = omp_get_max_threads();
    std::vector<std::unique_ptr<MCTSNode>> roots(num_threads);

    for (int t = 0; t < num_threads; ++t)
        roots[t] = std::make_unique<MCTSNode>(board);

    constexpr int SEARCH_DEPTH = 6;

    #pragma omp parallel for
    for (int t = 0; t < num_threads; ++t) {
        while (!exceeded_budget) {
            chess::Board local_board = board;
            MCTSNode* leaf = tree_descend(roots[t].get(), local_board);
            double result = rollout(local_board, SEARCH_DEPTH);
            backprop(leaf, result);

            #ifdef USE_PAPI
            #pragma omp critical
            {
                long long cycles_used;
                PAPI_read(event_set, &cycles_used);
                if (cycles_used >= cycle_budget)
                    exceeded_budget = true;
            }
            #else
            exceeded_budget = true;
            #endif
        }
    }

    #ifdef USE_PAPI
    long long cycles_used;
    PAPI_stop(event_set, &cycles_used);
    PAPI_cleanup_eventset(event_set);
    PAPI_destroy_eventset(&event_set);
    #endif

    chess::Move best = pick_best(roots);

    std::strncpy(out_move, chess::uci::moveToUci(best).c_str(), out_len);
    out_move[out_len - 1] = '\0';

    return 0;
}