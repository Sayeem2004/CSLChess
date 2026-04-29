#include <cstring>
#include <memory>
#include <random>
#include <atomic>
#include <mutex>
#include <omp.h>
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

        std::uniform_int_distribution<int> dist(0, (int)moves.size() - 1);
        state.makeMove(moves[dist(rng)]);
    }

    int eval = engine_evaluate(state);
    if (state.sideToMove() != root_color) eval = -eval;
    constexpr double k = 0.003;
    return 1.0 / (1.0 + std::exp(-k * eval));
}


struct MCTSNode {
    chess::Move move;
    std::atomic<double> wins{0.0}; 
    std::atomic<int> visits{0};
    std::atomic<int> virtual_loss{0};
    
    std::vector<chess::Move> untried_moves;
    std::vector<std::unique_ptr<MCTSNode>> children;
    MCTSNode* parent = nullptr;
    
    std::mutex mtx; // to protect thread access to untried_moves and children

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
            
            // if a node is completely unexplored, immediately prioritize it
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


// tree_descend() navigates the shared tree, applies virtual losses along the path to force thread diversification
static MCTSNode* tree_descend(MCTSNode* node, chess::Board& board) {
    node->virtual_loss++; // Apply virtual loss to the root immediately

    while (true) {
        std::unique_lock<std::mutex> lock(node->mtx);
        
        if (node->is_terminal()) return node; // reached a terminal node, return it for rollout

        if (!node->fully_expanded()) {
            // Expand tree by creating a new child node
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
        
        best->virtual_loss++;
        node = best;
        board.makeMove(node->move); 
    }
}


static void backprop(MCTSNode* node, double result) {
    // traverse back up to the root, updating stats and removing the virtual losses
    while (node != nullptr) {
        node->visits++;
        double old = node->wins.load(std::memory_order_relaxed);
        while (!node->wins.compare_exchange_weak(old, old + result, std::memory_order_relaxed));
        node->virtual_loss--;
        result = 1.0 - result;
        node = node->parent;
    }
}


// Add an overload that accepts an existing root to accumulate into
static void run_simulations(MCTSNode& root, const chess::Board& board,
                            int max_depth, int num_simulations) {
    #pragma omp parallel for schedule(dynamic, 16)
    for (int i = 0; i < num_simulations; ++i) {
        chess::Board local_board = board;
        MCTSNode* leaf = tree_descend(&root, local_board);
        double result = rollout(local_board, max_depth);
        backprop(leaf, result);
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

chess::Move monte_carlo_search(const chess::Board& board, int max_depth = 5, int num_simulations = 1000) {
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

    static bool printed = false;
    if (!printed) { printed = true; fprintf(stderr, "[monte-carlo] threads: %d\n", omp_get_max_threads()); }
    
    MCTSNode root(board);
    run_simulations(root, board, depth, 100000); // fixed number of simulations for depth-limited search

    chess::Move best = pick_best(root);
    std::strncpy(out_move, chess::uci::moveToUci(best).c_str(), out_len);
    out_move[out_len-1] = '\0';
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
    if (!printed) { printed = true; fprintf(stderr, "[monte-carlo] threads: %d\n", omp_get_max_threads()); }

    using clock = std::chrono::steady_clock;
    auto deadline = clock::now() + std::chrono::milliseconds(time_ms);

    std::thread timer([deadline]() {
        while (!exceeded_budget && clock::now() < deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        exceeded_budget = true;
    });

    constexpr int BATCH_SIZE = 1000;
    constexpr int SEARCH_DEPTH = 5;

    MCTSNode root(board);
    while (!exceeded_budget)
        run_simulations(root, board, SEARCH_DEPTH, BATCH_SIZE);

    exceeded_budget = true;
    timer.join();

    chess::Move best = pick_best(root);
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
    if (!printed) { printed = true; fprintf(stderr, "[monte-carlo] threads: %d\n", omp_get_max_threads()); }

    #ifdef USE_PAPI
    std::call_once(papi_init_flag, init_papi);
    int event_set = PAPI_NULL;
    PAPI_create_eventset(&event_set);
    PAPI_add_event(event_set, PAPI_TOT_CYC);
    PAPI_start(event_set);
    long long cycle_budget = (long long)megacycle_budget * 1'000'000LL;
    #endif

    constexpr int BATCH_SIZE = 1000;
    constexpr int SEARCH_DEPTH = 5;

    MCTSNode root(board);
    while (!exceeded_budget) {
        run_simulations(root, board, SEARCH_DEPTH, BATCH_SIZE);

    #ifdef USE_PAPI
        long long cycles_used;
        PAPI_read(event_set, &cycles_used);
        if (cycles_used >= cycle_budget) exceeded_budget = true;
    #else
        // fallback if PAPI unavailable
        (void)megacycle_budget;
        exceeded_budget = true; // run exactly one batch
    #endif
    }

    #ifdef USE_PAPI
    long long cycles_used;
    PAPI_stop(event_set, &cycles_used);
    PAPI_cleanup_eventset(event_set);
    PAPI_destroy_eventset(&event_set);
    #endif

    chess::Move best = pick_best(root);
    std::strncpy(out_move, chess::uci::moveToUci(best).c_str(), out_len);
    out_move[out_len - 1] = '\0';
    return 0;
}