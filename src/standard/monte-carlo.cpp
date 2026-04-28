#include <cstring>
#include <memory>
#include <random>
#include <omp.h>

#include "common.hpp"
#include "evaluate.hpp"


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
    //chess::Board board; // Board state at this node
    chess::Move move; // Move that led to this node
    double wins = 0.0; int visits = 0;
    std::vector<chess::Move> untried_moves;
    std::vector<std::unique_ptr<MCTSNode>> children;
    MCTSNode* parent = nullptr;

    explicit MCTSNode(const chess::Board& b, chess::Move m = chess::Move::NO_MOVE, MCTSNode* p = nullptr)
        : move(m), parent(p) {
        // Generate randomized list of legal moves upon node construction
        chess::Movelist ml;
        chess::movegen::legalmoves(ml, b);
        for (int i = 0; i < (int)ml.size(); ++i)
            untried_moves.push_back(ml[i]);
        static thread_local std::mt19937 rng(std::random_device{}());

        // Saves RNG calls later for O(1) selection of untried move
        std::shuffle(untried_moves.begin(), untried_moves.end(), rng);
    }

    bool is_terminal() const { return untried_moves.empty() && children.empty(); }
    bool fully_expanded() const { return untried_moves.empty(); }

    MCTSNode* best_child(double c = std::sqrt(0.0)) const { // TODO: tune constant c
        MCTSNode* best    = nullptr;
        double best_score = -1e18;
        double log_visits = std::log((double)visits);

        for (const auto& child : children) {
            // Win rate is from the child's mover's perspective, so we flip it
            double q_s_a = 1.0 - (child->wins / child->visits); // Exploitation term
            double n_s_a = c * std::sqrt(log_visits / child->visits); // Exploration term
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
            // Expand tree by creating a new child node for an untried move
            chess::Move m = node->untried_moves.back();
            node->untried_moves.pop_back();

            //chess::Board next = node->board;
            board.makeMove(m); // Play out the move to get the next board state

            node->children.push_back(std::make_unique<MCTSNode>(board, m, node));
            return node->children.back().get(); // Return the newly expanded node
        }

        // Descend into the best child according to UCB1 until we reach a node that is not fully expanded or is terminal
        node = node->best_child();
        board.makeMove(node->move); // play move to keep node in sync
    }
    return node; // Return a terminal node
}


static void backprop(MCTSNode* node, double result) {
    // Result is from PoV of the root's side to move, but we need to flip it as we go up the tree
    // because each level alternates whose turn it is
    while (node != nullptr) {
        node->visits++;
        node->wins += result;
        result = 1.0 - result;
        node = node->parent;
    }
}


// Best move found after num_simulations playouts/simulations of MCTS from the given board state.
// TODO: tune default max_depth and num_simulations parameters

chess::Move monte_carlo_search(const chess::Board& board, int max_depth = 50, int num_simulations = 1000) {
    int num_threads = omp_get_max_threads();
    int sims_per_thread = num_simulations / num_threads;

    // Each thread builds its own independent tree
    std::vector<std::unique_ptr<MCTSNode>> roots(num_threads);
    for (int t = 0; t < num_threads; ++t)
        roots[t] = std::make_unique<MCTSNode>(board);

    #pragma omp parallel for schedule(static, 1)
    for (int t = 0; t < num_threads; ++t) {
        MCTSNode* root = roots[t].get();
        for (int i = 0; i < sims_per_thread; ++i) {
            chess::Board local_board = board; // Each thread needs its own copy of the board to play out moves
            MCTSNode* leaf = tree_descend(root, local_board);
            double result = rollout(local_board, max_depth);
            backprop(leaf, result);
        }
    }

    // Merge visit/win counts across all roots' children by move
    std::unordered_map<uint16_t, std::pair<int,double>> move_stats; // move -> {visits, wins}
    for (int t = 0; t < num_threads; ++t) {
        for (const auto& child : roots[t]->children) {
            uint16_t key = child->move.move(); // raw move bits as key
            auto& s = move_stats[key];
            s.first  += child->visits;
            s.second += child->wins;
        }
    }

    uint16_t best_key = 0;
    int most_visits = -1;
    chess::Move best_move = chess::Move::NO_MOVE;

    for (int t = 0; t < num_threads; ++t) {
        for (const auto& child : roots[t]->children) {
            uint16_t key = child->move.move();
            int v = move_stats[key].first;
            if (v > most_visits) {
                most_visits = v;
                best_key = key;
                best_move = child->move;
            }
        }
    }

    auto& best_stats = move_stats[best_key];
    double expected_win_rate = 1.0 - (best_stats.second / best_stats.first);
    printf("MCTS prefers move %s with %s rate %.2f%% (%d visits across %d threads)\n",
           chess::uci::moveToUci(best_move).c_str(),
           expected_win_rate >= 0.5 ? "win" : "loss",
           std::max(expected_win_rate, 1.0 - expected_win_rate) * 100.0,
           most_visits, num_threads);

    return best_move;
}


int best_move_monte_carlo_depth(const char* fen, int depth, char* out_move, int out_len) {
    chess::Board board;
    if (!board.setFen(fen)) return -2; // Invalid FEN

    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);
    if (moves.empty()) return -1; // Checkmate or stalemate

    // Use branching_factor^depth simulations to somewhat match alpha-beta's node budget at the same depth
    constexpr int BRANCHING_FACTOR = 10;
    int bounded_depth = std::min(depth, 7);
    int num_simulations = 1000000//std::pow(BRANCHING_FACTOR, bounded_depth); // TODO: tune this formula somehow?

    chess::Move best = monte_carlo_search(board, depth, num_simulations);
    std::string uci  = chess::uci::moveToUci(best);
    std::strncpy(out_move, uci.c_str(), out_len);
    out_move[out_len - 1] = '\0';
    return 0;
}


int best_move_monte_carlo_time(const char* fen, int time_ms, char* out_move, int out_len) {
    chess::Board board;
    if (!board.setFen(fen)) return -2; // Invalid FEN

    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);
    if (moves.empty()) return -1; // Checkmate or stalemate

    // TODO: run simulations until time_ms budget is exhausted
    chess::Move best = monte_carlo_search(board);
    std::string uci  = chess::uci::moveToUci(best);
    std::strncpy(out_move, uci.c_str(), out_len);
    out_move[out_len - 1] = '\0';
    return 0;
}


int best_move_monte_carlo_flops(const char* fen, int megaflop_budget, char* out_move, int out_len) {
    chess::Board board;
    if (!board.setFen(fen)) return -2; // Invalid FEN

    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);
    if (moves.empty()) return -1; // Checkmate or stalemate

    // TODO: run simulations until megaflop_budget is exhausted
    chess::Move best = monte_carlo_search(board);
    std::string uci  = chess::uci::moveToUci(best);
    std::strncpy(out_move, uci.c_str(), out_len);
    out_move[out_len - 1] = '\0';
    return 0;
}
