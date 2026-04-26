#include <cstring>
#include <memory>
#include <random>

#include "common.hpp"
#include "evaluate.hpp"


// rollout() simulates a random playout from the given state until a terminal state is reached or a maximum depth is exceeded.
// Returns 1.0 (win), 0.0 (loss), or 0.5 (draw) from the perspective of the side that was to move when rollout() was first called.
double rollout(chess::Board state, int max_depth = 50) {
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
    return eval > 0 ? 1.0 : (eval < 0 ? 0.0 : 0.5); // Win if eval positive, loss if eval negative, draw if eval zero
}


struct MCTSNode {
    chess::Board board; // Board state at this node
    chess::Move move; // Move that led to this node
    double wins = 0.0; int visits = 0;
    std::vector<chess::Move> untried_moves;
    std::vector<std::unique_ptr<MCTSNode>> children;
    MCTSNode* parent = nullptr;

    explicit MCTSNode(const chess::Board& b, chess::Move m = chess::Move::NO_MOVE, MCTSNode* p = nullptr)
        : board(b), move(m), parent(p) {
        // Generate randomized list of legal moves upon node construction
        chess::Movelist ml;
        chess::movegen::legalmoves(ml, board);
        for (int i = 0; i < (int)ml.size(); ++i)
            untried_moves.push_back(ml[i]);
        static thread_local std::mt19937 rng(std::random_device{}());

        // Saves RNG calls later for O(1) selection of untried move
        std::shuffle(untried_moves.begin(), untried_moves.end(), rng);
    }

    bool is_terminal() const { return untried_moves.empty() && children.empty(); }
    bool fully_expanded() const { return untried_moves.empty(); }

    MCTSNode* best_child(double c = std::sqrt(2.0)) const { // TODO: tune constant c
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
static MCTSNode* tree_descend(MCTSNode* node) {
    while (!node->is_terminal()) {
        if (!node->fully_expanded()) {
            // Expand tree by creating a new child node for an untried move
            chess::Move m = node->untried_moves.back();
            node->untried_moves.pop_back();

            chess::Board next = node->board;
            next.makeMove(m); // Play out the move to get the next board state

            node->children.push_back(std::make_unique<MCTSNode>(next, m, node));
            return node->children.back().get(); // Return the newly expanded node
        }

        // Descend into the best child according to UCB1 until we reach a node that is not fully expanded or is terminal
        node = node->best_child();
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
    MCTSNode root(board); // Create root node of the search tree

    for (int i = 0; i < num_simulations; ++i) {
        MCTSNode* leaf = tree_descend(&root);
        double result = rollout(leaf->board, max_depth);
        backprop(leaf, result);
    }

    // TODO: best move is the one with most visits for now, is this the correct criterion?
    MCTSNode* best = nullptr;
    int most_visits = -1;
    for (const auto& child : root.children) {
        if (child->visits > most_visits) {
            most_visits = child->visits;
            best = child.get();
        }
    }

    return best->move; // Return the move that leads to the best child of the root after MCTS
}


int best_move_monte_carlo_depth(const char* fen, int depth, char* out_move, int out_len) {
    chess::Board board;
    if (!board.setFen(fen)) return -2; // Invalid FEN

    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);
    if (moves.empty()) return -1; // Checkmate or stalemate

    // Use branching_factor^depth simulations to somewhat match alpha-beta's node budget at the same depth
    constexpr int BRANCHING_FACTOR = 5;
    int bounded_depth = std::min(depth, 7);
    int num_simulations = std::pow(BRANCHING_FACTOR, bounded_depth); // TODO: tune this formula somehow?

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


int best_move_monte_carlo_flops(const char* fen, int flop_budget, char* out_move, int out_len) {
    chess::Board board;
    if (!board.setFen(fen)) return -2; // Invalid FEN

    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);
    if (moves.empty()) return -1; // Checkmate or stalemate

    // TODO: run simulations until flop_budget is exhausted
    chess::Move best = monte_carlo_search(board);
    std::string uci  = chess::uci::moveToUci(best);
    std::strncpy(out_move, uci.c_str(), out_len);
    out_move[out_len - 1] = '\0';
    return 0;
}
