#include <cstring>

#include "../../chess-library/include/chess.hpp"
#include "common.hpp"
#include "utils.h"
#include <random>
#include <memory>

// rollout() simulates a random playout from the given state until a terminal state is reached or a maximum depth is exceeded.
// returns 1.0 (win), 0.0 (loss), or 0.5 (draw) from the perspective of the side that was to move when rollout() was first called.
// ---------------------------------------------------------------------------
double rollout(chess::Board state) {
    static thread_local std::mt19937 rng(std::random_device{}()); // seed/create RNG object once

    const int max_depth = 50; //TODO: change truncation depth?
    const chess::Color root_color = state.sideToMove(); // side to move at start of rollout.

    for (int depth = 0; depth < max_depth; ++depth) {
        // Check 50-50 / repetition draws before generating moves
        if (state.isHalfMoveDraw() || state.isRepetition(1)) { // we set repetition threshold to 1, so this checks if the current position has occurred before in the game (not counting the current position), rather than threefold repition legality condition. this is purely for search eff. but can be changed to 2 (def) if desired.
            return 0.5; // draw
        }

        chess::Movelist moves;
        chess::movegen::legalmoves(moves, state);

        if (moves.empty()) {
            if (state.inCheck()) {
                // checkmate: the side to move is mated
                return (state.sideToMove() == root_color) ? 0.0 : 1.0; // loss if root_color is to move, win if opponent is to move
            } else {
                return 0.5; // stalemate
            }
        }

        // apply/continue with a random legal move
        std::uniform_int_distribution<int> dist(0, (int)moves.size() - 1);
        state.makeMove(moves[dist(rng)]);
    }

    // if rollout truncated, we use eval as a proxy
    int eval = evaluate(state);
    // eval is relative to current side to move, but we need it relative to root_color
    if (state.sideToMove() != root_color) eval = -eval;
    return eval > 0 ? 1.0 : (eval < 0 ? 0.0 : 0.5); // win if eval positive, loss if eval negative, draw if eval zero
}

// MCTSNode class
struct MCTSNode {
    chess::Board board; // board state at this node
    chess::Move move; // move that led to this node
    double wins = 0.0;
    int visits = 0;
    std::vector<chess::Move> untried_moves;
    std::vector<std::unique_ptr<MCTSNode>> children;
    MCTSNode* parent = nullptr;

    explicit MCTSNode(const chess::Board& b, chess::Move m = chess::Move::NO_MOVE, MCTSNode* p = nullptr) : board(b), move(m), parent(p)
    {
        // generate randomized list of legal moves upon node construction
        chess::Movelist ml;
        chess::movegen::legalmoves(ml, board);
        for (int i = 0; i < (int)ml.size(); ++i)
            untried_moves.push_back(ml[i]);
        static thread_local std::mt19937 rng(std::random_device{}());
        std::shuffle(untried_moves.begin(), untried_moves.end(), rng); // saves RNG calls later for O(1) selection of untried move
    }

    bool is_terminal() const { return untried_moves.empty() && children.empty(); }
    bool fully_expanded() const { return untried_moves.empty(); }

    MCTSNode* best_child(double c = std::sqrt(2.0)) const { // TODO: tune constant c
        MCTSNode* best = nullptr;
        double best_score = -1e18;
        double log_visits = std::log((double)visits);

        for (const auto& child : children) {
            // win rate is from the child's mover's perspective, so we flip it
            double q_s_a = 1.0 - (child->wins / child->visits); // exploitation term
            double n_s_a = c * std::sqrt(log_visits / child->visits); // exploration term
            double score = q_s_a + n_s_a;
            if (score > best_score) {
                best_score = score;
                best = child.get();
            }
        }
        return best;
    }
};

// tree_descend() descends from the given node until it finds a node that is not fully expanded or is terminal, expanding a new child if possible, and then returns the node that was reached
static MCTSNode* tree_descend(MCTSNode* node) {
    while (!node->is_terminal()) {
        if (!node->fully_expanded()) {
            // expand tree by creating a new child node for an untried move
            chess::Move m = node->untried_moves.back();
            node->untried_moves.pop_back();

            chess::Board next = node->board;
            next.makeMove(m); // play out the move to get the next board state

            node->children.push_back(std::make_unique<MCTSNode>(next, m, node));
            return node->children.back().get(); // return the newly expanded node
        }
        node = node->best_child(); // descend into the best child according to UCB1 until we reach a node that is not fully expanded or is terminal
    }
    return node; // return a terminal node
}

static void backprop(MCTSNode* node, double result) {
    // result is from PoV of the root's side to move, but we need to flip it as we go up the tree because each level alternates whose turn it is
    while (node != nullptr) {
        node->visits++;
        node->wins += result;
        result = 1.0 - result;
        node = node->parent;
    }
}


// best move found after num_simulations playouts/simulations of MCTS from the given board state.
chess::Move monte_carlo_search(const chess::Board& board, int num_simulations = 1000) {
    
    MCTSNode root(board); // create root node of the search tree

    for (int i = 0; i < num_simulations; ++i) {
        MCTSNode* leaf = tree_descend(&root);
        double result = rollout(leaf->board);
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

    return best->move; // return the move that leads to the best child of the root after MCTS
}

int best_move_monte_carlo(const char* fen, int num_simulations, char* out_move, int out_len) {
    chess::Board board;
    if (!board.setFen(fen)) return -1; // Invalid FEN

    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);
    if (moves.empty()) return -2; // no legal moves

    chess::Move best_move = monte_carlo_search(board, num_simulations);

    std::string uci = chess::uci::moveToUci(best_move);
    std::strncpy(out_move, uci.c_str(), out_len); // copy move string to output buffer
    out_move[out_len - 1] = '\0'; // null terminator
    return 0;
}