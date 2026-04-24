#include <cstring>

#include "../../chess-library/include/chess.hpp"
#include "common.hpp"
#include "utils.h"
#include <random>

// rollout() simulates a random playout from the given state until a terminal state is reached or a maximum depth is exceeded.
// ---------------------------------------------------------------------------
// Random rollout from `state` up to max_depth half-moves.
// Returns 1.0 (win), 0.0 (loss), or 0.5 (draw) from the perspective of the
// side that was to move when rollout() was first called.
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

// TODO: implement monte carlo tree search
void monte_carlo_search(const chess::Bitboard& board, chess::Bitboard& next_board) {
    return;
}


// Stub: returns the first legal move found (placeholder until MCTS is implemented).
int best_move_monte_carlo(const char* fen, int num_simulations, char* out_move, int out_len) {
    chess::Bitboard board, next_board;
    if (!board.setFen(fen)) return -2;  // Invalid FEN

    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);
    if (moves.empty()) return -1;  // Checkmate or stalemate

    monte_carlo_search(board, next_board);
    return 0; // TODO return the best move in UCI format
}