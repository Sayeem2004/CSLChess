#pragma once

// Single-header chess library by Disservin — provides chess::Board (bitboard-based),
// https://github.com/Disservin/chess-library/releases/latest/download/chess.hpp
#include "../../chess-library/include/chess.hpp"

// ---------------------------------------------------------------------------
// C interface (extern "C" so Python ctypes / cffi can call these directly)
//
// Both functions accept a FEN string representing the current position and
// write the best move in UCI notation (e.g. "e2e4", "e7e8q") into out_move.
// We think this is optimal because bitboard sounds a bit annoying to do transfer
// between C++/Python and CSL/Python
//
// Returns  0 on success.
// Returns -1 if no legal move exists (checkmate / stalemate).
// Returns -2 if the FEN string is invalid.
// ---------------------------------------------------------------------------
extern "C" {
    // Alpha-Beta pruning: searches to the given depth.
    int best_move_alpha_beta(const char* fen, int depth, char* out_move, int out_len);

    // Monte Carlo Tree Search: runs num_simulations rollouts.
    int best_move_monte_carlo(const char* fen, int num_simulations, char* out_move, int out_len);
}
