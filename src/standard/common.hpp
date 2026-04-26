#pragma once

// Single-header chess library by Disservin — provides chess::Board (bitboard-based),
// https://github.com/Disservin/chess-library/releases/latest/download/chess.hpp
#include "../../chess-library/include/chess.hpp"

// ---------------------------------------------------------------------------
// C interface (extern "C" so Python ctypes / cffi can call these directly)
//
// All functions accept a FEN string representing the current position and
// write the best move in UCI notation (e.g. "e2e4", "e7e8q") into out_move.
//
// Returns  0 on success.
// Returns -1 if no legal move exists (checkmate / stalemate).
// Returns -2 if the FEN string is invalid.
// ---------------------------------------------------------------------------
extern "C" {
    // Alpha-Beta pruning — depth-limited (searches to the given depth).
    int best_move_alpha_beta_depth(const char* fen, int depth, char* out_move, int out_len);

    // Alpha-Beta pruning — time-limited (searches for at most time_ms milliseconds).
    int best_move_alpha_beta_time(const char* fen, int time_ms, char* out_move, int out_len);

    // Alpha-Beta pruning — flop-limited (searches within the given floating-point op budget).
    int best_move_alpha_beta_flops(const char* fen, int flop_budget, char* out_move, int out_len);

    // Monte Carlo Tree Search — depth-limited (runs ~branching_factor^depth simulations).
    int best_move_monte_carlo_depth(const char* fen, int depth, char* out_move, int out_len);

    // Monte Carlo Tree Search — time-limited (runs simulations for at most time_ms milliseconds).
    int best_move_monte_carlo_time(const char* fen, int time_ms, char* out_move, int out_len);

    // Monte Carlo Tree Search — flop-limited (runs simulations within the given floating-point op budget).
    int best_move_monte_carlo_flops(const char* fen, int flop_budget, char* out_move, int out_len);
}
