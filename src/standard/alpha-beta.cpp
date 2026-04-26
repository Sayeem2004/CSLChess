#include <cstring>

#include "common.hpp"
#include "evaluate.hpp"

// TODO: implement alpha-beta minimax search
// Returns the best move found and writes it into out_move in UCI format.
// Returns 0 on success, -1 if no legal moves, -2 if FEN is invalid.
static int run_alpha_beta(const char* fen, char* out_move, int out_len) {
    chess::Board board;
    if (!board.setFen(fen)) return -2;

    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);
    if (moves.empty()) return -1;

    // TODO: alpha-beta search — pick best move and write UCI string to out_move
    return 0;
}

int best_move_alpha_beta_depth(const char* fen, int depth, char* out_move, int out_len) {
    // TODO: use depth to limit search
    return run_alpha_beta(fen, out_move, out_len);
}

int best_move_alpha_beta_time(const char* fen, int time_ms, char* out_move, int out_len) {
    // TODO: use time_ms to limit search
    return run_alpha_beta(fen, out_move, out_len);
}

int best_move_alpha_beta_flops(const char* fen, int flop_budget, char* out_move, int out_len) {
    // TODO: use flop_budget to limit search
    return run_alpha_beta(fen, out_move, out_len);
}
