#include <cstring>

#include "../../chess-library/include/chess.hpp"
#include "common.hpp"


// TODO: implement a proper board evaluation function
int evaluate(const chess::Bitboard& board) {
    return 0;
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