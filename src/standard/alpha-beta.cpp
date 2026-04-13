#include <cstring>

#include "../../chess-library/include/chess.hpp"
#include "common.hpp"
#include "utils.hpp"


// TODO: implement alpha-beta minimax search
void alpha_beta_search(const chess::Board& board, chess::Board& next_board) {
    return;
}


int best_move_alpha_beta(const char* fen, int depth, char* out_move, int out_len) {
    chess::Board board, next_board;
    if (!board.setFen(fen)) return -2;  // Invalid FEN

    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);
    if (moves.empty()) return -1;  // Checkmate or stalemate

    alpha_beta_search(board, next_board);
    return 0; // TODO return the best move in UCI format
}
