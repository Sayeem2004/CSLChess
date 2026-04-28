#include "common.hpp"


static long long perft(chess::Board& board, int depth) {
    if (depth == 0) return 1;
    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);
    long long nodes = 0;
    for (const auto& move : moves) {
        board.makeMove(move);
        nodes += perft(board, depth - 1);
        board.unmakeMove(move);
    }
    return nodes;
}


extern "C" long long count_nodes_depth(const char* fen, int depth) {
    chess::Board board;
    if (!board.setFen(fen)) return -2;
    return perft(board, depth);
}