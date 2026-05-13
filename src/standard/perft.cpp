#include <algorithm>
#include "common.hpp"
#include "evaluate.hpp"


static chess::Move ab_killers[AB_MAX_PLY][2];


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


static long long ab_count(chess::Board& board, int depth, int alpha, int beta, long long& nodes) {
    nodes++;
    if (board.isRepetition(2) || board.isHalfMoveDraw()) return 0;

    if (depth == 0) return engine_evaluate(board);

    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);
    if (moves.empty()) return board.inCheck() ? -20000 - depth * 100 : 0;

    int ki = std::min(depth, AB_MAX_PLY - 1);
    std::sort(moves.begin(), moves.end(), [&](const chess::Move& a, const chess::Move& b) {
        return ab_score_move(board, a, chess::Move{}, ab_killers[ki]) >
               ab_score_move(board, b, chess::Move{}, ab_killers[ki]);
    });

    int best = -1000000;
    for (const auto& move : moves) {
        board.makeMove(move);
        int score = (int)-ab_count(board, depth - 1, -beta, -alpha, nodes);
        board.unmakeMove(move);

        if (score > best) best = score;
        if (score > alpha) alpha = score;
        if (alpha >= beta) {
            if (move.typeOf() != chess::Move::ENPASSANT &&
                board.at(move.to()).type() == chess::PieceType::NONE) {
                if (ab_killers[ki][0] != move) {
                    ab_killers[ki][1] = ab_killers[ki][0];
                    ab_killers[ki][0] = move;
                }
            }
            break;
        }
    }
    return best;
}

extern "C" long long count_nodes_alpha_beta_depth(const char* fen, int depth) {
    chess::Board board;
    if (!board.setFen(fen)) return -2;

    long long nodes = 0;
    ab_count(board, depth, -1000000, 1000000, nodes);
    return nodes;
}
