#pragma once
#include <algorithm>
#include "common.hpp"


int engine_evaluate(const chess::Board& board);


inline constexpr int PIECE_VALUES[] = {100, 320, 330, 500, 900, 20000, 0};
inline constexpr chess::PieceType PIECE_TYPES[] = {
    chess::PieceType::PAWN,   chess::PieceType::KNIGHT, chess::PieceType::BISHOP,
    chess::PieceType::ROOK,   chess::PieceType::QUEEN,  chess::PieceType::KING,
};
inline constexpr int AB_MAX_PLY = 64;


// Priority: TT move > captures (MVV-LVA) > killer moves > quiet moves.
inline int ab_score_move(const chess::Board& board, const chess::Move& move,
                         chess::Move tt_mv, const chess::Move* kl = nullptr) {
    if (move == tt_mv) return 2'000'000;
    if (move.typeOf() == chess::Move::ENPASSANT) // Special Capture
        return 1'000'000 + PIECE_VALUES[0] * 10 - PIECE_VALUES[0];

    chess::PieceType vt = board.at(move.to()).type();
    if (vt != chess::PieceType::NONE) { // Normal Capture
        chess::PieceType pt = board.at(move.from()).type();
        return 1'000'000 + PIECE_VALUES[(int)vt] * 10 - PIECE_VALUES[(int)pt];
    }

    if (kl && kl[0] == move) return 900'000;
    if (kl && kl[1] == move) return 800'000;
    return 0;
}
