#include "../../chess-library/include/chess.hpp"
#include "common.hpp"
#include "utils.h"

static constexpr int PIECE_VALUES[] = {
    0,    // NONE
    100,  // PAWN
    320,  // KNIGHT
    330,  // BISHOP
    500,  // ROOK
    900,  // QUEEN
    20000 // KING
};
static constexpr chess::PieceType PIECE_TYPES[] = {
    chess::PieceType::PAWN,
    chess::PieceType::KNIGHT,
    chess::PieceType::BISHOP,
    chess::PieceType::ROOK,
    chess::PieceType::QUEEN,
    chess::PieceType::KING
};

// static constexpr int PAWN_PST[64] = {
//       0,  0,  0,  0,  0,  0,  0,  0,
//      50, 50, 50, 50, 50, 50, 50, 50,
//      10, 10, 20, 30, 30, 20, 10, 10,
//       5,  5, 10, 25, 25, 10,  5,  5,
//       0,  0,  0, 20, 20,  0,  0,  0,
//       5, -5,-10,  0,  0,-10, -5,  5,
//       5, 10, 10,-20,-20, 10, 10,  5,
//       0,  0,  0,  0,  0,  0,  0,  0
// };

// inline int pop_lsb(uint64_t &bb) {
//     int sq = __builtin_ctzll(bb); // count trailing zeros (index of LSB)
//     bb &= bb - 1;                 // clear the LSB
//     return sq;
// }

// int evaluate(const chess::Board& board) {
//     int score = 0;
//     using namespace chess;
//     for (Color color : {Color::WHITE, Color::BLACK}) {
//         int sign = (color == Color::WHITE) ? 1 : -1;

//         for (PieceType pt : {
//             PieceType::PAWN,
//             PieceType::KNIGHT,
//             PieceType::BISHOP,
//             PieceType::ROOK,
//             PieceType::QUEEN,
//             PieceType::KING
//         }) {
//             uint64_t bb = board.pieces(pt, color).getBits();
//             int value = PIECE_VALUES[(int)pt];

//             while (bb) {
//                 int sq = pop_lsb(bb);
//                 int index = (color == Color::WHITE) ? sq : (sq ^ 56);
//                 int pst_bonus = 0;
//                 if (pt == PieceType::PAWN) {
//                     pst_bonus = PAWN_PST[index];
//                 }

//                 score += sign * (value + pst_bonus);
//             }
//         }
//     }

//     return score;
// }

// eval function based on material count only, for now/simplicity
int evaluate(const chess::Board& board) {
    int score = 0;
    chess::Color stm = board.sideToMove();
    for (int pt = 0; pt < 6; ++pt) {
        chess::PieceType type = PIECE_TYPES[pt];
        int white_count = board.pieces(type, chess::Color::WHITE).count();
        int black_count = board.pieces(type, chess::Color::BLACK).count();
        int delta = (white_count - black_count) * PIECE_VALUES[pt];
        score += (stm == chess::Color::WHITE) ? delta : -delta;
    }
    return score;
}