#include "../../chess-library/include/chess.hpp"
#include "common.hpp"
#include "utils.h"

#include <mutex>
#include "../stockfish_11_src/src/bitboard.h"
#include "../stockfish_11_src/src/position.h"
#include "../stockfish_11_src/src/evaluate.h"
#include "../stockfish_11_src/src/thread.h"
#include "../stockfish_11_src/src/uci.h"
#include "../stockfish_11_src/src/endgame.h"
#include "../stockfish_11_src/src/search.h"

namespace PSQT {
    void init();
}

static std::once_flag sf_init_flag;
static void init_stockfish() {
    UCI::init(Options);
    PSQT::init();
    Bitboards::init();
    Position::init();
    Bitbases::init();
    Endgames::init();
    Threads.set(1);
    Search::clear();
}

// eval function wrapping Stockfish's evaluate
int evaluate(const chess::Board& board) {
    std::call_once(sf_init_flag, init_stockfish);

    std::string fen = board.getFen();
    Position pos;
    StateInfo si;
    pos.set(fen, false, &si, Threads.main());

    if (pos.checkers()) {
        // Stockfish's evaluate() asserts!pos.checkers(), fallback to material count if in check
        constexpr int PIECE_VALUES[] = {0, 100, 320, 330, 500, 900, 20000};
        constexpr chess::PieceType PIECE_TYPES[] = {
            chess::PieceType::PAWN, chess::PieceType::KNIGHT, chess::PieceType::BISHOP,
            chess::PieceType::ROOK, chess::PieceType::QUEEN, chess::PieceType::KING
        };
        int score = 0;
        chess::Color stm = board.sideToMove();
        for (int pt = 0; pt < 6; ++pt) {
            chess::PieceType type = PIECE_TYPES[pt];
            int white_count = board.pieces(type, chess::Color::WHITE).count();
            int black_count = board.pieces(type, chess::Color::BLACK).count();
            int delta = (white_count - black_count) * PIECE_VALUES[pt + 1];
            score += (stm == chess::Color::WHITE) ? delta : -delta;
        }
        return score;
    }

    return Eval::evaluate(pos);
}