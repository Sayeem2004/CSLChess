#include <mutex>

#include "../../stockfish/stockfish-11-src/bitboard.h"
#include "../../stockfish/stockfish-11-src/endgame.h"
#include "../../stockfish/stockfish-11-src/evaluate.h"
#include "../../stockfish/stockfish-11-src/position.h"
#include "../../stockfish/stockfish-11-src/search.h"
#include "../../stockfish/stockfish-11-src/thread.h"
#include "../../stockfish/stockfish-11-src/uci.h"

#include "common.hpp"
#include "evaluate.hpp"


namespace PSQT { void init(); }
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


int engine_evaluate(const chess::Board& board) {
    std::call_once(sf_init_flag, init_stockfish);

    Position pos;
    StateInfo si;
    pos.set(board.getFen(), false, &si, Threads.main());

    // Stockfish's Eval::evaluate() asserts !pos.checkers(), so fall back to
    // simple material counting when the side to move is in check.
    if (pos.checkers()) {
        static constexpr int PIECE_VALUES[] = {100, 320, 330, 500, 900, 20000};
        static constexpr chess::PieceType PIECE_TYPES[] = {
            chess::PieceType::PAWN,   chess::PieceType::KNIGHT, chess::PieceType::BISHOP,
            chess::PieceType::ROOK,   chess::PieceType::QUEEN,  chess::PieceType::KING,
        };

        int score = 0;
        chess::Color stm = board.sideToMove();
        for (int i = 0; i < 6; i++) {
            int delta = (board.pieces(PIECE_TYPES[i], chess::Color::WHITE).count()
                         - board.pieces(PIECE_TYPES[i], chess::Color::BLACK).count())
                         * PIECE_VALUES[i];
            score += (stm == chess::Color::WHITE) ? delta : -delta;
        }
        return score;
    }

    return Eval::evaluate(pos);
}
