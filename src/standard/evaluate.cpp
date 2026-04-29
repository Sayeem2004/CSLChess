#include <mutex>
#include <omp.h>

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
    Threads.set(omp_get_max_threads());
    Search::clear();
}


// Assign each OS thread a unique stable index for Stockfish's thread pool.
// omp_get_thread_num() is team-relative; with nested OMP every inner team has its own
// thread 0/1/2, so multiple outer threads collide on the same Stockfish Thread object.
static std::atomic<int> next_sf_thread_id{0};
static thread_local int sf_thread_id = next_sf_thread_id++;


int engine_evaluate(const chess::Board& board) {
    std::call_once(sf_init_flag, init_stockfish);

    Position pos;
    StateInfo si;
    int tid = sf_thread_id % (int)Threads.size();
    pos.set(board.getFen(), false, &si, Threads[tid]);

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
