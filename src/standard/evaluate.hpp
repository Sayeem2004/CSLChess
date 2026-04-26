#pragma once

// Returns a static evaluation of `board` in centipawns from the perspective
// of the side to move. Wraps Stockfish's Eval::evaluate(); falls back to
// material counting when the position is in check (Stockfish asserts otherwise).
int engine_evaluate(const chess::Board& board);
