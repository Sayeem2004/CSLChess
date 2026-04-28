#include <atomic>
#include <chrono>
#include <cstring>
#include <mutex>
#include <papi.h>
#include <thread>
#include <unordered_map>

#include "common.hpp"
#include "evaluate.hpp"


// Tools for enforcing time and cycle limits.
static std::once_flag papi_init_flag;
static void init_papi() {
    PAPI_library_init(PAPI_VER_CURRENT);
}
static std::atomic<bool> exceeded_budget{false};


// Transposition Table (TT) for caching previously evaluated positions.
enum TTFlag { TT_EXACT, TT_LOWER, TT_UPPER };
struct TTEntry {
    int         score;
    int         depth;
    TTFlag      flag;
    chess::Move best_move;
};
static std::unordered_map<uint64_t, TTEntry> tt;


// Returns a score in centipawns from the perspective of the side to move.
// `alpha` - lower bound (best score the maximising side is guaranteed so far)
// `beta`  - upper bound (best score the minimising side is guaranteed so far)
static int negamax(chess::Board& board, int depth, int alpha, int beta) {
    if (exceeded_budget) return 0; // Abort immediately — result will be discarded
    if (board.isRepetition(2) || board.isHalfMoveDraw()) return 0; // Draw

    const int alpha_orig = alpha;
    uint64_t hash        = board.hash();
    auto it              = tt.find(hash);

    if (it != tt.end() && it->second.depth >= depth) {
        const TTEntry& e = it->second;
        if (e.flag == TT_EXACT)                         return e.score;
        if (e.flag == TT_LOWER && e.score > alpha)      alpha = e.score;
        else if (e.flag == TT_UPPER && e.score < beta)  beta  = e.score;
        if (alpha >= beta)                              return e.score;
    }

    if (depth == 0) {
        if (!board.inCheck()) return engine_evaluate(board);
        chess::Movelist moves;
        chess::movegen::legalmoves(moves, board);
        if (moves.empty()) return -20000; // Checkmate at leaf
        return engine_evaluate(board); // In check but has escape moves
    }

    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);
    if (moves.empty()) {
        if (!board.inCheck()) return 0; // Stalemate
        return -20000 - (depth * 100); // Checkmate: prefer faster mates
    }

    // Move ordering: try the TT best move first
    if (it != tt.end()) {
        for (int i = 0; i < (int)moves.size(); i++) {
            if (moves[i] == it->second.best_move) {
                std::swap(moves[0], moves[i]);
                break;
            }
        }
    }

    chess::Move best_move = moves[0];
    int best = -1000000;

    for (const auto& move : moves) {
        board.makeMove(move);
        int score = -negamax(board, depth - 1, -beta, -alpha);
        board.unmakeMove(move);

        if (score > best) { best = score; best_move = move; }
        if (score > alpha) alpha = score;
        if (alpha >= beta) break; // Pruning cutoff
    }

    if (!exceeded_budget) {
        TTFlag flag = (best <= alpha_orig) ? TT_UPPER : (best >= beta) ? TT_LOWER : TT_EXACT;
        tt[hash]    = {best, depth, flag, best_move};
    }
    return best;
}


// Root search: returns best move and score at the given depth.
static chess::Move search_root(chess::Board& board, int depth) {
    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);

    // Try TT best move first for move ordering
    uint64_t root_hash = board.hash();
    auto it = tt.find(root_hash);
    if (it != tt.end()) {
        for (int i = 0; i < (int)moves.size(); i++) {
            if (moves[i] == it->second.best_move) {
                std::swap(moves[0], moves[i]);
                break;
            }
        }
    }

    chess::Move best_move = moves[0];
    int best_score = -1000000;
    int alpha      = -1000000;
    int beta       =  1000000;

    for (const auto& move : moves) {
        board.makeMove(move);
        int score = -negamax(board, depth-1, -beta, -alpha);
        board.unmakeMove(move);

        if (score > best_score) { best_score = score; best_move = move; }
        if (score > alpha) alpha = score;
    }

    // Store root in TT so the next iterative deepening depth orders moves correctly
    if (!exceeded_budget) tt[root_hash] = {best_score, depth, TT_EXACT, best_move};
    return best_move;
}


int best_move_alpha_beta_depth(const char* fen, int depth, char* out_move, int out_len) {
    chess::Board board;
    if (!board.setFen(fen)) return -2;

    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);
    if (moves.empty()) return -1;

    tt.clear();
    exceeded_budget = false;
    chess::Move best = search_root(board, depth);

    std::string uci  = chess::uci::moveToUci(best);
    std::strncpy(out_move, uci.c_str(), out_len);
    out_move[out_len-1] = '\0';
    return 0;
}


// Iterative Deepening: Each iteration seeds the TT with the best moves from
// the previous depth — should nearly half the chess branching factor.
int best_move_alpha_beta_time(const char* fen, int time_ms, char* out_move, int out_len) {
    chess::Board board;
    if (!board.setFen(fen)) return -2;

    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);
    if (moves.empty()) return -1;

    tt.clear();
    exceeded_budget = false;

    using clock = std::chrono::steady_clock;
    auto deadline = clock::now() + std::chrono::milliseconds(time_ms);

    std::thread timer([deadline]() {
        while (!exceeded_budget && clock::now() < deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        exceeded_budget = true;
    });

    chess::Move best = moves[0];
    for (int depth = 1; !exceeded_budget; depth++) {
        chess::Move candidate = search_root(board, depth);
        if (!exceeded_budget) best = candidate;
        auto it = tt.find(board.hash()); // Forced mate should terminate early
        if (it != tt.end() && std::abs(it->second.score) > 15000) break;
    }
    exceeded_budget = true;
    timer.join();

    std::string uci = chess::uci::moveToUci(best);
    std::strncpy(out_move, uci.c_str(), out_len);
    out_move[out_len-1] = '\0';
    return 0;
}


// Iterative Deepening: Each iteration seeds the TT with the best moves from
// the previous depth — should nearly half the chess branching factor.
int best_move_alpha_beta_cycles(const char* fen, int megacycle_budget, char* out_move, int out_len) {
    chess::Board board;
    if (!board.setFen(fen)) return -2;

    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);
    if (moves.empty()) return -1;

    std::call_once(papi_init_flag, init_papi);
    int event_set = PAPI_NULL;
    PAPI_create_eventset(&event_set);
    PAPI_add_event(event_set, PAPI_TOT_CYC);
    PAPI_start(event_set);

    tt.clear();
    exceeded_budget = false;

    long long cycle_budget = (long long)megacycle_budget * 1'000'000LL;
    chess::Move best = moves[0];
    for (int depth = 1; !exceeded_budget; depth++) {
        chess::Move candidate = search_root(board, depth);
        if (!exceeded_budget) best = candidate;

        long long cycles_used;
        PAPI_read(event_set, &cycles_used);
        if (cycles_used >= cycle_budget) break;

        auto it = tt.find(board.hash());
        if (it != tt.end() && std::abs(it->second.score) > 15000) break;
    }
    exceeded_budget = true;

    long long cycles_used;
    PAPI_stop(event_set, &cycles_used);
    PAPI_cleanup_eventset(event_set);
    PAPI_destroy_eventset(&event_set);

    std::string uci = chess::uci::moveToUci(best);
    std::strncpy(out_move, uci.c_str(), out_len);
    out_move[out_len-1] = '\0';
    return 0;
}
