#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <thread>

#include <omp.h>
#ifdef USE_PAPI
#include <papi.h>
#endif

#include "common.hpp"
#include "evaluate.hpp"


static std::atomic<bool> exceeded_budget{false};
// Chess branching factor — sets the number of inner (root-parallel) threads per search_root call.
// Outer thread count = omp_get_max_threads() / BRANCH_FACTOR, used for Lazy SMP staggering.
static constexpr int BRANCH_FACTOR = 32;


// Fixed-size lock-free TT. Concurrent reads/writes cause benign races (at worst a missed entry).
// `key` detects index collisions; `gen` invalidates stale entries without memset.
enum TTFlag : uint8_t { TT_EXACT, TT_LOWER, TT_UPPER };
struct TTEntry {
    uint64_t    key;
    int         score;
    int16_t     depth;
    uint8_t     gen;
    TTFlag      flag;
    chess::Move best_move;
};


// 4M entries * (16B + ~32B best move) = ~200MB
static constexpr int TT_SIZE = 1 << 22;
static TTEntry tt[TT_SIZE];
static uint8_t tt_gen = 0;


// Returns a score in centipawns from the perspective of the side to move.
// `alpha` - lower bound (best score the maximising side is guaranteed so far)
// `beta`  - upper bound (best score the minimising side is guaranteed so far)
static int negamax(chess::Board& board, int depth, int alpha, int beta) {
    if (exceeded_budget) return 0; // Abort immediately
    if (board.isRepetition(2) || board.isHalfMoveDraw()) return 0; // Draw

    const int alpha_orig = alpha;
    uint64_t  hash       = board.hash();
    int       idx        = hash & (TT_SIZE - 1);
    TTEntry&  e          = tt[idx];
    bool      tt_valid   = (e.key == hash && e.gen == tt_gen);

    if (tt_valid && e.depth >= depth) {
        if (e.flag == TT_EXACT)                          return e.score;
        if (e.flag == TT_LOWER && e.score > alpha)       alpha = e.score;
        else if (e.flag == TT_UPPER && e.score < beta)   beta  = e.score;
        if (alpha >= beta)                               return e.score;
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

    if (tt_valid) {
        for (int i = 0; i < (int)moves.size(); i++) {
            if (moves[i] == e.best_move) {
                std::swap(moves[0], moves[i]);
                break;
            }
        }
    }

    chess::Move best_move = moves[0];
    int         best      = -1000000;

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
        tt[idx] = {hash, best, (int16_t)depth, tt_gen, flag, best_move};
    }
    return best;
}


// Root search with internal root parallelism: nthreads threads split the root moves.
static chess::Move search_root(chess::Board& board, int depth, int nthreads) {
    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);

    uint64_t root_hash = board.hash();
    int      idx       = root_hash & (TT_SIZE - 1);
    TTEntry& e         = tt[idx];
    if (e.key == root_hash && e.gen == tt_gen) {
        for (int i = 0; i < (int)moves.size(); i++) {
            if (moves[i] == e.best_move) {
                std::swap(moves[0], moves[i]);
                break;
            }
        }
    }

    chess::Move      best_move  = moves[0];
    int              best_score = -1000000;
    std::atomic<int> depth_alpha{-1000000};

    // Each inner thread searches one root move; dynamic scheduling gives idle threads
    // the next unstarted move, hopefully load balancing better than a static split.
    #pragma omp parallel for num_threads(nthreads) schedule(dynamic, 1) \
            shared(best_move, best_score, depth_alpha)
    for (int i = 0; i < (int)moves.size(); i++) {
        if (exceeded_budget) continue;
        chess::Board local = board;
        local.makeMove(moves[i]);

        int alpha = depth_alpha.load();
        int score = -negamax(local, depth-1, -1000000, -alpha);

        #pragma omp critical
        {
            if (score > best_score) { best_score = score; best_move = moves[i]; }
            int prev = depth_alpha.load();
            while (score > prev && !depth_alpha.compare_exchange_weak(prev, score));
        }
    }

    if (!exceeded_budget)
        tt[idx] = {root_hash, best_score, (int16_t)depth, tt_gen, TT_EXACT, best_move};
    return best_move;
}


int best_move_alpha_beta_depth(const char* fen, int depth, char* out_move, int out_len) {
    chess::Board board;
    if (!board.setFen(fen)) return -2;
    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);
    if (moves.empty()) return -1;

    tt_gen++;
    exceeded_budget  = false;
    
    static bool printed = false;
    if (!printed) { printed = true; fprintf(stderr, "[alpha-beta] threads: %d (root-parallel)\n", omp_get_max_threads()); }
    chess::Move best = search_root(board, depth, omp_get_max_threads());

    std::string uci = chess::uci::moveToUci(best);
    std::strncpy(out_move, uci.c_str(), out_len);
    out_move[out_len-1] = '\0';
    return 0;
}


// Hybrid Lazy SMP + Root Parallelism.
// Outer threads = max_threads / BRANCH_FACTOR, each doing iterative deepening at staggered depths.
// Inner threads = BRANCH_FACTOR, splitting root moves in parallel within each search_root call..
int best_move_alpha_beta_time(const char* fen, int time_ms, char* out_move, int out_len) {
    chess::Board board;
    if (!board.setFen(fen)) return -2;
    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);
    if (moves.empty()) return -1;

    tt_gen++;
    exceeded_budget = false;

    int outer_threads = std::max(1, omp_get_max_threads() / BRANCH_FACTOR);
    int inner_threads = BRANCH_FACTOR;
    
    static bool printed = false;
    if (!printed) { printed = true; fprintf(stderr, "[alpha-beta] threads: %d outer x %d inner = %d total\n", outer_threads, inner_threads, outer_threads * inner_threads); }
    
    using clock = std::chrono::steady_clock;
    auto deadline = clock::now() + std::chrono::milliseconds(time_ms);
    std::thread timer([deadline]() {
        while (!exceeded_budget && clock::now() < deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        exceeded_budget = true;
    });
    chess::Move best = moves[0];

    omp_set_nested(true);
    omp_set_max_active_levels(2);
    #pragma omp parallel num_threads(outer_threads) shared(best, exceeded_budget)
    {
        chess::Board local_board = board;
        int my_depth = 1 + omp_get_thread_num(); // stagger: thread 0→1, thread 1→2, ...

        while (!exceeded_budget) {
            chess::Move candidate = search_root(local_board, my_depth, inner_threads);

            if (omp_get_thread_num() == 0 && !exceeded_budget)
                best = candidate;

            my_depth += outer_threads;

            int root_idx = local_board.hash() & (TT_SIZE - 1);
            if (tt[root_idx].key == local_board.hash() && std::abs(tt[root_idx].score) > 15000) break;
        }
    }

    exceeded_budget = true;
    timer.join();

    std::string uci = chess::uci::moveToUci(best);
    std::strncpy(out_move, uci.c_str(), out_len);
    out_move[out_len-1] = '\0';
    return 0;
}


// Same hybrid strategy with PAPI cycle counting.
// Outer thread 0 owns the PAPI event set and sets exceeded_budget when over budget.
int best_move_alpha_beta_cycles(const char* fen, int megacycle_budget, char* out_move, int out_len) {
    chess::Board board;
    if (!board.setFen(fen)) return -2;
    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);
    if (moves.empty()) return -1;

    tt_gen++;
    exceeded_budget = false;

    int outer_threads = std::max(1, omp_get_max_threads() / BRANCH_FACTOR);
    int inner_threads = BRANCH_FACTOR;
    
    static bool printed = false;
    if (!printed) { printed = true; fprintf(stderr, "[alpha-beta] threads: %d outer x %d inner = %d total\n", outer_threads, inner_threads, outer_threads * inner_threads); }
    
    chess::Move best = moves[0];
    omp_set_nested(true);
    omp_set_max_active_levels(2);

    #pragma omp parallel num_threads(outer_threads) shared(best, exceeded_budget)
    {
        chess::Board local_board = board;
        int my_depth = 1 + omp_get_thread_num();

        #ifdef USE_PAPI
        int       event_set   = PAPI_NULL;
        long long cycle_budget = (long long)megacycle_budget * 1'000'000LL;
        if (omp_get_thread_num() == 0) {
            PAPI_library_init(PAPI_VER_CURRENT);
            PAPI_create_eventset(&event_set);
            PAPI_add_event(event_set, PAPI_TOT_CYC);
            PAPI_start(event_set);
        }
        #endif

        while (!exceeded_budget) {
            chess::Move candidate = search_root(local_board, my_depth, inner_threads);

            if (omp_get_thread_num() == 0 && !exceeded_budget) {
                best = candidate;
                #ifdef USE_PAPI
                long long cycles_used;
                PAPI_read(event_set, &cycles_used);
                if (cycles_used >= cycle_budget) exceeded_budget = true;
                #endif
            }

            my_depth += outer_threads;

            // All threads check for forced mate so any thread can trigger early exit
            int root_idx = local_board.hash() & (TT_SIZE - 1);
            if (tt[root_idx].key == local_board.hash() && std::abs(tt[root_idx].score) > 15000) break;
        }

        #ifdef USE_PAPI
        if (omp_get_thread_num() == 0 && event_set != PAPI_NULL) {
            long long cycles_used;
            PAPI_stop(event_set, &cycles_used);
            PAPI_cleanup_eventset(event_set);
            PAPI_destroy_eventset(&event_set);
        }
        #endif
    }

    exceeded_budget = true;

    std::string uci = chess::uci::moveToUci(best);
    std::strncpy(out_move, uci.c_str(), out_len);
    out_move[out_len-1] = '\0';
    return 0;
}
