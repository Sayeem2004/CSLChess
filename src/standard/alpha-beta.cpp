#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <mutex>
#include <thread>

#include <omp.h>
#ifdef USE_PAPI
#include <papi.h>
#endif

#include "common.hpp"
#include "evaluate.hpp"


static std::atomic<bool> exceeded_budget{false};
static std::once_flag omp_init_flag;
static void init_omp() { omp_set_max_active_levels(2); }
// Chess branching factor — sets the number of inner (root-parallel) threads per search_root call.
// Outer thread count = omp_get_max_threads() / BRANCH_FACTOR, used for Lazy SMP staggering.
static constexpr int BRANCH_FACTOR = 32;


#ifdef USE_PAPI
// Thread-local PAPI state so negamax can poll cycles without cross-thread access.
// Only the thread that calls PAPI_start sets tl_event_set != PAPI_NULL; others skip the check.
static thread_local int       tl_event_set    = PAPI_NULL;
static thread_local long long tl_cycle_budget = 0;
static thread_local int       tl_node_count   = 0;
static constexpr int PAPI_CHECK_NODES = 5000;
#endif

// Fixed-size lock-free TT. Concurrent reads/writes cause benign races (at worst a missed entry).
// `key` detects index collisions; `gen` invalidates stale entries without memset.
enum TTFlag : uint8_t { TT_EXACT, TT_LOWER, TT_UPPER };
struct alignas(32) TTEntry {
    uint64_t    key;
    int         score;
    int16_t     depth;
    uint8_t     gen;
    TTFlag      flag;
    chess::Move best_move;
};


// 4M entries * 18 bytes (aligned to 32) ≈ 128MB
static constexpr int TT_SIZE = 1 << 22;
static TTEntry tt[TT_SIZE];
static uint8_t tt_gen = 0;
static thread_local chess::Move tl_killers[AB_MAX_PLY][2];


// Returns a score in centipawns from the perspective of the side to move.
// `alpha` - lower bound (best score the maximising side is guaranteed so far)
// `beta`  - upper bound (best score the minimising side is guaranteed so far)
static int negamax(chess::Board& board, int depth, int alpha, int beta) {
    if (exceeded_budget.load(std::memory_order_relaxed)) return 0; // Abort immediately

    #ifdef USE_PAPI
    // Poll cycle counter every PAPI_CHECK_NODES nodes on the thread that owns the event set.
    if (tl_event_set != PAPI_NULL && ++tl_node_count >= PAPI_CHECK_NODES) {
        tl_node_count = 0;
        long long cycles_used;
        if (PAPI_read(tl_event_set, &cycles_used) == PAPI_OK) {
            if (cycles_used >= tl_cycle_budget) exceeded_budget = true;
        }
    }
    #endif

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

    int ki = std::min(depth, AB_MAX_PLY - 1);
    chess::Move tt_mv = tt_valid ? e.best_move : chess::Move{};
    std::sort(moves.begin(), moves.end(), [&](const chess::Move& a, const chess::Move& b) {
        return ab_score_move(board, a, tt_mv, tl_killers[ki]) >
               ab_score_move(board, b, tt_mv, tl_killers[ki]);
    });

    chess::Move best_move = moves[0];
    int         best      = -1000000;

    for (const auto& move : moves) {
        board.makeMove(move);
        int score = -negamax(board, depth - 1, -beta, -alpha);
        board.unmakeMove(move);

        if (score > best) { best = score; best_move = move; }
        if (score > alpha) alpha = score;
        if (alpha >= beta) { // Pruning/killer cutoff
            if (move.typeOf() != chess::Move::ENPASSANT &&
                board.at(move.to()).type() == chess::PieceType::NONE) {
                if (tl_killers[ki][0] != move) {
                    tl_killers[ki][1] = tl_killers[ki][0];
                    tl_killers[ki][0] = move;
                }
            }
            break;
        }
    }

    if (!exceeded_budget.load(std::memory_order_relaxed)) {
        TTFlag flag = (best <= alpha_orig) ? TT_UPPER : (best >= beta) ? TT_LOWER : TT_EXACT;
        tt[idx] = {hash, best, (int16_t)depth, tt_gen, flag, best_move};
    }
    return best;
}


// Root search with internal root parallelism: nthreads threads split the root moves.
static chess::Move search_root(chess::Board& board, int depth, int nthreads) {
    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);

    uint64_t  root_hash = board.hash();
    int       idx       = root_hash & (TT_SIZE - 1);
    TTEntry&  e         = tt[idx];
    bool      tt_valid  = (e.key == root_hash && e.gen == tt_gen);

    chess::Move tt_mv = tt_valid ? e.best_move : chess::Move{};
    std::sort(moves.begin(), moves.end(), [&](const chess::Move& a, const chess::Move& b) {
        return ab_score_move(board, a, tt_mv) > ab_score_move(board, b, tt_mv);
    });

    chess::Move      best_move  = moves[0];
    int              best_score = -1000000;
    std::atomic<int> depth_alpha{-1000000};

    // Each inner thread searches one root move; dynamic scheduling gives idle threads
    // the next unstarted move, hopefully load balancing better than a static split.
    #pragma omp parallel for num_threads(nthreads) schedule(dynamic, 1) \
            shared(best_move, best_score, depth_alpha)
    for (int i = 0; i < (int)moves.size(); i++) {
        if (exceeded_budget.load(std::memory_order_relaxed)) continue;
        chess::Board local = board;
        local.makeMove(moves[i]);

        int alpha = depth_alpha.load(std::memory_order_relaxed);
        int score = -negamax(local, depth-1, -1000000, -alpha);

        // Update shared alpha lower-bound lock-free (no critical needed — already atomic).
        int prev = depth_alpha.load(std::memory_order_relaxed);
        while (score > prev &&
               !depth_alpha.compare_exchange_weak(prev, score,
                   std::memory_order_relaxed, std::memory_order_relaxed));

        // Only best_score/best_move (non-atomic pair) needs mutual exclusion.
        #pragma omp critical
        {
            if (score > best_score) { best_score = score; best_move = moves[i]; }
        }
    }

    if (!exceeded_budget.load(std::memory_order_relaxed))
        tt[idx] = {root_hash, best_score, (int16_t)depth, tt_gen, TT_EXACT, best_move};
    return best_move;
}


int best_move_alpha_beta_depth(const char* fen, int depth, char* out_move, int out_len) {
    chess::Board board;
    if (!board.setFen(fen)) return -2;
    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);
    if (moves.empty()) return -1;

    // Increment to reduce memset frequency
    if (++tt_gen == 0) std::memset(tt, 0, sizeof(tt));
    exceeded_budget  = false;

    int inner_threads = std::min(BRANCH_FACTOR, omp_get_max_threads());
    int outer_threads = std::max(1, omp_get_max_threads() / inner_threads);

    static bool printed = false;
    if (!printed) { printed = true; fprintf(stderr, "[alpha-beta] threads: %d outer x %d inner = %d total\n", outer_threads, inner_threads, outer_threads * inner_threads); }

    chess::Move best       = moves[0];
    int         best_depth = 0;
    std::call_once(omp_init_flag, init_omp);
    #pragma omp parallel num_threads(outer_threads) shared(best, best_depth, exceeded_budget)
    {
        chess::Board local_board = board;
        int my_depth = 1 + omp_get_thread_num();

        while (!exceeded_budget && my_depth <= depth) {
            chess::Move candidate = search_root(local_board, my_depth, inner_threads);

            // Any thread may update best — take the result from the deepest completed search.
            if (!exceeded_budget) {
                #pragma omp critical
                {
                    if (my_depth > best_depth) {
                        best_depth = my_depth;
                        best = candidate;
                        if (best_depth >= depth) exceeded_budget = true;
                    }
                }
            }
            my_depth += outer_threads;

            // All threads check for forced mate so any thread can trigger early exit
            int root_idx = local_board.hash() & (TT_SIZE - 1);
            if (tt[root_idx].key == local_board.hash() && std::abs(tt[root_idx].score) > 15000) break;
        }
    }

    exceeded_budget = false;

    std::string uci = chess::uci::moveToUci(best);
    std::strncpy(out_move, uci.c_str(), out_len);
    out_move[out_len-1] = '\0';
    return best_depth;
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

    // Increment to reduce memset frequency
    if (++tt_gen == 0) std::memset(tt, 0, sizeof(tt));
    exceeded_budget = false;

    int inner_threads = std::min(BRANCH_FACTOR, omp_get_max_threads());
    int outer_threads = std::max(1, omp_get_max_threads() / inner_threads);

    static bool printed = false;
    if (!printed) { printed = true; fprintf(stderr, "[alpha-beta] threads: %d outer x %d inner = %d total\n", outer_threads, inner_threads, outer_threads * inner_threads); }

    using clock = std::chrono::steady_clock;
    auto deadline = clock::now() + std::chrono::milliseconds(time_ms);
    std::thread timer([deadline]() {
        while (!exceeded_budget && clock::now() < deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        exceeded_budget = true;
    });

    chess::Move best       = moves[0];
    int         best_depth = 0;
    std::call_once(omp_init_flag, init_omp);
    #pragma omp parallel num_threads(outer_threads) shared(best, best_depth, exceeded_budget)
    {
        chess::Board local_board = board;
        int my_depth = 1 + omp_get_thread_num();

        while (!exceeded_budget) {
            chess::Move candidate = search_root(local_board, my_depth, inner_threads);

            // Any thread may update best — take the result from the deepest completed search.
            if (!exceeded_budget) {
                #pragma omp critical
                {
                    if (my_depth > best_depth) {
                        best_depth = my_depth; best = candidate;
                    }
                }
            }
            my_depth += outer_threads;

            // All threads check for forced mate so any thread can trigger early exit
            int root_idx = local_board.hash() & (TT_SIZE - 1);
            if (tt[root_idx].key == local_board.hash() && std::abs(tt[root_idx].score) > 15000) break;
        }
    }

    exceeded_budget = true;
    timer.join();

    std::string uci = chess::uci::moveToUci(best);
    std::strncpy(out_move, uci.c_str(), out_len);
    out_move[out_len-1] = '\0';
    return best_depth;
}


// Same hybrid strategy with PAPI cycle counting.
// Outer thread 0 owns the PAPI event set and sets exceeded_budget when over budget.
int best_move_alpha_beta_cycles(const char* fen, int megacycle_budget, char* out_move, int out_len) {
    chess::Board board;
    if (!board.setFen(fen)) return -2;
    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);
    if (moves.empty()) return -1;

    // Increment to reduce memset frequency
    if (++tt_gen == 0) std::memset(tt, 0, sizeof(tt));
    exceeded_budget = false;

    int inner_threads = std::min(BRANCH_FACTOR, omp_get_max_threads());
    int outer_threads = std::max(1, omp_get_max_threads() / inner_threads);

    static bool printed = false;
    if (!printed) { printed = true; fprintf(stderr, "[alpha-beta] threads: %d outer x %d inner = %d total\n", outer_threads, inner_threads, outer_threads * inner_threads); }

    chess::Move best       = moves[0];
    int         best_depth = 0;

    // Safety timer: at 1 GHz, 1 megacycle = 1ms. Used as backstop if PAPI fails.
    // Joined after the parallel region (not detached) to avoid firing into the next call.
    long long time_ms = std::max(1LL, (long long)megacycle_budget);
    using clock = std::chrono::steady_clock;
    auto deadline = clock::now() + std::chrono::milliseconds(time_ms);
    std::thread timer([deadline]() {
        while (!exceeded_budget && clock::now() < deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        exceeded_budget = true;
    });

    #ifdef USE_PAPI
    PAPI_library_init(PAPI_VER_CURRENT);
    #endif

    std::call_once(omp_init_flag, init_omp);
    #pragma omp parallel num_threads(outer_threads) shared(best, best_depth, exceeded_budget)
    {
        chess::Board local_board = board;
        int my_depth = 1 + omp_get_thread_num();

        #ifdef USE_PAPI
        if (omp_get_thread_num() == 0) {
            tl_cycle_budget = (long long)megacycle_budget * 1'000'000LL;
            tl_node_count   = 0;
            tl_event_set    = PAPI_NULL;
            bool papi_ok = (PAPI_create_eventset(&tl_event_set) == PAPI_OK) &&
                           (PAPI_add_event(tl_event_set, PAPI_TOT_CYC) == PAPI_OK) &&
                           (PAPI_start(tl_event_set) == PAPI_OK);
            if (!papi_ok) {
                if (tl_event_set != PAPI_NULL) {
                    PAPI_cleanup_eventset(tl_event_set);
                    PAPI_destroy_eventset(&tl_event_set);
                }
                tl_event_set = PAPI_NULL;
            }
        }

        // Barrier so thread 0 finishes PAPI_start before any thread enters negamax.
        #pragma omp barrier
        #endif

        while (!exceeded_budget) {
            chess::Move candidate = search_root(local_board, my_depth, inner_threads);

            // Any thread may update best — take the result from the deepest completed search.
            if (!exceeded_budget) {
                #pragma omp critical
                {
                    if (my_depth > best_depth) { best_depth = my_depth; best = candidate; }
                }
            }
            my_depth += outer_threads;

            // All threads check for forced mate so any thread can trigger early exit
            int root_idx = local_board.hash() & (TT_SIZE - 1);
            if (tt[root_idx].key == local_board.hash() && std::abs(tt[root_idx].score) > 15000) break;
        }

        #ifdef USE_PAPI
        if (tl_event_set != PAPI_NULL) {
            long long cycles_used;
            PAPI_stop(tl_event_set, &cycles_used);
            PAPI_cleanup_eventset(tl_event_set);
            PAPI_destroy_eventset(&tl_event_set);
            tl_event_set = PAPI_NULL;
        }
        #endif
    }

    exceeded_budget = true;
    timer.join();

    std::string uci = chess::uci::moveToUci(best);
    std::strncpy(out_move, uci.c_str(), out_len);
    out_move[out_len-1] = '\0';
    return best_depth;
}
