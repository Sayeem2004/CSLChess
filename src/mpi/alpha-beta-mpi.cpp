#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <mpi.h>
#include <omp.h>

#ifdef __x86_64__
#include <x86intrin.h>
static inline uint64_t read_cycles() { return __rdtsc(); }
#else
// ARM fallback: nanoseconds from steady_clock (1 megacycle ~ 1ms ~ 1,000,000 ns at ~1GHz)
static inline uint64_t read_cycles() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
#endif
static inline uint64_t budget_to_cycles(int megacycles) { return (uint64_t)megacycles * 1'000'000ULL; }

#include "../standard/common.hpp"
#include "../standard/evaluate.hpp"


static std::atomic<bool> exceeded_budget{false};
// Chess branching factor — sets the number of inner (root-parallel) threads per search_root call.
// Outer thread count = omp_get_max_threads() / BRANCH_FACTOR, used for Lazy SMP staggering.
static constexpr int     BRANCH_FACTOR = 32;


static uint64_t          g_cycle_start  = 0;
static uint64_t          g_cycle_budget = 0;
static thread_local int  tl_node_count  = 0;
static constexpr int     CHECK_NODES    = 100;


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
static constexpr int TT_SIZE = 1 << 22; // 4M entries ≈ 128MB
static TTEntry       tt[TT_SIZE];
static uint8_t       tt_gen = 0;
static thread_local chess::Move tl_killers[AB_MAX_PLY][2];


// One-sided RMA window over the TT — created once in main(), kept locked for passive-target puts.
static MPI_Win g_tt_win = MPI_WIN_NULL;
static int     g_rank   = 0;
static int     g_nranks = 1;


// Push our root TT entry to all other ranks asynchronously (passive target, no barrier).
static void push_root_entry(uint64_t root_hash) {
    MPI_Aint idx  = (MPI_Aint)(root_hash & (TT_SIZE - 1));
    TTEntry& src  = tt[idx];
    if (src.key != root_hash) return;

    MPI_Aint disp = idx * (MPI_Aint)sizeof(TTEntry);
    for (int r = 0; r < g_nranks; r++) {
        if (r == g_rank) continue;
        MPI_Put(&src, sizeof(TTEntry), MPI_BYTE, r, disp, sizeof(TTEntry), MPI_BYTE, g_tt_win);
    }
    MPI_Win_flush_all(g_tt_win);
}


// Returns a score in centipawns from the perspective of the side to move.
// `alpha` - lower bound (best score the maximising side is guaranteed so far)
// `beta`  - upper bound (best score the minimising side is guaranteed so far)
static int negamax(chess::Board& board, int depth, int alpha, int beta) {
    if (exceeded_budget.load(std::memory_order_relaxed)) return 0; // Abort immediately

    if (g_cycle_budget && ++tl_node_count >= CHECK_NODES) {
        tl_node_count = 0;
        if (read_cycles() - g_cycle_start >= g_cycle_budget)
            exceeded_budget = true;
    }

    if (board.isRepetition(2) || board.isHalfMoveDraw()) return 0; // Draw

    const int alpha_orig = alpha;
    uint64_t  hash       = board.hash();
    int       idx        = hash & (TT_SIZE - 1);
    TTEntry&  e          = tt[idx];
    bool      tt_valid   = (e.key == hash && e.gen == tt_gen);

    if (tt_valid && e.depth >= depth) {
        if (e.flag == TT_EXACT)                        return e.score;
        if (e.flag == TT_LOWER && e.score > alpha)     alpha = e.score;
        else if (e.flag == TT_UPPER && e.score < beta) beta  = e.score;
        if (alpha >= beta)                             return e.score;
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
        if (alpha >= beta) {
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

    uint64_t    root_hash = board.hash();
    int         idx       = root_hash & (TT_SIZE - 1);
    TTEntry&    e         = tt[idx];
    bool        tt_valid  = (e.key == root_hash && e.gen == tt_gen);
    chess::Move tt_mv     = tt_valid ? e.best_move : chess::Move{};

    std::sort(moves.begin(), moves.end(), [&](const chess::Move& a, const chess::Move& b) {
        return ab_score_move(board, a, tt_mv) > ab_score_move(board, b, tt_mv);
    });

    chess::Move      best_move  = moves[0];
    int              best_score = -1000000;
    std::atomic<int> depth_alpha{-1000000};

    // Each inner thread searches one root move; dynamic scheduling gives idle threads
    // the next unstarted move, hopefully load balancing better than a static split.
    int actual_threads = std::min(nthreads, (int)moves.size());
    #pragma omp parallel for num_threads(actual_threads) schedule(dynamic, 1) \
            shared(best_move, best_score, depth_alpha)
    for (int i = 0; i < (int)moves.size(); i++) {
        if (exceeded_budget.load(std::memory_order_relaxed)) continue;
        chess::Board local = board;
        local.makeMove(moves[i]);
        int score = -negamax(local, depth - 1, -1000000, -depth_alpha.load(std::memory_order_relaxed));

        // Update shared alpha lower-bound lock-free (no critical needed — already atomic).
        int prev = depth_alpha.load(std::memory_order_relaxed);
        while (score > prev &&
               !depth_alpha.compare_exchange_weak(prev, score,
                   std::memory_order_relaxed, std::memory_order_relaxed));

        // Only best_score/best_move (non-atomic pair) needs mutual exclusion.
        #pragma omp critical
        {
            if (score > best_score) {
                best_score = score;
                best_move  = moves[i];
                if (!exceeded_budget.load(std::memory_order_relaxed)) {
                    tt[idx] = {root_hash, best_score, (int16_t)depth, tt_gen, TT_EXACT, best_move};
                    push_root_entry(root_hash);
                }
            }
        }
    }

    if (!exceeded_budget.load(std::memory_order_relaxed))
        tt[idx] = {root_hash, best_score, (int16_t)depth, tt_gen, TT_EXACT, best_move};
    return best_move;
}


// Result each rank shares after completing a depth iteration.
struct DepthResult {
    int  depth;
    int  score;
    char move[8];
};


// Sync one completed depth across all ranks; updates best and propagates exceeded_budget.
// Returns true if search should stop.
static bool sync_depth(const chess::Board& board, chess::Move candidate, int my_depth,
                       int& best_depth, int& best_score, std::string& best_move_str,
                       int target_depth = 1_000_000_000) {
    int candidate_score = -1000000;
    {
        uint64_t h  = board.hash();
        TTEntry& te = tt[h & (TT_SIZE - 1)];
        if (te.key == h) candidate_score = te.score;
    }

    bool my_exceeded = exceeded_budget.load();
    DepthResult local;
    local.depth = my_exceeded ? 0 : my_depth; // Don't report incomplete depths
    local.score = candidate_score;
    std::string uci = chess::uci::moveToUci(candidate);
    std::strncpy(local.move, uci.c_str(), sizeof(local.move) - 1);
    local.move[sizeof(local.move) - 1] = '\0';

    std::vector<DepthResult> all(g_nranks);
    MPI_Allgather(&local, sizeof(DepthResult), MPI_BYTE,
                  all.data(), sizeof(DepthResult), MPI_BYTE, MPI_COMM_WORLD);
    for (const auto& r : all) {
        if (r.depth > best_depth || (r.depth == best_depth && r.score > best_score)) {
            best_depth    = r.depth;
            best_score    = r.score;
            best_move_str = std::string(r.move);
        }
    }

    int local_ex = (my_exceeded || best_depth >= target_depth) ? 1 : 0, global_ex = 0;
    MPI_Allreduce(&local_ex, &global_ex, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (global_ex) exceeded_budget = true;

    // Inject global best into every rank's TT root entry so the next depth
    // starts with the best-known move for ordering — hopefully equivalent to shared-TT Lazy SMP.
    if (best_depth > 0) {
        chess::Move best_mv = chess::uci::uciToMove(board, best_move_str);
        uint64_t h = board.hash();
        tt[h & (TT_SIZE - 1)] = {h, best_score, (int16_t)best_depth, tt_gen, TT_EXACT, best_mv};
    }

    return global_ex != 0;
}


// MPI ranks replace outer OMP threads: rank r searches depths r+1, r+1+nranks, ...
// Within each rank, inner_threads do root parallelism (same as alpha-beta.cpp).
static std::string search_depth(const chess::Board& board, int depth) {
    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);
    if (moves.empty()) return "";
    if (moves.size() == 1) return chess::uci::moveToUci(moves[0]);

    // Increment to reduce memset frequency
    if (++tt_gen == 0) std::memset(tt, 0, sizeof(tt));
    exceeded_budget = false;

    int inner_threads = std::min(BRANCH_FACTOR, omp_get_max_threads());

    std::string best_move_str = chess::uci::moveToUci(moves[0]);
    int best_depth = 0;
    int best_score = -1000000;
    int my_depth   = g_rank + 1;

    // All ranks loop in lockstep — ranks past the target submit depth=0 dummy results
    // so the MPI_Allgather in sync_depth always has all ranks participating.
    while (!exceeded_budget) {
        chess::Move candidate = moves[0];
        int report_depth = 0;
        if (my_depth <= depth) {
            chess::Board local_board = board;
            candidate    = search_root(local_board, my_depth, inner_threads);
            report_depth = my_depth;
        }
        if (sync_depth(board, candidate, report_depth, best_depth, best_score, best_move_str, depth)) break;
        my_depth += g_nranks;
    }

    exceeded_budget = false;
    return best_move_str;
}


static std::string search_time(const chess::Board& board, int time_ms) {
    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);
    if (moves.empty()) return "";
    if (moves.size() == 1) return chess::uci::moveToUci(moves[0]);

    // Increment to reduce memset frequency
    if (++tt_gen == 0) std::memset(tt, 0, sizeof(tt));
    exceeded_budget = false;

    int inner_threads = std::min(BRANCH_FACTOR, omp_get_max_threads());

    using clock = std::chrono::steady_clock;
    auto deadline = clock::now() + std::chrono::milliseconds(time_ms);
    std::thread timer([deadline]() {
        while (!exceeded_budget && clock::now() < deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(min(10, time_ms)));
        exceeded_budget = true;
    });

    std::string best_move_str = chess::uci::moveToUci(moves[0]);
    int best_depth = 0;
    int best_score = -1000000;
    int my_depth   = g_rank + 1;

    while (!exceeded_budget) {
        chess::Board local_board = board;
        chess::Move  candidate   = search_root(local_board, my_depth, inner_threads);
        if (sync_depth(board, candidate, my_depth, best_depth, best_score, best_move_str)) break;
        my_depth += g_nranks;
    }

    exceeded_budget = true;
    timer.join();
    exceeded_budget = false;
    return best_move_str;
}


static std::string search_cycles(const chess::Board& board, int megacycle_budget) {
    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);
    if (moves.empty()) return "";
    if (moves.size() == 1) return chess::uci::moveToUci(moves[0]);

    // Increment to reduce memset frequency
    if (++tt_gen == 0) std::memset(tt, 0, sizeof(tt));
    exceeded_budget = false;

    int inner_threads = std::min(BRANCH_FACTOR, omp_get_max_threads());

    g_cycle_start  = read_cycles();
    g_cycle_budget = budget_to_cycles(megacycle_budget);

    std::string best_move_str = chess::uci::moveToUci(moves[0]);
    int best_depth = 0;
    int best_score = -1000000;
    int my_depth   = g_rank + 1;

    while (!exceeded_budget) {
        chess::Board local_board = board;
        chess::Move  candidate   = search_root(local_board, my_depth, inner_threads);
        if (sync_depth(board, candidate, my_depth, best_depth, best_score, best_move_str)) break;
        my_depth += g_nranks;
    }

    g_cycle_budget  = 0;
    exceeded_budget = false;
    return best_move_str;
}


int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_nranks);
    omp_set_max_active_levels(2);

    // Create a passive-target RMA window over the TT so push_root_entry can MPI_Put
    // directly into other ranks' TT memory without a matching receive call.
    MPI_Win_create(tt, (MPI_Aint)TT_SIZE * sizeof(TTEntry), 1,
                   MPI_INFO_NULL, MPI_COMM_WORLD, &g_tt_win);
    MPI_Win_lock_all(MPI_MODE_NOCHECK, g_tt_win);

    if (g_rank == 0) {
        fprintf(stderr, "[mpi-alpha-beta] %d ranks x %d OMP threads each\n",
                g_nranks, omp_get_max_threads());
    }

    constexpr int BUF_SIZE = 4096;
    char buf[BUF_SIZE];

    while (true) {
        // Rank 0 reads the next command and broadcasts to all ranks.
        int len = 0;
        if (g_rank == 0) {
            std::string line;
            if (!std::getline(std::cin, line) || line == "quit") {
                len = -1;
            } else {
                std::strncpy(buf, line.c_str(), BUF_SIZE - 1);
                buf[BUF_SIZE - 1] = '\0';
                len = (int)strlen(buf);
            }
        }

        MPI_Bcast(&len, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (len < 0) break; // All ranks need to quit so after broadcast
        MPI_Bcast(buf, BUF_SIZE, MPI_CHAR, 0, MPI_COMM_WORLD);

        // Parse: "<mode> <budget> <fen...>"
        std::istringstream ss(std::string(buf));
        std::string mode;
        int         budget;
        std::string fen;

        ss >> mode >> budget;
        std::getline(ss, fen);
        if (!fen.empty() && fen[0] == ' ') fen = fen.substr(1);

        chess::Board board;
        if (!board.setFen(fen)) {
            if (g_rank == 0) std::cout << "error\n" << std::flush;
            continue;
        }

        std::string move;
        if      (mode == "depth")  move = search_depth(board, budget);
        else if (mode == "time")   move = search_time(board, budget);
        else if (mode == "cycles") move = search_cycles(board, budget);
        else { if (g_rank == 0) std::cout << "error\n" << std::flush; continue; }

        if (g_rank == 0) std::cout << move << "\n" << std::flush;
    }

    MPI_Win_unlock_all(g_tt_win);
    MPI_Win_free(&g_tt_win);
    MPI_Finalize();
    return 0;
}
