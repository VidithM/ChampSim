#include <algorithm>
#include <cassert>
#include <map>
#include <set>
#include <vector>
#include <random>

#include "cache.h"

namespace {
    constexpr int MAX_RRIP = 7;
    constexpr int MAX_SATURATING = 7;
    constexpr int IP_HASH_MAX = (1 << 13) - 1;
    constexpr int SAMPLE_SIZE = 64; // run OPTGen only on 64 random sets to save space


    std::map<CACHE*, std::vector<uint64_t>> samples;                // list of sampled set indices (0.5 KB)
    std::map<CACHE*, std::vector<std::vector<uint64_t>>> occupancy; // occupancy vector for sampled sets - used for OPTGen (16 KB)
    std::map<CACHE*, std::vector<uint64_t>> last_instr;             // last instruction to access some line (uses PC hash to save space) (65 KB)
    

    std::map<CACHE*, std::vector<uint8_t>> loaded;                  // whether a line has been accessed before (32 KB)
    std::map<CACHE*, std::vector<uint32_t>> priority;               // RRIP line priorities - used to select victim (32 KB)
    std::map<CACHE*, std::vector<uint32_t>> pc_prediction;          // Saturating counters for hashed program counters - used for predictions (8 KB)
                                                                    // - if the third bit is a 1 (i.e. values 4-7), cache-friendly, else cache-averse (values 0-3)

    std::map<CACHE*, std::vector<std::vector<uint64_t>>> history;   // history of accesses for sampled sets: used for OPTGen to determine last access time
                                                                    // on a line. (65 KB)
    
    std::map<CACHE*, std::vector<uint64_t>> current_time;           // maintains current time in each sampled set (0.5 KB). Used to index into history and
                                                                    // occupancy vectors
}

// produces a 13-bit XOR hash of a 64 bit program counter
uint64_t hash_instr(uint64_t ip){
    uint64_t hash = ip;
    hash ^= (hash >> 16);
    hash ^= (hash >> 32);
    return (hash & ((1 << 13) - 1));
}

void generate_sample(std::vector<uint64_t> &ans, uint32_t max){
    std::set<uint32_t> used;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> dis(0, max - 1);

    for(int i = 0; i < SAMPLE_SIZE; i++){
        uint32_t cand = dis(gen);
        if(used.count(cand)){ continue; }
        used.insert(cand);
        ans.push_back(cand);
    }
}

void CACHE::initialize_replacement() {
    ::occupancy[this].resize(SAMPLE_SIZE);
    ::last_instr[this].resize(NUM_SET * NUM_WAY);

    ::loaded[this].resize(NUM_SET * NUM_WAY);
    std::fill(::loaded[this].begin(), ::loaded[this].end(), false);

    ::priority[this].resize(NUM_SET * NUM_WAY);
    std::fill(::priority[this].begin(), ::priority[this].end(), 0);

    ::pc_prediction[this].resize(IP_HASH_MAX);
    std::fill(::pc_prediction[this].begin(), ::pc_prediction[this].end(), 4); // start all PCs at the lowest confidence cache-friendly

    ::history[this].resize(SAMPLE_SIZE);
    ::current_time[this].resize(SAMPLE_SIZE);

    generate_sample(::samples[this], NUM_SET - 1);

    for(size_t i = 0; i < SAMPLE_SIZE; i++){
        ::current_time[this][i] = 0;
        ::occupancy[this][i].resize(NUM_WAY * 8);
        ::history[this][i].resize(NUM_WAY * 8);
    }
}

uint32_t CACHE::find_victim(uint32_t triggering_cpu, uint64_t instr_id, uint32_t set, const BLOCK* current_set, uint64_t ip, uint64_t full_addr, uint32_t type) {
    // scan over all lines in priority
    // select the one with the max RRIP
    // if the last_instr of this chosen line is predicted as cache-friendly, need to detrain that instruction by decrementing its counter
    uint32_t set_start = set * NUM_WAY;
    int victim = -1;
    int max_RRIP = -1;
    for(size_t i = set_start; i < set_start + NUM_WAY; i++){
        if((int) ::priority[this][i] > max_RRIP){
            max_RRIP = ::priority[this][i];
            victim = i;
        }
    }
    if(max_RRIP == MAX_RRIP){
        return victim;
    }
    // max_RRIP is not 7 -> the last instr was predicted as cache friendly. need to detrain if the set is sampled
    bool is_sampled = false;
    for(size_t i = 0; i < SAMPLE_SIZE; i++){
        if(::samples[this][i] == set){
            is_sampled = true;
            break;
        }
    }
    if(is_sampled){
        uint64_t instr = ::last_instr[this][victim];
        if(::pc_prediction[this][instr] > 0){ ::pc_prediction[this][instr]--; }
    }
    return victim;
}   

// Called on each cache hit or cache fill (write)
void CACHE::update_replacement_state(uint32_t triggering_cpu, uint32_t set, uint32_t way, uint64_t full_addr, uint64_t ip, uint64_t victim_addr, uint32_t type, uint8_t hit) {
    uint64_t line_id = NUM_WAY * set + way;
    if(ip == 0){
        // skip prefetches from other levels
        return;
    }
    uint64_t pc_hashed = hash_instr(ip);
    ::last_instr[this][line_id] = pc_hashed;

    bool first_load = true;
    if(::loaded[this][line_id]){
        first_load = false; 
    }
    ::loaded[this][line_id] = true;
    if(first_load){
        // do nothing for the first load
        return;
    }

    bool is_sampled = false;
    uint32_t sample_id;
    for(size_t i = 0; i < SAMPLE_SIZE; i++){
        if(::samples[this][i] == set){
            is_sampled = true;
            sample_id = i;
            break;
        }
    }

    if(is_sampled){
        // run OPTGen
        uint64_t now = ::current_time[this][sample_id];
        size_t idx = now;
        if(idx == 0){
            idx = NUM_WAY * 8 - 1;
        } else {
            idx--;
        }
        bool is_full = false;
        int found_line = -1;
        for(int i = 0; i < (int) NUM_WAY * 8; i++){
            if(::history[this][sample_id][idx] == 0){
                // no instruction present; reached end of history
                break;
            }
            if(::history[this][sample_id][idx] == line_id){
                found_line = idx;
            }
            if(::occupancy[this][sample_id][idx] == NUM_WAY){
                is_full = true;
            }
            if(found_line != -1){
                break;
            }
            if(idx == 0){
                idx = NUM_WAY * 8 - 1;
            } else {
                idx--;
            }
        }
        // write a 0 to the current time
        ::history[this][sample_id][now] = line_id;
        ::occupancy[this][sample_id][now] = 0;
        // determine if hit or miss
        bool did_hit = false;
        if(found_line){
            if(!is_full){
                did_hit = true;
            }
        }
        if(did_hit){
            // increment the occupancy of everything in the usage interval
            idx = now;
            if(idx == 0){
                idx = NUM_WAY * 8 - 1;
            } else {
                idx--;
            }
            while(true){
                ::occupancy[this][sample_id][idx]++;
                if((int) idx == found_line){
                    break;
                }
                if(idx == 0){
                    idx = NUM_WAY * 8 - 1;
                } else {
                    idx--;
                }
            }
            // train PC positively
            if(::pc_prediction[this][pc_hashed] < 7){
                ::pc_prediction[this][pc_hashed]++;
            }
        } else {
            // set current occupancy entry to 1 since no bypassing
            ::occupancy[this][sample_id][now] = 1;

            // train PC negatively
            if(::pc_prediction[this][pc_hashed] > 0){
                ::pc_prediction[this][pc_hashed]--;
            }
        }
    }

    uint32_t prediction = ::pc_prediction[this][pc_hashed];
    if(prediction < 4){
        // cache-averse; set RRIP to 7
        ::priority[this][line_id] = 7;
    } else {
        // set RRIP to 0 initially
        ::priority[this][line_id] = 0;
        if(!hit){
            // age all lines in this set
            uint32_t set_start = set * NUM_WAY; 
            for(size_t i = set_start; i < set_start + NUM_WAY; i++){
                if(::priority[this][i] < 6){
                    ::priority[this][i]++;
                }
            }
        }
    }
    /*
    2 steps:
        1. perform OPTGen update (if set is sampled)
            a. Start at current_time of the set - 1, scan back (wrap around) until reaching current_time, check if at any point the set was full.
               Stop when you find line_id (the current line).
                - If line_id was not found, assume it was a cache miss. 
                - Otherwise, it was a cache_miss if the set was full at some point

                - Else, increment all entries in the usage interval to update the liveness interval
            c. If the result was a miss, take the last_instr, and train it negatively
                - Decrease pc_prediction[hash(last_instr[line_id])]
            d. If the result is a hit, train the last_instr positively
                - Same as (c) except increase pc_prediction

        2. Get hawkeye prediction, update RRIP line priority for the current line using 2 pieces of information:
            a. Prediction for current PC
                - pc_prediction[ip]
            b. Whether this was a hit or miss
                - Passed as an argument
    */


}

void CACHE::replacement_final_stats() {}