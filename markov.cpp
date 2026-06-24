// TODO!!!: comment everything / make it more readable

#include "markov.h"
#include <iostream>
#include <string>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <random>
#include <chrono>
#include <sstream>
#include <fstream>
#include <algorithm>


// Reusable standard modern RNG engine
static std::mt19937& get_rng() {
    static std::mt19937 gen(std::chrono::system_clock::now().time_since_epoch().count());
    return gen;
}


// From 0 to 1
static double get_rand_double() {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(get_rng());
}


// Also from 0 to 1
static int get_rand_int(int min, int max) {
    if (min > max) return min;
    std::uniform_int_distribution<int> dist(min, max);
    return dist(get_rng());
}


// Constructor
Markov::Markov() {
    vocabulary.push_back("[START]");
    vocabulary.push_back("[END]");
    word_to_id["[START]"] = START;
    word_to_id["[END]"] = END;
}


// from word string (vocab.txt) to ID (used in memory / reverse_memory.dat)
int Markov::get_id(std::string word) {
    if (word_to_id.find(word) == word_to_id.end()) {
        int new_id = vocabulary.size();
        vocabulary.push_back(word);
        word_to_id[word] = new_id;
        return new_id;
    }
    return word_to_id[word];
}


std::string Markov::sanitize(std::string raw) {
    std::string clean;
    for (unsigned char c : raw) {
        if (c >= 32) clean += c; // removed, now supports all unicode / emoji
    }
    return clean;
}


int Markov::pick_weighted(std::map<int, int>& options, bool f, int stop_token) { // damping deprecated
    // NOTE: pair.first is word, pair.second is weight

    int total = 0; // accumulate ceiling weight first
    for (auto const& pair : options) { // push max weight up if not start/end
        if (f && (pair.first == END || pair.first == START) && options.size() > 1) continue; // skip end tokens if -f
        total += pair.second;
    }
    if (total <= 0) return stop_token;

    // Roll dice, then walk down the road to see which neighborhood it lands in
    int roll = get_rand_int(0, total - 1);
    for (auto const& pair : options) {
        if (f && (pair.first == END || pair.first == START) && options.size() > 1) continue; // skip if end token
        if (roll < pair.second) return pair.first;
        roll -= pair.second;
    }
    return stop_token;
}


int Markov::pick_random(std::map<int, int>& options, bool f, int stop_token) { // damping deprecated
    // NOTE: pair.first is word, pair.second is weight
    std::vector<int> keys;
    for (auto const& pair : options) {
        if (f && (pair.first == END || pair.first == START) && options.size() > 1) continue; // skip end tokens if -f
        keys.push_back(pair.first);
    }
    if (keys.empty()) return stop_token;

    // Roll dice (nothing else)
    int roll = get_rand_int(0, keys.size() - 1);
    return keys[roll];
}


// Advances `context_window` by one token using either memory or reverse_memory.
// Returns the chosen token id (int).
// Forwards / reverse compatibility decided by which memory you pass in.
int Markov::iterate_chain(std::vector<int>& context_window, std::map<std::vector<int>, std::map<int, int>>& memory,
                        int o, bool w, bool r, bool f, double damping, double context_entropy) {
    // stop token - consts START, END declared at top
    int stop_token;
    if (r) stop_token = START;
    else stop_token = END;

    // if context > 1 & entropy rolls a true, turn into context window = 1
    if (context_window.size() > 1 && get_rand_double() < context_entropy) {
        context_window.erase(context_window.begin());
    }
    // if memory can't find the context, downgrade context size
    while (memory.find(context_window) == memory.end() && !context_window.empty()) {
        context_window.erase(context_window.begin());
    }
    if (context_window.empty()) return stop_token; // fallback

    // IMPORTANT: options is a KV list saying each possible word + frequency of occurence
    std::map<int, int>& options = memory[context_window];
    
    int next_id = stop_token;
    for (int i = 0; i < 10; i++) { // try 10 times max
        if (w) next_id = pick_weighted(options, f, stop_token);
        else next_id = pick_random(options, f, stop_token);
        if (next_id != stop_token) break; // generate, then end if token is valid
        
        if (get_rand_double() < damping) break; // break after probability set by damping
    }

    if (next_id == stop_token) {
        if (f && vocabulary.size() > 2) // f only runs if vocab size doesn't have only end token
            next_id = get_rand_int(2, vocabulary.size() - 1); // get int between first / last word
        else return stop_token; // fallback
    }

    context_window.push_back(next_id); // pointer
    if (context_window.size() > o) context_window.erase(context_window.begin()); // check for o (order) flag
    return next_id;
}


// generate without seed.
// NOTE: context window fully managed by iterate_chain(context, memory, o, w, r, f, d, e).
// 'c' arg is REQUIRED - pass in python with default.
std::string Markov::generate(int o, bool w, int c, bool f, double damping, double entropy) {
    std::vector<int> context_window(o, START);
    std::string result = "";

    for (int i = 0; i < c; i++) {
        int next_id = iterate_chain(context_window, memory, o, w, false, f, damping, entropy); // r is always false (physically impossible)

        if (next_id == END) break; // no multi-token handling, one way
        result += " " + vocabulary[next_id]; // append word of id generated
        // no context management, moved to iterate_chain
    }
    return result;
}


// o: order, c: max words, r: reverse, _i: infix, f: force
// infix is _i because of for loops
std::string Markov::generate_seeded(std::string seed, int o, bool w, int c, bool r, bool _i, bool f, double damping, double entropy) {
    std::string clean_seed = sanitize(seed);
    if (clean_seed.empty()) return "uuhNAHH";
    if (word_to_id.find(clean_seed) == word_to_id.end()) return "uuhNAHH"; // early return in case of no match in data
    int seed_id = word_to_id[clean_seed];

    if (_i) {
        std::string backward_part = "";
        std::string forward_part = "";
        int half_count = c / 2;

        // backwards:
        std::vector<int> back_context_window(o, END);
        back_context_window.push_back(seed_id); // manage context window init
            if (back_context_window.size() > o) back_context_window.erase(back_context_window.begin());
        
        for (int i = 0; i < half_count; i++) {
            int next_id = iterate_chain(back_context_window, reverse_memory, o, w, true, f, damping, entropy);

            if (next_id == START) break; // reverse so START ends the chain
            backward_part = vocabulary[next_id] + " " + backward_part; // append word of id generated, but backwards
        }

        // forwards:
        std::vector<int> fore_context_window(o, START);
        fore_context_window.push_back(seed_id); // manage context window init
            if (fore_context_window.size() > o) fore_context_window.erase(fore_context_window.begin());
        
        for (int i = 0; i < half_count; i++) {
            int next_id = iterate_chain(fore_context_window, memory, o, w, false, f, damping, entropy);

            if (next_id == END) break;
            forward_part += " " + vocabulary[next_id]; // append word of id generated
        }
        return backward_part + " " + clean_seed + " " + forward_part;
    }

    // reverse: (basically copy paste of _i)
    if (r) {
        std::string result = "";

        // backwards:
        std::vector<int> back_context_window(o, END);
        back_context_window.push_back(seed_id); // manage context window init
            if (back_context_window.size() > o) back_context_window.erase(back_context_window.begin());
        
        for (int i = 0; i < c; i++) {
            int next_id = iterate_chain(back_context_window, reverse_memory, o, w, true, f, damping, entropy);

            if (next_id == START) break; // reverse so START ends the chain
            result = vocabulary[next_id] + " " + result; // append word of id generated, but backwards
        }
        return result + " " + clean_seed;
    }

    // forwards: (normal)
    else {
        std::string result = "";

        // forwards:
        std::vector<int> context_window(o, START);
        context_window.push_back(seed_id); // manage context window init
            if (context_window.size() > o) context_window.erase(context_window.begin());
        
        for (int i = 0; i < c; i++) {
            int next_id = iterate_chain(context_window, memory, o, w, false, f, damping, entropy);
            
            if (next_id == END) break;
            result += " " + vocabulary[next_id];
        }
        return clean_seed + " " + result;
    }
    return "uuhNAHH"; // for sanity
}

void Markov::train(std::string raw_message, int max_order) {
    std::string clean = sanitize(raw_message);
    if (clean.empty()) return;
    
    std::stringstream ss(clean);
    std::string word;
    std::vector<int> tokens;
    for (int i = 0; i < max_order; i++) tokens.push_back(START);
    while (ss >> word) tokens.push_back(get_id(word));
    tokens.push_back(END);

    for (size_t i = max_order; i < tokens.size(); i++) {
        int suffix = tokens[i];
        for (int o = 1; o <= max_order; o++) {
            std::vector<int> prefix;
            for (int j = o; j > 0; j--) prefix.push_back(tokens[i - j]);
            memory[prefix][suffix]++;
        }
    }

    std::vector<int> rev_tokens;
    for (int i = 0; i < max_order; i++) rev_tokens.push_back(START);
    std::vector<int> fwd(tokens.begin() + max_order, tokens.end());
    std::reverse(fwd.begin(), fwd.end());
    for (int id : fwd) rev_tokens.push_back(id);

    for (size_t i = max_order; i < rev_tokens.size(); i++) {
        int suffix = rev_tokens[i];
        for (int o = 1; o <= max_order; o++) {
            std::vector<int> prefix;
            for (int j = o; j > 0; j--) prefix.push_back(rev_tokens[i - j]);
            reverse_memory[prefix][suffix]++;
        }
    }
}

void Markov::train_from_file(std::string filename, int o) {
    std::ifstream file(filename);
    if (!file.is_open()) return;
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) train(line, o);
    }
    file.close();
}

void Markov::save_brain(std::string folder) {
    std::ofstream vocab_file(folder + "/vocab.txt");
    if (!vocab_file.is_open()) return; 
    
    for (const auto& v : vocabulary) vocab_file << v << "\n";
    vocab_file.close();

    std::ofstream mem_file(folder + "/memory.dat");
    if (!mem_file.is_open()) return;
    for (auto it = memory.begin(); it != memory.end(); ++it) {
        const std::vector<int>& prefix = it->first;
        const std::map<int, int>& suffixes = it->second;
        mem_file << prefix.size() << " ";
        for (int id : prefix) mem_file << id << " ";
        mem_file << suffixes.size() << " ";
        for (auto const& s_pair : suffixes) mem_file << s_pair.first << " " << s_pair.second << " ";
        mem_file << "\n";
    }
    mem_file.close();

    std::ofstream rmem_file(folder + "/reverse_memory.dat");
    if (!rmem_file.is_open()) return;
    for (auto it = reverse_memory.begin(); it != reverse_memory.end(); ++it) {
        const std::vector<int>& prefix = it->first;
        const std::map<int, int>& suffixes = it->second;
        rmem_file << prefix.size() << " ";
        for (int id : prefix) rmem_file << id << " ";
        rmem_file << suffixes.size() << " ";
        for (auto const& s_pair : suffixes) rmem_file << s_pair.first << " " << s_pair.second << " ";
        rmem_file << "\n";
    }
    rmem_file.close();
}

void Markov::load_brain(std::string folder) {
    std::ifstream vocab_file(folder + "/vocab.txt");
    if (!vocab_file.is_open()) return;
    
    vocabulary.clear();
    word_to_id.clear();
    std::string word;
    while (std::getline(vocab_file, word)) {
        int id = vocabulary.size();
        vocabulary.push_back(word);
        word_to_id[word] = id;
    }
    vocab_file.close();

    memory.clear();
    std::ifstream mem_file(folder + "/memory.dat");
    int prefix_size, suffix_count;
    if (mem_file.is_open()) {
        while (mem_file >> prefix_size) {
            std::vector<int> prefix;
            for (int i = 0; i < prefix_size; i++) {
                int id; mem_file >> id;
                prefix.push_back(id);
            }
            mem_file >> suffix_count;
            for (int i = 0; i < suffix_count; i++) {
                int s_id, count;
                mem_file >> s_id >> count;
                memory[prefix][s_id] = count;
            }
        }
        mem_file.close();
    }

    reverse_memory.clear();
    std::ifstream rmem_file(folder + "/reverse_memory.dat");
    if (rmem_file.is_open()) {
        while (rmem_file >> prefix_size) {
            std::vector<int> prefix;
            for (int i = 0; i < prefix_size; i++) {
                int id; rmem_file >> id;
                prefix.push_back(id);
            }
            rmem_file >> suffix_count;
            for (int i = 0; i < suffix_count; i++) {
                int s_id, count;
                rmem_file >> s_id >> count;
                reverse_memory[prefix][s_id] = count;
            }
        }
        rmem_file.close();
    }
}

void Markov::purge(std::vector<std::string> blocked_words) {
    std::unordered_set<int> blocked_ids;
    for (const auto& word : blocked_words) {
        auto it = word_to_id.find(word);
        if (it != word_to_id.end()) {
            blocked_ids.insert(it->second);
        }
    }
    if (blocked_ids.empty()) return;

    auto is_blocked = [&](int id) {
        return blocked_ids.find(id) != blocked_ids.end();
    };

    for (auto it = memory.begin(); it != memory.end();) {
        bool bad = false;
        for (int id : it->first) {
            if (is_blocked(id)) { bad = true; break; }
        }
        if (bad) { 
            it = memory.erase(it); 
            continue; 
        }
        for (auto sit = it->second.begin(); sit != it->second.end();) {
            if (is_blocked(sit->first)) sit = it->second.erase(sit);
            else ++sit;
        }
        ++it;
    }

    for (auto it = reverse_memory.begin(); it != reverse_memory.end();) {
        bool bad = false;
        for (int id : it->first) {
            if (is_blocked(id)) { bad = true; break; }
        }
        if (bad) { 
            it = reverse_memory.erase(it); 
            continue; 
        }
        for (auto sit = it->second.begin(); sit != it->second.end();) {
            if (is_blocked(sit->first)) sit = it->second.erase(sit);
            else ++sit;
        }
        ++it;
    }

    int uuh_id = get_id("uuh");

    // Cleanly link blocked keywords directly to the unified "uuh" token ID
    for (const auto& word : blocked_words) {
        if (word == "uuh") continue; 
        
        auto it = word_to_id.find(word);
        if (it != word_to_id.end()) {
            int bad_id = it->second;
            word_to_id[word] = uuh_id; 
            vocabulary[bad_id] = "uuh"; 
        }
    }
}