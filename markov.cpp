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

static double get_rand_double() {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(get_rng());
}

static int get_rand_int(int min, int max) {
    if (min > max) return min;
    std::uniform_int_distribution<int> dist(min, max);
    return dist(get_rng());
}

Markov::Markov() {
    vocabulary.push_back("[START]");
    vocabulary.push_back("[END]");
    word_to_id["[START]"] = START;
    word_to_id["[END]"] = END;
}

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
        if (c >= 32 && c <= 126) clean += c;
    }
    return clean;
}

int Markov::pick_weighted(std::map<int, int>& options, bool f, double damping, double context_entropy) {
    int max_weight = 0;
    for (auto const& pair : options) {
        if (pair.first != END && pair.first != START && pair.second > max_weight) {
            max_weight = pair.second;
        }
    }

    int total = 0;
    for (auto const& pair : options) {
        if (f && pair.first == END && options.size() > 1) continue;
        
        // Damping override for eternal yapping
        if (damping == 0.0 && (pair.first == END || pair.first == START)) continue;

        // Context Entropy filter for pruning low frequency options
        if (context_entropy > 0.0 && pair.first != END && pair.first != START) {
            if (pair.second < max_weight * context_entropy) continue;
        }

        if (pair.first == END || pair.first == START) {
            total += (damping == 0.0) ? 0 : std::max(1, static_cast<int>(pair.second * damping));
        } else {
            total += pair.second;
        }
    }
    if (total <= 0) return END;

    int roll = get_rand_int(0, total - 1);

    for (auto const& pair : options) {
        if (f && pair.first == END && options.size() > 1) continue;
        if (damping == 0.0 && (pair.first == END || pair.first == START)) continue;
        
        if (context_entropy > 0.0 && pair.first != END && pair.first != START) {
            if (pair.second < max_weight * context_entropy) continue;
        }

        int current_weight = (pair.first == END || pair.first == START) 
                             ? ((damping == 0.0) ? 0 : std::max(1, static_cast<int>(pair.second * damping))) 
                             : pair.second;

        if (roll < current_weight) return pair.first;
        roll -= current_weight;
    }
    return END;
}

int Markov::pick_random(std::map<int, int>& options, bool f, double damping, double context_entropy) {
    std::vector<int> keys;
    
    int max_weight = 0;
    for (auto const& pair : options) {
        if (pair.first != END && pair.first != START && pair.second > max_weight) {
            max_weight = pair.second;
        }
    }

    for (auto const& pair : options) {
        if (f && pair.first == END && options.size() > 1) continue;
        if (damping == 0.0 && (pair.first == END || pair.first == START)) continue;
        if (context_entropy > 0.0 && pair.first != END && pair.first != START) {
            if (pair.second < max_weight * context_entropy) continue;
        }
        keys.push_back(pair.first);
    }

    // Fallback: entropy wiped the pool, retry keeping damping and f
    if (keys.empty()) {
        for (auto const& pair : options) {
            if (f && pair.first == END && options.size() > 1) continue;  // f fix
            if (damping == 0.0 && (pair.first == END || pair.first == START)) continue;
            keys.push_back(pair.first);
        }
    }

    if (keys.empty()) return END;

    // Pick uniformly, but treat damping as a rejection probability for END/START
    while (true) {
        int candidate = keys[get_rand_int(0, keys.size() - 1)];
        if ((candidate == END || candidate == START) && get_rand_double() > damping) continue;
        return candidate;
    }
}

std::string Markov::generate(int o, bool w, int c, bool r, bool f, double damping, double context_entropy) {
    std::vector<int> current_state(o, START);
    int word_counter = 0;
    std::string result = "";
    
    for (int i = 0; i < c; i++) {
        if (current_state.size() > 1 && get_rand_double() < context_entropy) {
            current_state.erase(current_state.begin());
        }

        while (memory.find(current_state) == memory.end() && !current_state.empty()) {
            current_state.erase(current_state.begin());
        }
        if (current_state.empty()) break;

        std::map<int, int>& options = memory[current_state];
        int next_id = w ? pick_weighted(options, f, damping, context_entropy) : pick_random(options, f, damping, context_entropy);
        
        if (f && (next_id == END || next_id == -1)) {
            if (vocabulary.size() > 2) {
                next_id = get_rand_int(2, vocabulary.size() - 1);
            } else {
                break;
            }
        } else if (next_id == END || next_id == -1) {
            break;
        }

        if (next_id >= 0 && next_id < vocabulary.size()) {
            result += vocabulary[next_id] + " ";
        }
        word_counter++;
        
        current_state.push_back(next_id);
        if (current_state.size() > o) current_state.erase(current_state.begin());
    }
    return result;
}

std::string Markov::generate_seeded(std::string seed, int o, bool w, int c, bool r, bool infix, bool f, double damping, double context_entropy) {
    std::string clean_seed = sanitize(seed);
    
    if (infix) {
        if (word_to_id.find(clean_seed) == word_to_id.end()) return "uuh";
        int seed_id = word_to_id[clean_seed];
        
        std::string backward_part = "";
        std::string forward_part = "";
        int half_count = c / 2;

        std::vector<int> rev_state;
        for (auto const& pair : reverse_memory) {
            if (!pair.first.empty() && pair.first.back() == seed_id) { 
                rev_state = pair.first; 
                break; 
            }
        }
        if (!rev_state.empty()) {
            for (int i = 0; i < half_count; i++) {
                if (rev_state.size() > 1 && get_rand_double() < context_entropy) {
                    rev_state.erase(rev_state.begin());
                }
                while (reverse_memory.find(rev_state) == reverse_memory.end() && !rev_state.empty()) {
                    rev_state.erase(rev_state.begin());
                }
                if (rev_state.empty()) break;

                std::map<int, int>& options = reverse_memory[rev_state];
                int next_id = w ? pick_weighted(options, f, damping, context_entropy) : pick_random(options, f, damping, context_entropy);
                
                if (next_id == START || next_id == -1) break;
                
                if (f && next_id == END) {
                    if (vocabulary.size() > 2) next_id = get_rand_int(2, vocabulary.size() - 1);
                    else break;
                } else if (next_id == END) {
                    break;
                }

                if (next_id >= 0 && next_id < vocabulary.size()) {
                    backward_part = vocabulary[next_id] + " " + backward_part;
                }
                rev_state.push_back(next_id);
                if (rev_state.size() > o) rev_state.erase(rev_state.begin());
            }
        }

        std::vector<int> fwd_state;
        for (auto const& pair : memory) {
            if (!pair.first.empty() && pair.first.back() == seed_id) { 
                fwd_state = pair.first; 
                break; 
            }
        }
        if (!fwd_state.empty()) {
            for (int i = 0; i < half_count; i++) {
                if (fwd_state.size() > 1 && get_rand_double() < context_entropy) {
                    fwd_state.erase(fwd_state.begin());
                }
                while (memory.find(fwd_state) == memory.end() && !fwd_state.empty()) {
                    fwd_state.erase(fwd_state.begin());
                }
                if (fwd_state.empty()) break;

                std::map<int, int>& options = memory[fwd_state];
                int next_id = w ? pick_weighted(options, f, damping, context_entropy) : pick_random(options, f, damping, context_entropy);
                
                if (f && (next_id == END || next_id == -1)) {
                  if (vocabulary.size() > 2) next_id = get_rand_int(2, vocabulary.size() - 1);
                  else break;
                } else if (next_id == END || next_id == -1) {
                    break;
                }

                if (next_id >= 0 && next_id < vocabulary.size()) {
                    forward_part += vocabulary[next_id] + " ";
                }
                fwd_state.push_back(next_id);
                if (fwd_state.size() > o) fwd_state.erase(fwd_state.begin());
            }
        }
        return backward_part + clean_seed + " " + forward_part;
    }

    if (r) {
        int word_counter = 0;
        std::string result = "";
        if (word_to_id.find(clean_seed) == word_to_id.end()) return "uuh";
        int seed_id = word_to_id[clean_seed];

        std::vector<int> rev_state;
        bool found_start = false;

        for (auto const& pair : reverse_memory) {
            const std::vector<int>& state_vec = pair.first;
            if (!state_vec.empty() && state_vec.back() == seed_id) {
                rev_state = state_vec;
                found_start = true;
                break;
            }
        }

        if (!found_start) return seed + " ";

        for (int i = 0; i < c; i++) {
            if (rev_state.size() > 1 && get_rand_double() < context_entropy) {
                rev_state.erase(rev_state.begin());
            }
            while (reverse_memory.find(rev_state) == reverse_memory.end() && !rev_state.empty()) {
                rev_state.erase(rev_state.begin());
            }
            if (rev_state.empty()) break;

            std::map<int, int>& options = reverse_memory[rev_state];
            int next_id = w ? pick_weighted(options, f, damping, context_entropy) : pick_random(options, f, damping, context_entropy);

            if (next_id == START || next_id == -1) break;
            
            if (f && next_id == END) {
                if (vocabulary.size() > 2) next_id = get_rand_int(2, vocabulary.size() - 1);
                else break;
            } else if (next_id == END) {
                break;
            }

            if (next_id >= 0 && next_id < vocabulary.size()) {
                result = vocabulary[next_id] + " " + result;
            }
            word_counter++;

            rev_state.push_back(next_id);
            if (rev_state.size() > o) rev_state.erase(rev_state.begin());
        }
        return (word_counter == 0) ? seed + " " : result + seed + " ";
    }

    std::stringstream ss(clean_seed);
    std::string word;
    std::vector<int> current_state(o, START);
    while (ss >> word) {
        if (word_to_id.find(word) == word_to_id.end()) continue;
        current_state.push_back(word_to_id[word]);
        if (current_state.size() > o) current_state.erase(current_state.begin());
    }

    int word_counter = 0;
    std::string result = "";
    for (int i = 0; i < c; i++) {
        if (current_state.size() > 1 && get_rand_double() < context_entropy) {
            current_state.erase(current_state.begin());
        }
        while (memory.find(current_state) == memory.end() && !current_state.empty()) {
            current_state.erase(current_state.begin());
        }
        if (current_state.empty()) break;

        std::map<int, int>& options = memory[current_state];
        int next_id = w ? pick_weighted(options, f, damping, context_entropy) : pick_random(options, f, damping, context_entropy);
        
        if (f && (next_id == END || next_id == -1)) {
            if (vocabulary.size() > 2) next_id = get_rand_int(2, vocabulary.size() - 1);
            else break;
        } else if (next_id == END || next_id == -1) {
            break;
        }

        if (next_id >= 0 && next_id < vocabulary.size()) {
            result += vocabulary[next_id] + " ";
        }
        word_counter++;
        
        current_state.push_back(next_id);
        if (current_state.size() > o) current_state.erase(current_state.begin());
    }
    return result;
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

    // Get the ID of the fallback string safely
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