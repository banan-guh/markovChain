#include "markov.h"
#include <iostream>
#include <string>
#include <map>
#include <unordered_map>
#include <vector>
#include <random>
#include <chrono>
#include <sstream>
#include <fstream>
#include <algorithm>

Markov::Markov() {
  vocabulary.push_back("[START]");
  vocabulary.push_back("[END]");
  word_to_id["[START]"] = START;
  word_to_id["[END]"] = END;
}

// FIXED: Added double damping to the implementation signature and internal loop math
int Markov::pick_weighted(std::map<int, int>& options, bool f, double damping) {
  int total = 0;
  for (auto const& pair : options) {
    if (f && pair.first == END && options.size() > 1) continue;
    
    // Apply damping to both structural termination flags
    if (pair.first == END || pair.first == START) {
      total += std::max(1, static_cast<int>(pair.second * damping));
    } else {
      total += pair.second;
    }
  }
  if (total <= 0) return END;

  std::uniform_int_distribution<int> dist(0, total - 1);
  static std::mt19937 gen(std::chrono::system_clock::now().time_since_epoch().count());
  int roll = dist(gen);

  for (auto const& pair : options) {
    if (f && pair.first == END && options.size() > 1) continue;
    
    int current_weight = pair.second;
    if (pair.first == END || pair.first == START) {
      current_weight = std::max(1, static_cast<int>(pair.second * damping));
    }

    if (roll < current_weight) return pair.first;
    roll -= current_weight;
  }
  return END;
}

int Markov::pick_random(std::map<int, int>& options, bool f) {
  std::vector<int> keys;
  for (auto const& pair : options) {
    if (f && pair.first == END && options.size() > 1) continue;
    keys.push_back(pair.first);
  }
  if (keys.empty()) return END;
  std::uniform_int_distribution<int> dist(0, keys.size() - 1);
  static std::mt19937 gen(std::chrono::system_clock::now().time_since_epoch().count());
  return keys[dist(gen)];
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

std::string Markov::generate(int o, bool w, int c, bool r, bool f, double damping, double context_entropy) {
  std::vector<int> current_state(o, START);
  int word_counter = 0;
  std::string result = "";
  for (int i = 0; i < c; i++) {
    
    // --- DYNAMIC CONTEXT MIXING ---
    if (current_state.size() > 1 && ((double)rand() / RAND_MAX) < context_entropy) {
      current_state.erase(current_state.begin());
    }

    while (memory.find(current_state) == memory.end() && !current_state.empty()) {
      current_state.erase(current_state.begin());
    }
    if (current_state.empty()) break;

    std::map<int, int>& options = memory[current_state];
    int next_id = w ? pick_weighted(options, f, damping) : pick_random(options, f);
    
    if (f && (next_id == END || next_id == -1)) {
      next_id = 2 + (rand() % (vocabulary.size() - 2));
    }
    else if (next_id == END || next_id == -1) break;

    result += vocabulary[next_id] + " ";
    word_counter++;
    
    current_state.push_back(next_id);
    if (current_state.size() > o) current_state.erase(current_state.begin());
  }
  return result;
}

std::string Markov::generate_seeded(std::string seed, int o, bool w, int c, bool r, bool infix, bool f, double damping, double context_entropy) {
  std::string clean_seed = sanitize(seed);
  
  // ==========================================
  // BRANCH 1: INFIX GENERATION (-i flag)
  // ==========================================
  if (infix) {
    if (word_to_id.find(clean_seed) == word_to_id.end()) return "uuh";
    int seed_id = word_to_id[clean_seed];
    
    std::string backward_part = "";
    std::string forward_part = "";
    int half_count = c / 2;

    // 1. Backward Path (Left Side)
    std::vector<int> rev_state;
    for (auto const& pair : reverse_memory) {
      if (!pair.first.empty() && pair.first.back() == seed_id) { 
        rev_state = pair.first; 
        break; 
      }
    }
    if (!rev_state.empty()) {
      for (int i = 0; i < half_count; i++) {
        // Context Entropy Check
        if (rev_state.size() > 1 && ((double)rand() / RAND_MAX) < context_entropy) {
          rev_state.erase(rev_state.begin());
        }
        while (reverse_memory.find(rev_state) == reverse_memory.end() && !rev_state.empty()) {
          rev_state.erase(rev_state.begin());
        }
        if (rev_state.empty()) break;

        std::map<int, int>& options = reverse_memory[rev_state];
        int next_id = w ? pick_weighted(options, f, damping) : pick_random(options, f);
        
        if (next_id == START || next_id == -1) break;
        if (f && next_id == END) next_id = 2 + (rand() % (vocabulary.size() - 2));
        else if (next_id == END) break;

        backward_part = vocabulary[next_id] + " " + backward_part;
        rev_state.push_back(next_id);
        if (rev_state.size() > o) rev_state.erase(rev_state.begin());
      }
    }

    // 2. Forward Path (Right Side)
    std::vector<int> fwd_state;
    for (auto const& pair : memory) {
      if (!pair.first.empty() && pair.first.back() == seed_id) { 
        fwd_state = pair.first; 
        break; 
      }
    }
    if (!fwd_state.empty()) {
      for (int i = 0; i < half_count; i++) {
        // Context Entropy Check
        if (fwd_state.size() > 1 && ((double)rand() / RAND_MAX) < context_entropy) {
          fwd_state.erase(fwd_state.begin());
        }
        while (memory.find(fwd_state) == memory.end() && !fwd_state.empty()) {
          fwd_state.erase(fwd_state.begin());
        }
        if (fwd_state.empty()) break;

        std::map<int, int>& options = memory[fwd_state];
        int next_id = w ? pick_weighted(options, f, damping) : pick_random(options, f);
        
        if (next_id == END || next_id == -1) break;
        if (f && next_id == END) next_id = 2 + (rand() % (vocabulary.size() - 2));

        forward_part += vocabulary[next_id] + " ";
        fwd_state.push_back(next_id);
        if (fwd_state.size() > o) fwd_state.erase(fwd_state.begin());
      }
    }
    return backward_part + clean_seed + " " + forward_part;
  }

  // ==========================================
  // BRANCH 2: STANDARD REVERSE GENERATION (-r flag)
  // ==========================================
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
      // Context Entropy Check
      if (rev_state.size() > 1 && ((double)rand() / RAND_MAX) < context_entropy) {
        rev_state.erase(rev_state.begin());
      }
      while (reverse_memory.find(rev_state) == reverse_memory.end() && !rev_state.empty()) {
        rev_state.erase(rev_state.begin());
      }
      if (rev_state.empty()) break;

      std::map<int, int>& options = reverse_memory[rev_state];
      int next_id = w ? pick_weighted(options, f, damping) : pick_random(options, f);

      if (next_id == START || next_id == -1) break;
      if (f && next_id == END) {
        next_id = 2 + (rand() % (vocabulary.size() - 2));
      }
      else if (next_id == END) break;

      result = vocabulary[next_id] + " " + result;
      word_counter++;

      rev_state.push_back(next_id);
      if (rev_state.size() > o) rev_state.erase(rev_state.begin());
    }
    return (word_counter == 0) ? seed + " " : result + seed + " ";
  }

  // ==========================================
  // BRANCH 3: STANDARD SEEDED FORWARD GENERATION
  // ==========================================
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
    // Context Entropy Check
    if (current_state.size() > 1 && ((double)rand() / RAND_MAX) < context_entropy) {
      current_state.erase(current_state.begin());
    }
    while (memory.find(current_state) == memory.end() && !current_state.empty()) {
      current_state.erase(current_state.begin());
    }
    if (current_state.empty()) break;

    std::map<int, int>& options = memory[current_state];
    int next_id = w ? pick_weighted(options, f, damping) : pick_random(options, f);
    if (f && (next_id == END || next_id == -1)) {
      next_id = 2 + (rand() % (vocabulary.size() - 2));
    }
    else if (next_id == END || next_id == -1) break;

    result += vocabulary[next_id] + " ";
    word_counter++;
    
    current_state.push_back(next_id);
    if (current_state.size() > o) current_state.erase(current_state.begin());
  }
  return result;
}

void Markov::train(std::string raw_message, int max_order) {
  std::string clean = sanitize(raw_message);
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
  for (const auto& v : vocabulary) vocab_file << v << "\n";
  vocab_file.close();

  std::ofstream mem_file(folder + "/memory.dat");
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

// Helper function to check if a file exists
bool file_exists(const std::string& name) {
    struct stat buffer;   
    return (stat(name.c_str(), &buffer) == 0); 
}

bool Markov::load_brain(const std::string& folder_path) {
    std::string v_path = folder_path + "/vocab.txt";
    std::string b_path = folder_path + "/brain.dat";
    std::string m_path = folder_path + "/memory.dat";
    std::string rm_path = folder_path + "/reverse_memory.dat";

    // Strict Guard: No vocab.txt = No run.
    if (!file_exists(v_path)) return false;

    // Case 1: brain.dat exists (Fast Path)
    if (file_exists(b_path)) {
        std::ifstream in(b_path, std::ios::binary);
        if (!in) return false;

        unsigned int vocab_size;
        in.read(reinterpret_cast<char*>(&vocab_size), sizeof(vocab_size));
        vocab.clear(); word_to_id.clear();
        for (unsigned int i = 0; i < vocab_size; ++i) {
            unsigned int len; in.read(reinterpret_cast<char*>(&len), sizeof(len));
            std::string w(len, '\0'); in.read(&w[0], len);
            vocab.push_back(w); word_to_id[w] = i;
        }

        auto load_bin = [&](auto& matrix) {
            unsigned int m_size; in.read(reinterpret_cast<char*>(&m_size), sizeof(m_size));
            matrix.clear();
            for (unsigned int i = 0; i < m_size; ++i) {
                unsigned int p_size; in.read(reinterpret_cast<char*>(&p_size), sizeof(p_size));
                std::vector<int> pref(p_size);
                for (unsigned int j = 0; j < p_size; ++j) in.read(reinterpret_cast<char*>(&pref[j]), sizeof(int));
                unsigned int s_cnt; in.read(reinterpret_cast<char*>(&s_cnt), sizeof(s_cnt));
                for (unsigned int j = 0; j < s_cnt; ++j) {
                    int sid, cnt; in.read(reinterpret_cast<char*>(&sid), sizeof(sid));
                    in.read(reinterpret_cast<char*>(&cnt), sizeof(cnt));
                    matrix[pref][sid] = cnt;
                }
            }
        };
        load_bin(memory); load_bin(reverse_memory);
        return true;
    }

    // Case 2: Fallback to memory.dat and reverse_memory.dat
    if (file_exists(m_path) && file_exists(rm_path)) {
        std::ifstream vf(v_path); std::string line;
        vocab.clear(); word_to_id.clear();
        while (std::getline(vf, line)) {
            if (!line.empty()) { vocab.push_back(line); word_to_id[line] = vocab.size() - 1; }
        }

        auto load_txt = [&](const std::string& path, auto& matrix) {
            std::ifstream f(path); int p_size, s_count, pid, sid, count;
            while (f >> p_size) {
                std::vector<int> pref;
                for (int i = 0; i < p_size; ++i) { f >> pid; pref.push_back(pid); }
                f >> s_count;
                for (int i = 0; i < s_count; ++i) { f >> sid >> count; matrix[pref][sid] = count; }
            }
        };
        load_txt(m_path, memory); load_txt(rm_path, reverse_memory);
        this->save(folder_path); // Generates your brain.dat
        return true;
    }
    return false;
}

void Markov::purge(std::vector<std::string> blocked_words) {
  std::vector<int> blocked_ids;
  for (const auto& word : blocked_words) {
    if (word_to_id.find(word) != word_to_id.end()) {
      blocked_ids.push_back(word_to_id[word]);
    }
  }
  if (blocked_ids.empty()) return;

  auto is_blocked = [&](int id) {
    return std::find(blocked_ids.begin(), blocked_ids.end(), id) != blocked_ids.end();
    };

  for (auto it = memory.begin(); it != memory.end();) {
    bool bad = false;
    for (int id : it->first) if (is_blocked(id)) { bad = true; break; }
    if (bad) { it = memory.erase(it); continue; }
    for (auto sit = it->second.begin(); sit != it->second.end();) {
      if (is_blocked(sit->first)) sit = it->second.erase(sit);
      else ++sit;
    }
    ++it;
  }

  for (auto it = reverse_memory.begin(); it != reverse_memory.end();) {
    bool bad = false;
    for (int id : it->first) if (is_blocked(id)) { bad = true; break; }
    if (bad) { it = reverse_memory.erase(it); continue; }
    for (auto sit = it->second.begin(); sit != it->second.end();) {
      if (is_blocked(sit->first)) sit = it->second.erase(sit);
      else ++sit;
    }
    ++it;
  }

  for (const auto& word : blocked_words) {
    if (word_to_id.find(word) != word_to_id.end()) {
      int id = word_to_id[word];
      vocabulary[id] = "uuh";
      word_to_id.erase(word);
      word_to_id["uuh"] = id;
    }
  }
}