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

Markov::Markov() {
  vocabulary.push_back("[START]");
  vocabulary.push_back("[END]");
  word_to_id["[START]"] = START;
  word_to_id["[END]"] = END;
}

int Markov::pick_weighted(std::map<int, int>& options, bool f) {
  int total = 0;
  for (auto const& pair : options) {
    if (f && pair.first == END && options.size() > 1) continue;
    total += pair.second;
  }
  if (total <= 0) return END;

  std::uniform_int_distribution<int> dist(0, total - 1);
  static std::mt19937 gen(std::chrono::system_clock::now().time_since_epoch().count());
  int roll = dist(gen);

  for (auto const& pair : options) {
    if (f && pair.first == END && options.size() > 1) continue;
    if (roll < pair.second) return pair.first;
    roll -= pair.second;
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
    if (c >= 32 && c <= 126) {
      clean += c;
    }
  }
  return clean;
}

std::string Markov::generate(int o, bool w, int c, bool r, bool f) {
  std::vector<int> current_state(o, START);
  int word_counter = 0;
  std::string result = "";

  for (int i = 0; i < c; i++) {
    if (memory.find(current_state) == memory.end()) break;
    std::map<int, int>& options = memory[current_state];
    int next_id = -1;

    if (w) next_id = pick_weighted(options, f);
    else next_id = pick_random(options, f);

    if (f && (next_id == END || next_id == -1)) {
      next_id = 2 + (rand() % (vocabulary.size() - 2));
    }
    else if (next_id == END || next_id == -1) break;

    result += vocabulary[next_id] + " ";
    word_counter++;
    current_state.push_back(next_id);
    if (current_state.size() > o) current_state.erase(current_state.begin());

    while (memory.find(current_state) == memory.end() && !current_state.empty()) {
      current_state.erase(current_state.begin());
    }
    if (current_state.empty()) break;
  }
  return (word_counter == 0) ? "uuh" : result;
}

std::string Markov::generate_seeded(std::string seed, int o, bool w, int c, bool f) {
  std::stringstream ss(sanitize(seed));
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
    if (memory.find(current_state) == memory.end()) break;
    std::map<int, int>& options = memory[current_state];
    int next_id = w ? pick_weighted(options, f) : pick_random(options, f);

    if (f && (next_id == END || next_id == -1)) {
      next_id = 2 + (rand() % (vocabulary.size() - 2));
    }
    else if (next_id == END || next_id == -1) break;

    result += vocabulary[next_id] + " ";
    word_counter++;
    current_state.push_back(next_id);
    if (current_state.size() > o) current_state.erase(current_state.begin());

    while (memory.find(current_state) == memory.end() && !current_state.empty()) {
      current_state.erase(current_state.begin());
    }
    if (current_state.empty()) break;
  }
  return (word_counter == 0) ? "uuh" : result;
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
}

void Markov::load_brain(std::string folder) {
  vocabulary.clear();
  word_to_id.clear();
  std::ifstream vocab_file(folder + "/vocab.txt");
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