import os
import struct

def migrate_to_binary_pure_python():
    vocab_path = "./brain/vocab.txt"
    mem_path = "./brain/memory.dat"
    rmem_path = "./brain/reverse_memory.dat"
    output_bin_path = "./brain/cohesive_brain.dat"

    if not all(os.path.exists(p) for p in [vocab_path, mem_path, rmem_path]):
        print("Error: Missing legacy text brain files inside './brain/' directory.")
        return

    print("Opening output binary file stream...")
    with open(output_bin_path, "wb") as out:
        
        # ==========================================
        # 1. PROCESS VOCABULARY
        # ==========================================
        print("Processing vocab.txt...")
        with open(vocab_path, "r", encoding="utf-8", errors="ignore") as vf:
            # Read non-empty words
            vocab_words = [line.strip("\n") for line in vf]
        
        # Pack vocab count (4-byte unsigned int)
        out.write(struct.pack("<I", len(vocab_words)))
        
        for word in vocab_words:
            word_bytes = word.encode("utf-8", errors="ignore")
            # Pack string length (4-byte unsigned int) followed by string payload bytes
            out.write(struct.pack("<I", len(word_bytes)))
            out.write(word_bytes)

        # Helper function to stream-convert the space-separated matrix files
        def convert_text_matrix_to_binary(text_file_path):
            with open(text_file_path, "r", encoding="utf-8", errors="ignore") as f:
                # Count total entries/lines first to write the block prefix count header
                line_count = 0
                for line in f:
                    if line.strip(): line_count += 1
                
                # Write prefix map size (4-byte unsigned int)
                out.write(struct.pack("<I", line_count))
                
                # Rewind and parse tokens
                f.seek(0)
                for line in f:
                    parts = line.split()
                    if not parts:
                        continue
                        
                    # Format: [prefix_size] [prefix_ids...] [suffix_count] [suffix_id count suffix_id count...]
                    prefix_size = int(parts[0])
                    idx = 1
                    
                    prefix_ids = [int(x) for x in parts[idx : idx + prefix_size]]
                    idx += prefix_size
                    
                    suffix_count = int(parts[idx])
                    idx += 1
                    
                    # Pack prefix headers
                    out.write(struct.pack("<I", prefix_size))
                    for pid in prefix_ids:
                        out.write(struct.pack("<i", pid)) # signed 4-byte int
                        
                    out.write(struct.pack("<I", suffix_count))
                    
                    # Pack dynamic child suffix frequencies
                    for _ in range(suffix_count):
                        sid = int(parts[idx])
                        count = int(parts[idx+1])
                        out.write(struct.pack("<i", sid))    # signed 4-byte int
                        out.write(struct.pack("<i", count))  # signed 4-byte int
                        idx += 2

        # ==========================================
        # 2. PROCESS FORWARD MEMORY MATRIX
        # ==========================================
        print("Converting forward memory.dat to binary layout...")
        convert_text_matrix_to_binary(mem_path)

        # ==========================================
        # 3. PROCESS REVERSE MEMORY MATRIX
        # ==========================================
        print("Converting reverse_memory.dat to binary layout...")
        convert_text_matrix_to_binary(rmem_path)

    print(f"\nMigration complete! Cohesive binary brain saved at: {output_bin_path}")
    print(f"File size: {os.path.getsize(output_bin_path) / 1024**2:.2f} MB")

if __name__ == "__main__":
    migrate_to_binary_pure_python()