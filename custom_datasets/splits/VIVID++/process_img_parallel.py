from process_img import *
import os
import sys
from concurrent.futures import ProcessPoolExecutor
if __name__ == '__main__':
    folder_name = sys.argv[1]
    seq_name = [x for x in os.listdir(folder_name) if x.endswith('.bag')]
    seq_name = [x.replace('.bag', '') for x in seq_name]

    def process_sequence(seq):
        print(f"Processing sequence: {seq}")
        main(folder_name, seq)

    max_workers = min(5, len(seq_name))  # avoid over-allocating
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        executor.map(process_sequence, seq_name)
