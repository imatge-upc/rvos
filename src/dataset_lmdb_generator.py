# This script generates a data structure in the form of key-value storage. This is made in the huge amount of 
# calls to the function os.listdir inside base_youtube.py

import os
import lmdb

from args import get_parser


class LMDBGenerator:
    def __init__(self, ext='.jpg', gen_type='seq'):
        self.ext = ext
        self.gen_type = gen_type

    def generate_lmdb_file(self, root_dir, frames_dir):
        env = lmdb.open(os.path.join(root_dir, 'lmdb_' + self.gen_type))
        root_in_dirs = os.listdir(frames_dir)

        for d in root_in_dirs:
            folder_dir = os.path.join(frames_dir, d)

            _files_basename = sorted([f for f in os.listdir(folder_dir) if f.endswith(self.ext)])
            files_str_vec = '|'.join(_files_basename)

            print( "Generating lmdb for: " + folder_dir)
            with env.begin(write=True) as txn:
                txn.put(d.encode('ascii'), files_str_vec.encode())


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    
    if args.dataset == 'youtube':
        from misc.config_youtubeVOS import cfg

        frame_lmdb_generator_sequences = LMDBGenerator(ext='.jpg', gen_type='seq')
        frame_lmdb_generator_sequences.generate_lmdb_file(cfg.PATH.DATA, cfg.PATH.SEQUENCES_TRAIN)
        frame_lmdb_generator_sequences.generate_lmdb_file(cfg.PATH.DATA, cfg.PATH.SEQUENCES_TEST)
    
        frame_lmdb_generator_annotations = LMDBGenerator(ext='.png', gen_type='annot')
        frame_lmdb_generator_annotations.generate_lmdb_file(cfg.PATH.DATA, cfg.PATH.ANNOTATIONS_TRAIN)
        frame_lmdb_generator_annotations.generate_lmdb_file(cfg.PATH.DATA, cfg.PATH.ANNOTATIONS_TEST)
        
    else:
        from misc.config import cfg

        frame_lmdb_generator_sequences = LMDBGenerator(ext='.jpg', gen_type='seq')
        frame_lmdb_generator_sequences.generate_lmdb_file(cfg.PATH.DATA, cfg.PATH.SEQUENCES)
    
        frame_lmdb_generator_annotations = LMDBGenerator(ext='.png', gen_type='annot')
        frame_lmdb_generator_annotations.generate_lmdb_file(cfg.PATH.DATA, cfg.PATH.ANNOTATIONS)
