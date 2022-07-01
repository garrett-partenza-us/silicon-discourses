import argparse
import sentencepiece as spm

parser = argparse.ArgumentParser(description="flags for training tokenizer")
parser.add_argument("--path", required=True, type=str, help="path to .txt file")
parser.add_argument("--prefix", required=False, default="tokenizer", type=str, help="output file prefix")
parser.add_argument("--size", required=False, default=1000, type=int, help="vocabulary size")


def train_tokenizer(path, prefix, size):
    spm.SentencePieceTrainer.train(
        '--input={} --model_prefix={} --vocab_size={}'.format(
            path, prefix, size
        )
    )
    
if __name__ == '__main__':
    
    args = parser.parse_args()
    train_tokenizer(args.path, args.prefix, args.size)