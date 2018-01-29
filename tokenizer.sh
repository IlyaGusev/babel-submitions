set -x
set -e

OUTPUT_DIR="./models"
mkdir -p $OUTPUT_DIR

# Clone Moses
if [ ! -d "${OUTPUT_DIR}/mosesdecoder" ]; then
  echo "Cloning moses for data processing"
  git clone https://github.com/moses-smt/mosesdecoder.git "${OUTPUT_DIR}/mosesdecoder"
fi

langs=(en ru)

# Tokenize data
for l in ${langs[@]}; do
  f="$OUTPUT_DIR/corpus.$l"
  echo "Tokenizing $f..."
  ${OUTPUT_DIR}/mosesdecoder/scripts/tokenizer/tokenizer.perl -q -l $l -threads 8 < $f > ${f%.*}.tok.$l
done

function join_by { local IFS="$1"; shift; echo "$*"; }

for f in ${OUTPUT_DIR}/*.en; do
  fbase=${f%.*}
  echo "Cleaning ${fbase}..."
  ${OUTPUT_DIR}/mosesdecoder/scripts/training/clean-corpus-n.perl $fbase de en "${fbase}.clean" 1 80
done

# Clean all corpora
for f in ${OUTPUT_DIR}/*.tok.${langs}; do
  fbase=${f%.*}
  echo "Cleaning ${fbase}..."
  l=$(join_by " " ${langs[@]})
  ./clean-corpus-n-monolingual.perl $fbase $l "${fbase}.clean" 1 80
done

echo "All done."
