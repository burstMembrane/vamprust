GIT_HASH_SHORT=$(git rev-parse --short HEAD)
BRANCH=$(git rev-parse --abbrev-ref HEAD)
cmd_one="./vamp-plugin-sdk/build/vamp-simple-host nnls-chroma:chordino test.wav"
cmd_two="cargo run --release --example simple_host -- nnls-chroma:chordino test.wav"

hyperfine --warmup 3 "$cmd_one" "$cmd_two" --export-markdown benchmarks/shootout_${GIT_HASH_SHORT}_${BRANCH}.md
