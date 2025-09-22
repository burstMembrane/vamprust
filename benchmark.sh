GIT_HASH_SHORT=$(git rev-parse --short HEAD)
BRANCH=$(git rev-parse --abbrev-ref HEAD)
simple_host_cmd="vamp-simple-host-rs"
cmd_one="./vamp-plugin-sdk/build/vamp-simple-host nnls-chroma:chordino test.wav"
cmd_two="vamp-simple-host-rs nnls-chroma:chordino test.wav"
cmd_three="python3 python/examples/chordino.py test.wav"
mkdir -p benchmarks
hyperfine --warmup 5 -r 10 "$cmd_one" "$cmd_two" "$cmd_three" --export-markdown benchmarks/shootout_${GIT_HASH_SHORT}_${BRANCH}.md

# cmd_three="vamp-simple-host vamp-example-plugins:percussiononsets test.wav"
# cmd_four="vamp-simple-host-rs vamp-example-plugins:percussiononsets test.wav"

# hyperfine --warmup 3 "$cmd_three" "$cmd_four" --export-markdown benchmarks/shootout_${GIT_HASH_SHORT}_${BRANCH}_percussiononsets.md
