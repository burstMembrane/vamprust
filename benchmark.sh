GIT_HASH_SHORT=$(git rev-parse --short HEAD)
BRANCH=$(git rev-parse --abbrev-ref HEAD)
simple_host_cmd="simple_host"
cmd_one="./vamp-plugin-sdk/build/vamp-simple-host nnls-chroma:chordino test.wav"
cmd_two="simple_host nnls-chroma:chordino test.wav"
mkdir -p benchmarks
hyperfine --warmup 3 "$cmd_one" "$cmd_two" --export-markdown benchmarks/shootout_${GIT_HASH_SHORT}_${BRANCH}.md

cmd_three="vamp-simple-host vamp-example-plugins:percussiononsets test.wav"
cmd_four="simple_host vamp-example-plugins:percussiononsets test.wav"

hyperfine --warmup 3 "$cmd_three" "$cmd_four" --export-markdown benchmarks/shootout_${GIT_HASH_SHORT}_${BRANCH}_percussiononsets.md