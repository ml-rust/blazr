# Kokoro voice assets

Pre-converted style vectors for the 54 upstream Kokoro-82M voices. Each file
is a standalone safetensors file containing a single f32 tensor named
`style`.

## Naming

Voice IDs follow upstream convention:

| Prefix | Meaning               |
| :----- | :-------------------- |
| `af_`  | American female       |
| `am_`  | American male         |
| `bf_`  | British female        |
| `bm_`  | British male          |
| `ef_`  | Spanish female        |
| `em_`  | Spanish male          |
| `ff_`  | French female         |
| `hf_`  | Hindi female          |
| `hm_`  | Hindi male            |
| `if_`  | Italian female        |
| `im_`  | Italian male          |
| `jf_`  | Japanese female       |
| `jm_`  | Japanese male         |
| `pf_`  | Portuguese female     |
| `pm_`  | Portuguese male       |
| `zf_`  | Chinese female        |
| `zm_`  | Chinese male          |
| `kf_`  | Korean female         |

## Provenance

Converted from `hexgrad/Kokoro-82M` on HuggingFace. The upstream files are
pickled PyTorch tensors (`.pt`); blazr does not read pickle at runtime, so
voices are pre-converted to safetensors as a one-time build step and
checked in here.

To regenerate, download the upstream `voices/` directory and re-run the
internal conversion (monorepo `dev/` tooling, not shipped).
