# Developer 2 Setup: Local LLM (Ollama) & Agent Layer

This branch (`dev2-llm-guard`) contains the cognition side of ModelRisk-Guard: the local LLM client, the prompts, the AST safety guard, and their tests. Follow these steps to get it running on your machine.

## What's on this branch

| File | Purpose |
|---|---|
| `agent/llm_client.py` | Sends prompts to the local Ollama model and extracts the `def policy(robot):` code it writes |
| `agent/prompts.py` | System prompt, initial task prompt, and the hot-patch repair prompt |
| `agent/ast_guard.py` | Blocks imports and any robot call outside the whitelist before code runs |
| `tests/test_agent.py` | Standalone test suite for the above |
| `config.py` | Shared settings (model name, Ollama endpoint, thresholds) |

## 1. Pull the branch

```bash
git fetch origin
git checkout dev2-llm-guard
```

To bring it into your own branch instead:

```bash
git fetch origin
git merge origin/dev2-llm-guard
```

## 2. Install Ollama

| OS | Command |
|---|---|
| Windows | `winget install --id Ollama.Ollama -e --source winget` |
| macOS / Linux | `curl -fsSL https://ollama.com/install.sh \| sh` |

On Windows, add `--source winget`. Without it, winget may fail with a certificate error from the Microsoft Store source.

## 3. Pull the model

```bash
ollama pull qwen2.5-coder:7b
```

This downloads about 4.7 GB. Check it is installed:

```bash
ollama list
```

You should see `qwen2.5-coder:7b` in the list.

## 4. Confirm Ollama is running

```bash
curl http://localhost:11434
```

Expected reply: `Ollama is running`. If not, start it with `ollama serve`.

## 5. Warm up the model, then run the tests

```bash
ollama run qwen2.5-coder:7b "hi"
python -m tests.test_agent
```

Expected output:

```
[PASSED] test_ast_blocks_malicious_code
[PASSED] test_ast_approves_valid_cap
[PASSED] test_llm_hot_patch_repair_cycle

[SUCCESS] ALL DEVELOPER 2 TESTS PASSED CLEANLY.
```

**Always warm up first.** Loading the model from cold took about 94 s on our test laptop, but `llm_client.py` gives up after 15 s. When that happens, the code silently switches to a hard-coded offline fallback, so the tests pass even though the real model never answered. If you see `[WARN] Local Ollama request failed`, the model was not used. Ollama unloads the model after 5 minutes idle, so warm it up again before recording the demo.

## Known issues

- **The model skips `grip_plug`.** With the current prompts, the real model writes a policy without `robot.execute_primitive("grip_plug")`. In the sim, the plug only moves if it was gripped, so the pull does nothing. Adding a few-shot example to `agent/prompts.py` should fix this.
- **The initial pull distance is wrong.** The real model asks for `distance=10.0` rather than the `0.12` the demo script expects, because the prompt gives no sense of scale. The same few-shot example should fix it.
- **The hot-patch works.** After the repair prompt, the model contracts the stroke to `distance=0.035`, which meets the ≤ 0.04 m exit criterion.
