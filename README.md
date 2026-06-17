# G4LLM — Git for Large Language Models

G4LLM brings **Git-style version control** to LLM knowledge editing.
Track, apply, revert, branch, tag and merge targeted weight edits — just like
commits in a source-code repository.

---

## Quick Start

```bash
python test/demo.py
```

The demo loads **Qwen/Qwen3-0.6B**, performs a complete knowledge edit workflow,
and verifies every git operation including the new Task Arithmetic merge.

---

## Features

| Git Operation | G4LLM Equivalent | Description |
|---|---|---|
| `git init` | `G4LLMRepo.init()` | Create a new repository |
| `git add` | `repo.add(EditRequest(...))` | Stage a knowledge edit |
| `git status` | `repo.status()` | Show staged edits |
| `git commit` | `repo.commit(model, tokenizer)` | Apply ROME/MEMIT edit, save delta |
| `git log` | `repo.log()` | Show commit history |
| `git show` | `repo.diff(hash)` | Inspect weight delta of a commit |
| `git revert` | `repo.revert(hash, model)` | Exactly undo an edit |
| `git checkout` | `repo.checkout(ref, model, ...)` | Restore model to any commit |
| `git branch` | `repo.branch(name)` | Create a branch |
| `git switch` | `repo.switch(name)` | Change current branch |
| `git tag` | `repo.tag(name, hash)` | Tag a commit |
| `git merge` | `repo.merge([ft_models], ...)` | Task Arithmetic model merge |

---

## Example Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from G4LLM.repo import G4LLMRepo
from G4LLM.objects.edit_request import EditRequest
from G4LLM.algo.rome import ROMEConfig

# Load model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype="float32")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

# Init repo
repo = G4LLMRepo.init(".", model_name="Qwen/Qwen3-0.6B", author="researcher")

# Stage an edit
repo.add(EditRequest(
    subject    = "India",
    relation   = "is located in",
    prompt     = "India is located in",
    target_new = "Antarctica",
    target_old = "Asia",
))

# Commit with sequential editing and 5 epochs (MM4KE defaults)
cfg = ROMEConfig(epochs=5, v_num_grad_steps=25)
commits = repo.commit(model, tokenizer, sequential_edit=True)

# Revert
repo.revert(commits[0].hash, model)
```

---

## References

- [ROME — Rank-One Model Editing](https://github.com/kmeng01/rome) (Meng et al., NeurIPS 2022)
- [MEMIT — Mass-Editing Memory in a Transformer](https://github.com/kmeng01/memit) (Meng et al., ICLR 2023)
- [Task Arithmetic — Editing Models with Task Arithmetic](https://arxiv.org/abs/2212.04089) (Ilharco et al., ICLR 2023)
- [Rebuilding ROME](https://github.com/scalable-model-editing/rebuilding-rome) (2024)
- [WISE — Rethinking the Knowledge Memory for Lifelong Model Editing](https://github.com/zjunlp/EasyEdit) (NeurIPS 2024)
- [MM4KE — Model Merging for Knowledge Editing](https://github.com/Applied-Machine-Learning-Lab/MM4KE) (2025)
- [Tracing and Reversing Rank-One Model Edits](https://arxiv.org/abs/2505.20819) (2025)
- [ROME multi-hop limitations / Redundant Editing](https://arxiv.org/abs/2601.04600) (2026)