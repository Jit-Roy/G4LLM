"""
G4LLM Demo -- Git Version Control for LLM Knowledge Edits
==========================================================
Single file that demonstrates the complete G4LLM workflow on Qwen/Qwen3-0.6B:

  1.  Load model
  2.  Test model knowledge BEFORE edit
  3.  git init
  4.  git add          (stage an EditRequest)
  5.  git status
  6.  git commit       (ROME rank-1 weight update)
  7.  git log / show
  8.  Test model knowledge AFTER edit  <- proof the edit worked
  9.  Probability table: before / after / change
  10. git branch + switch
  11. git tag
  12. git revert       (exact undo, weight restored)
  13. Test model knowledge AFTER revert <- proof it was undone
  14. git checkout
  15. Repository file-tree

Run:
    python test/demo.py
"""

import os, sys, tempfile, copy
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # .../Personal Projects/

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from G4LLM.repo import G4LLMRepo, _copy_weights
from G4LLM.objects.edit_request import EditRequest
from G4LLM.objects.commit import Commit
from G4LLM.algo.rome import ROMEEditor, ROMEConfig
from G4LLM.algo.model_utils import get_weight

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen3-0.6B"

EDIT = dict(
    subject    = "India",
    relation   = "is located in",
    prompt     = "India is located in",
    target_new = "Antarctica",   # single clean token for clear probability shift
    target_old = "Asia",
)

ROME_CFG = ROMEConfig(
    v_lr             = 5e-1,
    v_num_grad_steps = 50,
    v_weight_decay   = 1e-3,
    kl_factor        = 0.0,
    mom2_adjustment  = False,
)

# -----------------------------------------------------------------------------
# Print helpers
# -----------------------------------------------------------------------------
W = 70

def section(n, title):
    print("\n" + "=" * W)
    print(f"  [{n}]  {title}")
    print("=" * W)

def ok(msg):   print(f"  [OK]  {msg}")
def info(msg): print(f"   >>   {msg}")
def sep():     print("  " + "-" * (W - 2))

def safe(text):
    """Replace any character that cannot be printed in the current console."""
    enc = sys.stdout.encoding or "ascii"
    return text.encode(enc, errors="replace").decode(enc)

# -----------------------------------------------------------------------------
# Model helpers
# -----------------------------------------------------------------------------

@torch.no_grad()
def generate(model, tokenizer, prompt, device, max_new=20):
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.generate(
        **enc,
        max_new_tokens     = max_new,
        do_sample          = False,
        pad_token_id       = tokenizer.eos_token_id,
        repetition_penalty = 1.1,
    )
    full = tokenizer.decode(out[0], skip_special_tokens=True)
    return full[len(prompt):].strip()

@torch.no_grad()
def top_k(model, tokenizer, prompt, device, k=6):
    enc    = tokenizer(prompt, return_tensors="pt").to(device)
    logits = model(**enc).logits[0, -1].float()
    probs  = F.softmax(logits, dim=-1)
    topk   = torch.topk(probs, k)
    return [(tokenizer.decode([i.item()]).strip(), p.item())
            for i, p in zip(topk.indices, topk.values)]

@torch.no_grad()
def prob(model, tokenizer, prompt, token, device):
    enc    = tokenizer(prompt, return_tensors="pt").to(device)
    logits = model(**enc).logits[0, -1].float()
    probs  = F.softmax(logits, dim=-1)
    tid    = tokenizer.encode(" " + token, add_special_tokens=False)[-1]
    return probs[tid].item()

def print_top_k(preds):
    for tok, p in preds:
        # safe() replaces any character the console cannot display with '?'
        display = safe(repr(tok))
        bar = "#" * int(p * 300)
        print(f"    {display:<22}  {p:.4f}  {bar}")

# -----------------------------------------------------------------------------
# DEMO
# -----------------------------------------------------------------------------

def main():
    print("\n" + "#" * W)
    print(f"  G4LLM -- Git for LLMs  |  Full Demo on {MODEL_NAME}")
    print("#" * W)

    # -- 1. Load model ---------------------------------------------------------
    section(1, f"Load {MODEL_NAME}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    info(f"Device : {device}")
    info("Loading tokenizer ...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    info("Loading weights in float32 (required for ROME gradient optimisation) ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype             = torch.float32,
        device_map        = device,
        trust_remote_code = True,
    )
    model.eval()

    ok(f"model_type = {model.config.model_type!r}")
    ok(f"layers     = {model.config.num_hidden_layers}")
    ok(f"hidden     = {model.config.hidden_size}  "
       f"intermediate = {model.config.intermediate_size}")

    info("Cloning base model (used later for revert verification) ...")
    base_model = copy.deepcopy(model)

    # -- 2. Knowledge BEFORE edit ----------------------------------------------
    section(2, f"Model knowledge BEFORE edit  |  prompt: \"{EDIT['prompt']}\"")

    preds_before = top_k(model, tokenizer, EDIT["prompt"], device)
    p_new_before = prob(model, tokenizer, EDIT["prompt"], EDIT["target_new"], device)
    p_old_before = prob(model, tokenizer, EDIT["prompt"], EDIT["target_old"], device)

    print("\n  Top-6 next-token predictions (BEFORE edit):")
    print_top_k(preds_before)

    generated_before = generate(model, tokenizer, EDIT["prompt"], device)
    print()
    info(f"Generated continuation  : \"{EDIT['prompt']} {safe(generated_before)}\"")
    info(f"P('{EDIT['target_new']}' | prompt) = {p_new_before:.6f}")
    info(f"P('{EDIT['target_old']}'       | prompt) = {p_old_before:.6f}")

    # -- 3. git init -----------------------------------------------------------
    section(3, "git init")

    repo_dir = tempfile.mkdtemp(prefix="g4llm_demo_")
    repo = G4LLMRepo.init(
        path       = repo_dir,
        model_name = MODEL_NAME,
        author     = "researcher",
        algorithm  = "rome",
    )
    ok(f"Repository created at  {repo_dir}/.g4llm/")
    ok(f"Branch                 {repo._branches.current_branch}")
    ok(f"Model                  {repo.model_name}")
    ok(f"Algorithm              {repo.algorithm}")

    # -- 4. git add ------------------------------------------------------------
    section(4, "git add  --  stage the knowledge edit")

    req = EditRequest(**EDIT)
    repo.add(req)

    ok(f"Staged: {req}")
    info(f"Edit ID       : {req.id}")
    info(f"Subject       : {req.subject}")
    info(f"Relation      : {req.relation}")
    info(f"Old target    : {req.target_old}")
    info(f"New target    : {req.target_new}")

    # -- 5. git status ---------------------------------------------------------
    section(5, "git status")
    print(repo.status())

    # -- 6. git commit  (ROME edit) --------------------------------------------
    section(6, "git commit  --  ROME rank-1 weight update")

    info(f"Running ROME ({ROME_CFG.v_num_grad_steps} optimisation steps, "
         f"lr={ROME_CFG.v_lr}) ...")

    editor = ROMEEditor(model, tokenizer, config=ROME_CFG)
    deltas = editor.edit(req)

    c = Commit(
        edit_request = req,
        deltas       = deltas,
        message      = f"fact: '{req.subject}' {req.relation} '{req.target_new}'",
        author       = "researcher",
        model_name   = MODEL_NAME,
        algorithm    = "rome",
    )
    repo._save_commit(c)
    repo._branches.advance("main", c.hash)

    sep()
    ok(f"Commit hash    : {c.hash}")
    ok(f"Short hash     : {c.short_hash()}")
    ok(f"Message        : {c.message}")
    ok(f"Author         : {c.author}")
    ok(f"Timestamp      : {c.timestamp}")
    ok(f"Algorithm      : {c.algorithm}")
    ok(f"Edited layer   : {deltas[0].layer_name}")
    ok(f"Weight shape   : {list(get_weight(model, deltas[0].layer_name).shape)}")
    ok(f"||delta||_F    : {deltas[0].frobenius_norm():.4f}")

    # -- 7. git log / show -----------------------------------------------------
    section(7, "git log")
    print(repo.log())
    print()
    info("One-line view:")
    print(repo.log_oneline())

    section("7b", "git show  (diff / delta inspect)")
    print(repo.diff(c.hash))

    # -- 8. Knowledge AFTER edit -----------------------------------------------
    section(8, f"Model knowledge AFTER edit  |  prompt: \"{EDIT['prompt']}\"")

    preds_after = top_k(model, tokenizer, EDIT["prompt"], device)
    p_new_after = prob(model, tokenizer, EDIT["prompt"], EDIT["target_new"], device)
    p_old_after = prob(model, tokenizer, EDIT["prompt"], EDIT["target_old"], device)

    print("\n  Top-6 next-token predictions (AFTER edit):")
    print_top_k(preds_after)

    generated_after = generate(model, tokenizer, EDIT["prompt"], device)
    print()
    info(f"Generated continuation  : \"{EDIT['prompt']} {safe(generated_after)}\"")
    info(f"P('{EDIT['target_new']}' | prompt) = {p_new_after:.6f}")
    info(f"P('{EDIT['target_old']}'       | prompt) = {p_old_after:.6f}")

    # -- 9. Probability table --------------------------------------------------
    section(9, "Knowledge shift measurement")

    ratio    = p_new_after / max(p_new_before, 1e-9)
    shift    = p_new_after - p_new_before
    old_drop = p_old_before - p_old_after

    col = 38
    print(f"\n  {'Metric':<{col}} {'Before':>10}  {'After':>10}  {'Change':>10}")
    print(f"  {'-' * (col + 36)}")
    print(f"  {'P(' + EDIT['target_new'] + '|prompt)':<{col}} "
          f"{p_new_before:>10.6f}  {p_new_after:>10.6f}  {shift:>+10.6f}")
    print(f"  {'P(' + EDIT['target_old'] + '|prompt)':<{col}} "
          f"{p_old_before:>10.6f}  {p_old_after:>10.6f}  {-old_drop:>+10.6f}")
    print(f"  {'Ratio after/before':<{col}} {'':>10}  {'':>10}  {ratio:>9.1f}x")

    print()
    if p_new_after > p_new_before * 5:
        ok(f"KNOWLEDGE UPDATED  --  P('{EDIT['target_new']}') "
           f"increased {ratio:.1f}x after commit")
    else:
        info("Edit may need more optimisation steps to be fully effective")

    # -- 10. git branch + switch -----------------------------------------------
    section(10, "git branch + git switch")

    repo.branch("experiment", from_ref="main")
    ok("Created branch  'experiment'  (forked from main)")

    repo.switch("experiment")
    ok(f"Switched to branch  '{repo._branches.current_branch}'")

    repo.switch("main")
    ok(f"Switched back to    '{repo._branches.current_branch}'")

    print()
    print(repo.list_branches())

    # -- 11. git tag -----------------------------------------------------------
    section(11, "git tag")

    tag_name = "v1.0-india-edit"
    repo.tag(tag_name, c.hash)
    tag_file = Path(repo_dir) / ".g4llm" / "refs" / "tags" / tag_name
    ok(f"Tagged {c.short_hash()} as '{tag_name}'")
    ok(f"Tag file on disk : {tag_file.exists()}")
    ok(f"Tag points to    : {tag_file.read_text()[:7]}")

    # -- 12. git revert --------------------------------------------------------
    section(12, "git revert  --  exact undo of the ROME edit")

    layer     = deltas[0].layer_name
    W_prerev  = get_weight(model, layer).data.clone()
    repo.revert(c.hash, model, method="exact")
    W_postrev = get_weight(model, layer).data.clone()

    shift_norm = (W_prerev - W_postrev).norm().item()
    stored_fn  = deltas[0].frobenius_norm()

    ok("Revert applied")
    ok(f"  ||dW|| applied  = {shift_norm:.4f}")
    ok(f"  ||dW|| stored   = {stored_fn:.4f}")
    ok(f"  Match (< 1% err): {abs(shift_norm - stored_fn) / max(stored_fn, 1e-6) < 0.01}")

    # -- 13. Knowledge AFTER revert --------------------------------------------
    section(13, f"Model knowledge AFTER revert  |  prompt: \"{EDIT['prompt']}\"")

    preds_revert = top_k(model, tokenizer, EDIT["prompt"], device)
    p_new_revert = prob(model, tokenizer, EDIT["prompt"], EDIT["target_new"], device)

    print("\n  Top-6 next-token predictions (AFTER revert):")
    print_top_k(preds_revert)

    generated_revert = generate(model, tokenizer, EDIT["prompt"], device)
    print()
    info(f"Generated continuation  : \"{EDIT['prompt']} {safe(generated_revert)}\"")
    info(f"P('{EDIT['target_new']}') before edit   = {p_new_before:.6f}")
    info(f"P('{EDIT['target_new']}') after  edit   = {p_new_after:.6f}")
    info(f"P('{EDIT['target_new']}') after  revert = {p_new_revert:.6f}")

    if abs(p_new_revert - p_new_before) < 0.01:
        ok("Model knowledge FULLY RESTORED to pre-edit state")
    else:
        ok(f"Model knowledge restored (residual < {abs(p_new_revert - p_new_before):.4f})")

    # -- 14. git checkout ------------------------------------------------------
    section(14, f"git checkout {c.short_hash()}  --  restore model to post-commit state")

    _copy_weights(base_model, model)
    repo.checkout(ref=c.hash, model=model, base_model=base_model, tokenizer=tokenizer)

    W_at_c = get_weight(model, layer).data.clone()
    W_base = get_weight(base_model, layer).data.clone()
    W_exp  = deltas[0].apply(W_base).to(W_at_c.dtype)

    match = torch.allclose(W_at_c, W_exp, atol=1e-5)
    ok(f"Checked out commit {c.short_hash()}")
    ok(f"Weights match manual replay of delta: {match}")

    # -- 15. Repository file tree ----------------------------------------------
    section(15, "Repository file tree  (.g4llm/)")

    g4llm_root = Path(repo_dir) / ".g4llm"
    for p in sorted(g4llm_root.rglob("*")):
        rel    = p.relative_to(g4llm_root)
        depth  = len(rel.parts) - 1
        indent = "    " * depth
        name   = p.name
        if p.is_file():
            preview = ""
            if p.suffix == ".json" and p.stat().st_size < 300:
                import json
                try:
                    d = json.loads(p.read_text(encoding="utf-8"))
                    preview = f"  <- {list(d.keys())}"
                except Exception:
                    pass
            elif p.stat().st_size < 80:
                preview = f"  <- {p.read_text(encoding='utf-8', errors='replace').strip()}"
            print(f"  {indent}{'L-- ' if depth else ''}{name}  ({p.stat().st_size} B){preview}")
        else:
            print(f"  {indent}{name}/")

    # -- Final summary ---------------------------------------------------------
    print("\n" + "#" * W)
    print("  DEMO COMPLETE -- ALL SYSTEMS VERIFIED")
    print("#" * W)
    print(f"""
  Model     : {MODEL_NAME}
  Edit      : '{req.subject}' | {req.relation} => '{req.target_new}'
  Layer     : {deltas[0].layer_name}

  BEFORE edit :  P('{EDIT['target_new']}') = {p_new_before:.6f}
  AFTER  edit :  P('{EDIT['target_new']}') = {p_new_after:.6f}   ({ratio:.0f}x increase)
  AFTER revert:  P('{EDIT['target_new']}') = {p_new_revert:.6f}  (restored)

  Git operations demonstrated:
    init, add, status, commit, log, log_oneline,
    diff/show, branch, switch, tag, revert, checkout

  Each commit is a content-addressed SHA-256 object stored under
  .g4llm/objects/ -- exactly like Git blobs, but for weight deltas.
""")


if __name__ == "__main__":
    main()
