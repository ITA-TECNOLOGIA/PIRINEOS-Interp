# ── utils/result_saver.py ──────────────────────────────────────────────────────
import os, json, datetime, pathlib, torch

class ResultSaver:
    """
    Collect one-row-per-question metadata in memory and dump it as JSON.
    Heavy arrays (entropy, KL, trajectories, …) are saved per-question as
    .pt files; the JSON just stores their relative paths.
    """

    def __init__(self, model_name: str, root: str = "eval_results"):
        ts   = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.dir = pathlib.Path(root) / f"{model_name}_{ts}"
        self.dir.mkdir(parents=True, exist_ok=True)

        self.rows = []                 # list of dicts – one per problem

    # --------------- small helpers ------------------
    def _dump_tensor(self, tensor, stem, qid):
        """Save tensor→pt, return relative fname (str)."""
        fname = f"{qid}_{stem}.pt"
        torch.save(tensor.cpu(), self.dir / fname)
        return fname
    
    def _dump_any(self, obj, stem, qid):
        fname = f"{qid}_{stem}.pt"
        torch.save(obj, self.dir / fname)
        return fname

    def add(self,
            qid: str,
            question: str,
            ground_answer_text: str,
            model_answers_text: list[str],
            extracted_model_answers: list[str],
            extracted_ground_answer: str,
            majority_answer: str,
            correct: bool,
            output_tokens: list[int],
            entropy=None, forward_kl=None, layer_avg=None,
            **extra_blobs):
        """
         Any *additional* keyword (e.g. topk_final_ids, topk_final_tokens, …)
+        is picked up by **extra_blobs and written to a separate .pt file.
        """

        row = dict(
            id=qid,
            question=question,
            ground_answer_text=ground_answer_text,
            model_answers_text=model_answers_text,
            extracted_model_answers=extracted_model_answers,
            extracted_ground_answer=extracted_ground_answer,
            majority_answer=majority_answer,
            correct=correct,
            output_tokens=output_tokens,      # list[int] – JSON-serialisable
        )

        # ───── fixed big blobs ────────────────────────────────
        if entropy is not None:
            row["entropy_file"] = self._dump_any(entropy, "entropy", qid)
        if forward_kl is not None:
            row["forward_kl_file"] = self._dump_any(forward_kl, "forward_kl", qid)
        if layer_avg is not None:
            row["layer_avg_logp_file"] = self._dump_any(layer_avg, "layer_avg_logp", qid)

        # ───── variable extras (top-k etc.) ───────────────────
        for key, obj in extra_blobs.items():
            if obj is None:
                continue
            # each key gets its own file, e.g.  000123_topk_final_ids.pt
            row[f"{key}_file"] = self._dump_any(obj, key, qid)
            
        self.rows.append(row)

    # ---------------- final dump --------------------
    def close(self):
        meta_path = self.dir / "index.json"
        with open(meta_path, "w", encoding="utf-8") as fp:
            json.dump(self.rows, fp, indent=2, ensure_ascii=False)
        print(f"🎯  Saved {len(self.rows)} rows ➜ {meta_path}")
