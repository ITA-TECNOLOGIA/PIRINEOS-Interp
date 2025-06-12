# ── utils/result_saver.py ──────────────────────────────────────────────────────
import os, json, datetime, pathlib, torch
import tempfile

import os
import psutil
import time

# Call this function wherever you want to log CPU and RAM usage.
def log_resource_usage(label=""):
    process = psutil.Process(os.getpid())
    # CPU percent since last call; use a small interval to get an updated snapshot
    cpu_pct = process.cpu_percent(interval=0.1)
    # Resident Set Size (RSS) in bytes → convert to GB
    mem_bytes = process.memory_info().rss
    mem_gb = mem_bytes / (1024 ** 3)
    print(f"[RESOURCE] {label} | CPU%: {cpu_pct:.2f}% | RAM: {mem_gb:.2f} GB")

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
        
        # first index.json save
        self.meta_path = self.dir / "index.json"
        with open(self.meta_path, "w", encoding="utf-8") as fp:
            json.dump([], fp, indent=2, ensure_ascii=False)

        self.rows = []                 # list of dicts – one per problem

    # --------------- small helpers ------------------
    def _dump_tensor(self, tensor, stem, qid):
        """Save tensor→pt, return relative fname (str)."""
        fname = f"{qid}_{stem}.pt"
        #log_resource_usage(f"before taking cpu tensor {fname}")
        tensor_to_save = tensor.cpu()
        #log_resource_usage(f"before dumping tensor {fname}")
        torch.save(tensor_to_save, self.dir / fname)
        #log_resource_usage(f"after dumping tensor {fname}")
        return fname
    
    def _dump_any(self, obj, stem, qid):
        fname = f"{qid}_{stem}.pt"
        #log_resource_usage(f"before dumping any {fname}")
        torch.save(obj, self.dir / fname)
        #log_resource_usage(f"after dumping any {fname}")
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
        #log_resource_usage(f"before creating row")
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
        #log_resource_usage(f"after appending row")
        
        # Save the metadata at each step
        self._flush_rows_to_disk()
        
    def _flush_rows_to_disk(self):
        # In order to avoid problems with concurrent writes,
        # we use an atomic file write strategy:
        # 1) Write the current rows to a temporary file
        # replace the old index.json with the temporary file
        #log_resource_usage(f"before flushing")
        fd, tmp_path = tempfile.mkstemp(dir=str(self.dir), suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as tmpf:
                json.dump(self.rows, tmpf, indent=2, ensure_ascii=False)
                #log_resource_usage(f"after dumping")
                tmpf.flush()
                #log_resource_usage(f"after flushing")
                os.fsync(tmpf.fileno())

            # 2) Atomically replace the old index.json
            os.replace(tmp_path, self.meta_path)
        except Exception:
            # If anything goes wrong, remove the temp file (best effort).
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            raise

    # ---------------- final dump --------------------
    def close(self):
        meta_path = self.dir / "index.json"
        with open(meta_path, "w", encoding="utf-8") as fp:
            json.dump(self.rows, fp, indent=2, ensure_ascii=False)
        print(f"🎯  Saved {len(self.rows)} rows ➜ {meta_path}")
