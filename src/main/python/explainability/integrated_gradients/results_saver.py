# utils/attribution_result_saver.py

import os
import json
import datetime
import pathlib
import tempfile
import torch


class AttributionResultSaver:
    """
    Collect one-row-per-question attribution data in memory and dump it as JSON.
    Heavy arrays (normalized attributions, raw tensors, etc.) are saved per-question
    as .pt files; the JSON just stores their relative paths.
    """

    def __init__(self, model_name: str, root: str = "eval_results"):
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.dir = pathlib.Path(root) / f"{model_name}_{ts}"
        self.dir.mkdir(parents=True, exist_ok=True)

        # initialize metadata file
        self.meta_path = self.dir / "index.json"
        with open(self.meta_path, "w", encoding="utf-8") as fp:
            json.dump([], fp, indent=2, ensure_ascii=False)

        self.rows = []

    def _dump_tensor(self, tensor: torch.Tensor, stem: str, qid: str) -> str:
        """
        Save a tensor to disk as <qid>_<stem>.pt and return its filename.
        """
        fname = f"{qid}_{stem}.pt"
        path = self.dir / fname
        torch.save(tensor.cpu(), path)
        return fname

    def _dump_any(self, obj, stem: str, qid: str) -> str:
        """
        Save any picklable object to disk as <qid>_<stem>.pt and return its filename.
        """
        fname = f"{qid}_{stem}.pt"
        path = self.dir / fname
        torch.save(obj, path)
        return fname

    def add(
        self,
        qid: str,
        question: str,
        correct: bool,
        numeric_answer: float,
        true_answer: float,
        input_tokens: list[str],
        output_tokens: list[str],
        # target logprob
        first_target_logprob_attr: torch.Tensor,
        avg_input_target_logprob_attr: float,
        avg_output_target_logprob_attr: float,
        # target entropy
        first_target_entropy_attr: torch.Tensor,
        avg_input_target_entropy_attr: float,
        avg_output_target_entropy_attr: float,
        # sequence_logprob
        seq_logprob_attr: torch.Tensor,
        avg_input_seq_logprob_attr: float,
        avg_output_seq_logprob_attr: float,
        # sequence entropy
        seq_entropy_attr: torch.Tensor,
        avg_input_seq_entropy_attr: float,
        avg_output_seq_entropy_attr: float,
        **extra_blobs
    ):
        """
        Add one example's attribution results.

        Args:
          qid:               unique question ID (used for filenames)
          question:          the question text
          input_tokens:      list of input token strings
          output_tokens:     list of output token strings (or ints)
          normed_attr:       1D Tensor [seq_len] of normalized attributions
          avg_input_attr:    float, average over the input-token attributions
          avg_output_attr:   float, average over the output-token attributions
          **extra_blobs:     any additional tensors/objects to save per example
        """
        row = {
            "id": qid,
            "question": question,
            "correct": correct,
            "numeric_answer": numeric_answer,
            "true_answer": true_answer,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "avg_input_target_logprob_attr": avg_input_target_logprob_attr,
            "avg_output_target_logprob_attr": avg_output_target_logprob_attr,
            "avg_input_target_entropy_attr": avg_input_target_entropy_attr,
            "avg_output_target_entropy_attr": avg_output_target_entropy_attr,
            "avg_input_seq_logprob_attr": avg_input_seq_logprob_attr,
            "avg_output_seq_logprob_attr": avg_output_seq_logprob_attr,
            "avg_input_seq_entropy_attr": avg_input_seq_entropy_attr,
            "avg_output_seq_entropy_attr": avg_output_seq_entropy_attr,
        }

        # save normalized attributions tensor
        row["first_target_logprob_attr_file"] = self._dump_tensor(
            first_target_logprob_attr, "first_target_logprob_attr", qid
        )
        row["first_target_entropy_attr_file"] = self._dump_tensor(
            first_target_entropy_attr, "first_target_entropy_attr", qid
        )
        row["seq_logprob_attr_file"] = self._dump_tensor(
            seq_logprob_attr, "seq_logprob_attr", qid
        )
        row["seq_entropy_attr_file"] = self._dump_tensor(
            seq_entropy_attr, "seq_entropy_attr", qid
        )

        # save any extra blobs as .pt files
        for key, obj in extra_blobs.items():
            if obj is None:
                continue
            # e.g. qid_raw_attr.pt, qid_hidden_states.pt, etc.
            row[f"{key}_file"] = self._dump_any(obj, key, qid)

        self.rows.append(row)
        self._flush_rows_to_disk()

    def _flush_rows_to_disk(self):
        """
        Atomically overwrite index.json with current rows.
        """
        fd, tmp_path = tempfile.mkstemp(dir=str(self.dir), suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as tmpf:
                json.dump(self.rows, tmpf, indent=2, ensure_ascii=False)
                tmpf.flush()
                os.fsync(tmpf.fileno())
            os.replace(tmp_path, self.meta_path)
        except Exception:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            raise

    def close(self):
        """
        Final write of the metadata JSON.
        """
        with open(self.meta_path, "w", encoding="utf-8") as fp:
            json.dump(self.rows, fp, indent=2, ensure_ascii=False)
        print(f"🎯  Saved {len(self.rows)} rows ➜ {self.meta_path}")
