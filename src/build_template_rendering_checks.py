from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import confirmatory_common as cf


def main():
    rows = [json.loads(x) for x in open(cf.CONFIRMATORY / "memory_smoke.jsonl")]
    checks = {}
    for row in rows:
        t = row["template_rendering_check"]
        cid = row["carrier_id"]
        suffix = t["assistant_boundary_suffix"]
        prompt_file = cf.CONFIRMATORY / "rendered_prompts" / f"{cid}-ascii.jsonl"
        rendered_rows = cf.read_jsonl(prompt_file) if prompt_file.exists() else []
        full_rendered = rendered_rows[0]["rendered_prompt"] if rendered_rows else suffix
        expected_user = cf.copy_user_content(cf.FEWSHOT_EXEMPLARS[0])
        checks[cid] = {
            **t,
            "smoke_prompt_sha256": cf.sha_text(row["smoke_prompt"]),
            "frozen_multiturn_user_content_present": expected_user in full_rendered,
            "ends_at_assistant_generation_boundary": suffix.rstrip().endswith(("<|assistant|>", "assistant\n", "<start_of_turn>model", "<|im_start|>assistant", "[/INST]", "<|start_header_id|>assistant<|end_header_id|>")) or suffix.endswith("<|im_start|>assistant\n") or suffix.endswith("<start_of_turn>model\n") or suffix.endswith("<|assistant|>\n") or suffix.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n"),
            "overlay_used": row["preparation"]["overlay_used"],
            "memory_smoke_pass": row["smoke_output"] == "<OUT>test</OUT>",
        }
    obj = {"created_utc": datetime.now(timezone.utc).isoformat(), "checks": checks, "template_rendering_checks_pass": bool(checks) and all(x["frozen_multiturn_user_content_present"] and x["ends_at_assistant_generation_boundary"] for x in checks.values())}
    cf.write_json(cf.CONFIRMATORY / "template_rendering_checks.json", obj)
    print(json.dumps({"carriers": sorted(checks), "template_rendering_checks_pass": obj["template_rendering_checks_pass"]}, indent=2))
    if not obj["template_rendering_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
