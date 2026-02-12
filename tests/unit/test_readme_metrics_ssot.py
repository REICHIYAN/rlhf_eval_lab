from __future__ import annotations

import re
from pathlib import Path

from rlhf_eval_lab.utils.update_readme_metrics_ssot import (
    END_MARKER,
    START_MARKER,
    render_metrics_ssot_block,
)


def test_readme_metrics_ssot_block_is_up_to_date() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert START_MARKER in readme and END_MARKER in readme, (
        "README.md にSSOTマーカーがありません。次の2行を追加してください:\n"
        f"{START_MARKER}\n{END_MARKER}\n"
    )

    m = re.search(rf"{re.escape(START_MARKER)}(.*?){re.escape(END_MARKER)}", readme, flags=re.S)
    assert m, "SSOT marker block not found"

    current = m.group(1).strip() + "\n"
    expected = "\n\n" + render_metrics_ssot_block().strip() + "\n"

    assert current == expected.strip() + "\n"
