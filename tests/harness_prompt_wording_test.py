"""Guard against harness prompts that ask the model to output its reasoning.

Newer Claude models refuse turns whose prompt asks them to include "the
reasoning that led you to your move" (Anthropic ``stop_reason: refusal``,
category ``reasoning_extraction``), which voids the episode. Harness prompts
ask for a brief *explanation* of the move instead. This test scans the string
literals in every harness and ``prompt_variants`` module (bare string
statements such as docstrings are skipped, since the model never sees them)
and fails if one of the old phrasings comes back.
"""

import ast
import pathlib
import re

from absl.testing import absltest

_PKG = pathlib.Path(__file__).resolve().parent.parent / "kaggle_environments"

# Matches core_harness.py, every */harness.py and */summarize_harness.py, and
# the ablation prompt_variants.py modules (werewolf and pyxis use other names).
_HARNESS_PATTERNS = ("*harness.py", "prompt_variants.py")

_BANNED = re.compile(
    r"your\s+(\*\*)?reasoning"
    r"|the\s+reasoning\s+that"
    r"|with\s+reasoning"
    r"|treated\s+as\s+reasoning"
    r"|reason\s+step\s+by\s+step"
    r"|think\s+step[\s-]+by[\s-]+step"
    r"|chain[\s-]+of[\s-]+thought"
    r"|thought\s+process"
    r"|'thinking'\s+key",
    re.IGNORECASE,
)


def _harness_files() -> list[pathlib.Path]:
    return sorted({path for pattern in _HARNESS_PATTERNS for path in _PKG.rglob(pattern)})


def _prompt_string_literals(path: pathlib.Path):
    """Yield (lineno, text) for every string literal that is not a docstring."""
    tree = ast.parse(path.read_text())
    docstrings = {
        id(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
            yield node.lineno, node.value


class HarnessPromptWordingTest(absltest.TestCase):
    def test_scans_the_game_arena_harnesses(self):
        # Guard against the scan silently matching nothing.
        names = {p.relative_to(_PKG).as_posix() for p in _harness_files()}
        self.assertLessEqual(
            {
                "core_harness.py",
                "envs/open_spiel_env/games/go/harness.py",
                "envs/open_spiel_env/games/bargaining/prompt_variants.py",
                "envs/word_association/harness.py",
            },
            names,
        )

    def test_prompts_do_not_ask_for_reasoning(self):
        offenders = []
        for path in _harness_files():
            for lineno, text in _prompt_string_literals(path):
                match = _BANNED.search(text)
                if match:
                    offenders.append(f"{path.relative_to(_PKG)}:{lineno}: {match.group(0)!r}")
        self.assertEqual(
            offenders,
            [],
            "Harness prompts must ask for a brief explanation of the move, not the "
            "model's reasoning / thinking (trips reasoning-extraction refusals):\n" + "\n".join(offenders),
        )


if __name__ == "__main__":
    absltest.main()
