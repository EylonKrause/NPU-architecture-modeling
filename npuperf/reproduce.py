"""Run the studies of docs/results.md and regenerate its tables and figures.

    python -m npuperf.reproduce                  # every study
    python -m npuperf.reproduce --quick          # the quick subset (the CI drift gate)
    python -m npuperf.reproduce --quick --check  # fail if a committed table differs
    python -m npuperf.reproduce --only S3,S6 --jobs 8

Writes results/studies/<study>.json (raw simulator records, git-ignored),
docs/figures/<figure>.svg, and the table between ``<!-- generated:NAME -->`` and
``<!-- end:NAME -->`` in docs/results.md; the text around the markers is left alone.
Every number is computed from npu-sim runs (npuperf.studies). The quick subset is the
studies marked quick; they run the same configurations in both modes, so --check
compares the tables a full run committed. --check writes no docs.
"""

from __future__ import annotations

import difflib
import os
import re
import sys
import time
from collections.abc import Mapping, Sequence

from . import cli, jsonio, studies
from .binary import ROOT

RESULTS_MD = ROOT / "docs" / "results.md"
MARKER = re.compile(r"<!-- generated:([A-Za-z0-9_.-]+) -->\n(.*?)<!-- end:\1 -->", re.S)


def sections(text: str) -> dict[str, str]:
    """Generated tables of a results document, by marker name."""
    return {match.group(1): match.group(2) for match in MARKER.finditer(text)}


def replace_sections(text: str, tables: Mapping[str, str]) -> str:
    """``text`` with each named section's body replaced; every name needs its markers."""
    missing = sorted(set(tables) - set(sections(text)))
    if missing:
        raise ValueError(f"{RESULTS_MD.name} lacks the markers of: {', '.join(missing)}")

    def body(match: re.Match) -> str:
        name = match.group(1)
        content = tables.get(name, match.group(2)).rstrip("\n")
        content = content + "\n" if content else ""
        return f"<!-- generated:{name} -->\n{content}<!-- end:{name} -->"

    return MARKER.sub(body, text)


def select(only: str | None, quick: bool) -> list[studies.Study]:
    if only:
        wanted = [item.strip().upper() for item in only.split(",") if item.strip()]
        unknown = [item for item in wanted if item not in studies.STUDIES]
        if unknown:
            raise ValueError(
                f"unknown study {', '.join(unknown)} (known: {', '.join(studies.STUDIES)})"
            )
        chosen = [studies.STUDIES[item] for item in wanted]
        if quick:
            chosen = [study for study in chosen if study.quick]
        return chosen
    return [study for study in studies.STUDIES.values() if study.quick or not quick]


def check(tables: Mapping[str, str], text: str) -> list[str]:
    """A unified diff per committed table that differs from ``tables``."""
    committed = sections(text)
    problems = []
    for name, content in tables.items():
        if name not in committed:
            problems.append(f"{RESULTS_MD.name} has no generated:{name} section\n")
            continue
        old = committed[name].rstrip("\n") + "\n"
        new = content.rstrip("\n") + "\n"
        if old != new:
            problems.append(
                "".join(
                    difflib.unified_diff(
                        old.splitlines(True),
                        new.splitlines(True),
                        f"committed {name}",
                        f"regenerated {name}",
                    )
                )
            )
    return problems


@cli.main_guard("reproduce")
def main(argv: Sequence[str] | None = None) -> int:
    parser = cli.Parser("reproduce", __doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="run only the quick studies: "
        + ", ".join(s.id for s in studies.STUDIES.values() if s.quick),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="with the quick subset, fail when a regenerated table differs "
        "from docs/results.md; writes no docs",
    )
    parser.add_argument("--only", metavar="S1,S2,...", help="run these studies only")
    parser.add_argument(
        "--jobs",
        type=cli.positive_int,
        default=os.cpu_count() or 1,
        help="simulator runs at a time",
    )
    cli.add_binary_options(parser)
    parser.set_defaults(timeout=900)
    args = parser.parse_args(argv)
    quick = args.quick or args.check
    chosen = select(args.only, quick)
    if not chosen:
        raise ValueError("no study selected")
    ctx = studies.Context(cli.simulator(args), jobs=args.jobs, timeout=args.timeout)
    started = time.perf_counter()
    tables: dict[str, str] = {}
    for study in chosen:
        began = time.perf_counter()
        output = study.run(ctx)
        tables.update(output.tables)
        jsonio.write_json(
            studies.RESULTS_DIR / f"{study.id.lower()}.json",
            {"study": study.id, "title": study.title, "tables": output.tables, **output.raw},
        )
        if not args.check:
            for name, figure in output.figures.items():
                figure.save(studies.FIGURE_DIR / f"{name}.svg")
        print(
            f"{study.id} {study.title}: {len(output.tables)} table(s), "
            f"{len(output.figures)} figure(s), {time.perf_counter() - began:.1f} s",
            file=sys.stderr,
        )
    text = RESULTS_MD.read_text(encoding="utf-8")
    if args.check:
        problems = check(tables, text)
        if problems:
            for problem in problems:
                print(problem, file=sys.stderr)
            print(
                f"reproduce: {len(problems)} generated table(s) differ from {RESULTS_MD}; "
                "run python -m npuperf.reproduce and commit the result",
                file=sys.stderr,
            )
            return cli.EXIT_FAILED
    else:
        jsonio.write_text(RESULTS_MD, replace_sections(text, tables))
    print(
        f"reproduce: {len(chosen)} studies, {len(ctx.cache) + ctx.extra_runs} simulator runs, "
        f"{time.perf_counter() - started:.1f} s",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
