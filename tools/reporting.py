"""Markdown output shared by the measurement scripts."""

from collections.abc import Sequence


def markdown_table(rows: Sequence[dict], columns: Sequence[tuple]) -> str:
    """Render dict rows as a Markdown table.

    `columns` holds (key, header) or (key, header, format) tuples, so the header
    cannot drift out of step with the keys the way two parallel lists do. The
    default format suits the levels these reports are full of; pass a third
    element for anything else.
    """
    headers = [c[1] for c in columns]
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join([":---"] * len(headers)) + " |",
    ]
    for row in rows:
        cells = []
        for column in columns:
            value = row[column[0]]
            fmt = column[2] if len(column) > 2 else "+.2f"
            cells.append(format(value, fmt) if isinstance(value, float) else str(value))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)
