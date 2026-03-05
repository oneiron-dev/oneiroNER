"""Sliding window over conversation turns for long dialogues."""


def window_turns(
    turns: list[dict], size: int = 4, stride: int = 2
) -> list[list[dict]]:
    n = len(turns)
    if n <= size:
        return [turns]

    windows = []
    for start in range(0, n, stride):
        end = min(start + size, n)
        window = turns[start:end]
        if len(window) < 2:
            break
        windows.append(window)
        if end == n:
            break

    return windows
