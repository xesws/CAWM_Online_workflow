# Stage 6.2: Logging Refactor and Resilience

## Objective
Refactor the AWM evaluation loop to improve console output readability and data persistence resilience.

## Changes
1.  **Console Output Truncation**:
    -   Implemented `ActionTruncatingFilter` in `OpenHands/evaluation/awm/loop.py`.
    -   Truncates `ACTION` log messages to 100 characters in the console output to reduce verbosity.

2.  **Real-time Result Logging**:
    -   Added `results.jsonl` logging in `AWMOnlineLoop`.
    -   Initializes `results.jsonl` at the start of the run.
    -   Appends each processing result (experience or error) immediately after processing, ensuring data is saved even if the run crashes or is interrupted.

## Verification
-   Verified `ActionTruncatingFilter` logic with unit tests (truncated string and object messages).
-   Verified `results.jsonl` creation and appending logic.
