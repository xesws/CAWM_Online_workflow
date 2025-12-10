# Stage 10: Reduce Verbose Console Output in AWM Run/Eval

## Problem

AWM run/eval programs (run_infer.py, single_evaluation.py) output excessively verbose git content to the console, causing terminal overflow. The user wants to truncate output after ~500 characters.

## Root Cause

- `ActionTruncatingFilter` in `loop.py` only truncates ACTION messages to 100 chars
- **OBSERVATION messages (containing git diff/status output) are NOT truncated at console level**
- `CmdOutputObservation` has 30,000 char limit - still too much for console display

## Solution

Extend the existing `ActionTruncatingFilter` in `evaluation/awm/loop.py` to also truncate OBSERVATION messages.

### Modified File: `evaluation/awm/loop.py`

**Current code (lines 30-43):**
```python
class ActionTruncatingFilter(logging.Filter):
    """Filter to truncate verbose action logs in console output."""
    def filter(self, record: logging.LogRecord) -> bool:
        if getattr(record, 'msg_type', '') == 'ACTION':
            msg_str = str(record.msg)
            if len(msg_str) > 100:
                record.msg = msg_str[:100] + "... [truncated]"
        return True
```

**New code:**
```python
class LogTruncatingFilter(logging.Filter):
    """Filter to truncate verbose action and observation logs in console output."""

    # Configurable truncation limits
    ACTION_MAX_CHARS = 500
    OBSERVATION_MAX_CHARS = 500

    def filter(self, record: logging.LogRecord) -> bool:
        msg_type = getattr(record, 'msg_type', '')
        msg_str = str(record.msg)

        if msg_type == 'ACTION' and len(msg_str) > self.ACTION_MAX_CHARS:
            record.msg = msg_str[:self.ACTION_MAX_CHARS] + "... [truncated]"
        elif msg_type == 'OBSERVATION' and len(msg_str) > self.OBSERVATION_MAX_CHARS:
            record.msg = msg_str[:self.OBSERVATION_MAX_CHARS] + "... [truncated]"

        return True
```

### Also update `_setup_logging()` (line 87)

Update the filter name from `ActionTruncatingFilter` to `LogTruncatingFilter`.

## Implementation Steps

1. Read `evaluation/awm/loop.py` to get exact current code
2. Rename `ActionTruncatingFilter` to `LogTruncatingFilter`
3. Add OBSERVATION message handling with configurable limit
4. Update `_setup_logging()` to use the renamed filter

## Expected Result

- Git diff/status output will be truncated to 500 chars in console
- ACTION messages will be truncated to 500 chars (increased from 100 for better visibility)
- Full content still preserved in actual events/history, only console display is truncated
