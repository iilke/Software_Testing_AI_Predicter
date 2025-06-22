import re
import pandas as pd
from datetime import datetime

# === 1. Read log file ===
with open("logs_1.txt", encoding='utf-8') as f:
    lines = f.readlines()

# === 2. Regex patterns ===
testcase_start_pattern = re.compile(r"\[TESTCASE\].*?(TestCase\d+_[\w\d_]+)")
step_pattern = re.compile(r"\[STEP\]\s*\d+:\s*(.+)")
timestamp_pattern = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})")
result_pattern = re.compile(r"\b(PASS|FAIL)\b", re.IGNORECASE)

# === 3. Variables ===
test_cases = []
current_case = None
current_steps = []
current_all_words = []
current_start = None
current_result = None
last_timestamp = None

# === 4. Process Logs Line by Line ===
for line in lines:
    timestamp_match = timestamp_pattern.match(line)
    timestamp = datetime.strptime(timestamp_match.group(1), "%Y-%m-%d %H:%M:%S,%f") if timestamp_match else None

    if testcase_start_pattern.search(line):
        if current_case:
            test_cases.append({
                "test_id": current_case,
                "num_steps": len(current_steps),
                "step_keywords": " ".join(current_all_words),
                "duration": (last_timestamp - current_start).total_seconds() if last_timestamp and current_start else None,
                "result": current_result
            })
        current_case = testcase_start_pattern.search(line).group(1)
        current_steps = []
        current_all_words = []
        current_start = timestamp
        current_result = None

    if step_pattern.search(line):
        desc = step_pattern.search(line).group(1).strip()
        current_steps.append(desc)
        current_all_words.extend(desc.split())

    if result_pattern.search(line):
        match = result_pattern.search(line)
        if match:
            current_result = match.group(1).upper()

    if timestamp:
        last_timestamp = timestamp

# === 5. Add last test case ===
if current_case and current_case not in [t["test_id"] for t in test_cases]:
    test_cases.append({
        "test_id": current_case,
        "num_steps": len(current_steps),
        "step_keywords": " ".join(current_all_words),
        "duration": (last_timestamp - current_start).total_seconds() if last_timestamp and current_start else None,
        "result": current_result
    })

# === 6. Turn into a dataframe and save as csv ===
df = pd.DataFrame(test_cases)
df.to_csv("parsed_test_cases.csv", index=False)
print("CSV dosyası oluşturuldu: parsed_test_cases.csv")
