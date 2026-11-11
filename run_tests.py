# Wrapper: run tests.py, write output to test_output.txt, and print it.
import subprocess
import sys

result = subprocess.run(
    [sys.executable, "-u", "tests.py"],
    capture_output=False,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    encoding="utf-8",
    errors="replace",
)

output = result.stdout
print(output, end="")

with open("test_output.txt", "w", encoding="utf-8") as f:
    f.write(output)

print(f"\n[Exit code: {result.returncode}]")

