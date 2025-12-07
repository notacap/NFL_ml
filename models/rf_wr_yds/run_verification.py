"""
Quick runner for verification script.
Outputs results to a text file for easy viewing.
"""
import sys
import io
from contextlib import redirect_stdout

# Add the models directory to path
sys.path.insert(0, r"C:\Users\nocap\Desktop\code\NFL_ml\models\rf_wr_yds")

from verify_rolling_features import run_full_verification, format_report_as_json
from pathlib import Path

# Capture all output
output = io.StringIO()

parquet_path = r"C:\Users\nocap\Desktop\code\NFL_ml\models\rf_wr_yds\data\processed\nfl_wr_features_v1_20251206_163420.parquet"

with redirect_stdout(output):
    report = run_full_verification(parquet_path)

# Write console output to file
output_text = output.getvalue()
output_file = Path(parquet_path).parent / "verification_output.txt"
with open(output_file, 'w') as f:
    f.write(output_text)

print(f"Console output saved to: {output_file}")

# Also save JSON report
json_file = Path(parquet_path).parent / "verification_report.json"
with open(json_file, 'w') as f:
    f.write(format_report_as_json(report))

print(f"JSON report saved to: {json_file}")

# Print summary
print("\n" + "="*70)
print("VERIFICATION COMPLETE")
print("="*70)
print(output_text)
