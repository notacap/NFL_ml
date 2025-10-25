"""
Generate comprehensive test report for NFL receiving statistics testing.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Test results summary
report_data = {
    "test_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "script_under_test": "receiving.py",
    "total_tests": 29,
    "tests_run": 27,  # 2 skipped due to missing data
    "tests_passed": 17,
    "tests_failed": 10,
    "pass_rate": 63.0,  # 17/27 * 100

    "critical_tests": {
        "total": 23,
        "passed": 13,
        "failed": 10,
        "pass_rate": 56.5
    },

    "edge_case_tests": {
        "total": 4,
        "passed": 3,
        "failed": 1,
        "pass_rate": 75.0
    },

    "test_results": {
        "ground_truth": {
            "week_17_exists": "✓ PASSED (all 3 seasons)",
        },
        "calculation_logic": {
            "adot_calculation": "✓ PASSED - Correctly calculated from cumulative totals",
            "passer_rating_known": "✓ PASSED - Matches expected value",
            "passer_rating_perfect": "✗ FAILED - Got 156.2 instead of 158.3 (test case issue)",
            "passer_rating_zero_attempts": "✓ PASSED - Returns NaN",
            "rate_stats_not_averaged": "✓ PASSED - Recalculated from cumulative, not averaged",
            "safe_divide": "✓ PASSED - Handles zero denominator"
        },
        "data_integrity": {
            "monotonic_progression": "✗ FAILED - Air yards and yards decreased for some players",
            "week_1_equals_game": "✗ FAILED - First downs mismatch between game and cumulative",
            "no_missing_players": "✓ PASSED (all 4 weeks tested)",
            "no_duplicates": "✓ PASSED (all 4 weeks tested)",
            "column_completeness": "Not tested (integrity test, not critical)"
        },
        "edge_cases": {
            "zero_targets": "✓ PASSED - No errors, rate stats = NaN",
            "zero_receptions": "✓ PASSED - Catch % = 0, yards/rec = NaN",
            "mid_season_start": "✓ PASSED - Cumulative from first appearance",
            "longest_vs_total": "✗ FAILED - 4 players have longest > total"
        }
    },

    "critical_findings": [
        {
            "issue": "Monotonic Progression Violations",
            "severity": "HIGH",
            "description": "Air yards (AYBC) and receiving yards decreased week-over-week for multiple players across all seasons",
            "affected": "~40-44 players per season for AYBC, 1-4 players per season for yards",
            "likely_cause": "Data quality issue OR script is not truly accumulating (recalculating instead)",
            "recommendation": "Investigate specific player IDs to determine if this is data correction or logic error"
        },
        {
            "issue": "Week 1 First Downs Mismatch",
            "severity": "MEDIUM",
            "description": "First downs differ between week 1 game data and week 1 cumulative data",
            "affected": "~62-67 players per season",
            "likely_cause": "Column mapping issue or data source discrepancy",
            "recommendation": "Review first down calculation logic and data sources"
        },
        {
            "issue": "Perfect Passer Rating Test Failure",
            "severity": "LOW",
            "description": "Test expected 158.3 but got 156.2 for perfect rating scenario",
            "affected": "Test case only",
            "likely_cause": "Test case setup incorrect (need more TDs or higher YPA to reach max)",
            "recommendation": "Adjust test case parameters"
        },
        {
            "issue": "Longest Reception > Total Yards",
            "severity": "LOW",
            "description": "4 players have longest reception exceeding total yards",
            "affected": "4 players in 2024 season",
            "likely_cause": "Data quality issue for low-volume players",
            "recommendation": "Review data for players 13873, 14479, 14586, 14588"
        }
    ],

    "positive_findings": [
        "✓ ADOT correctly calculated from cumulative totals (NOT averaged)",
        "✓ Rate statistics recalculated from cumulative, not averaged from game rates",
        "✓ Passer rating formula correctly implemented with component capping",
        "✓ Zero denominator handled gracefully (returns NaN)",
        "✓ All players from game data appear in cumulative output",
        "✓ No duplicate player-week combinations",
        "✓ Edge cases handled without errors (zero targets, zero receptions, mid-season starts)"
    ],

    "recommendations": {
        "immediate_action": [
            "1. Investigate monotonic progression violations - CRITICAL",
            "2. Review first downs calculation and data sources",
            "3. Examine the 4 players with longest > total yards data anomalies"
        ],
        "before_production": [
            "1. Add ground truth validation against known season totals",
            "2. Add data quality checks in the script itself",
            "3. Implement monotonic progression validation in the script",
            "4. Review and fix any data corrections that cause decreases"
        ],
        "test_improvements": [
            "1. Fix perfect passer rating test case",
            "2. Add more granular tests for specific players",
            "3. Add validation against authoritative NFL stats",
            "4. Test with 2025 data when available"
        ]
    },

    "production_readiness": {
        "status": "NOT READY",
        "pass_rate_threshold": "95%",
        "actual_pass_rate": "63%",
        "critical_issues": 2,
        "blocking_issues": [
            "Monotonic progression violations (HIGH priority)",
            "Week 1 first downs mismatch (MEDIUM priority)"
        ],
        "recommendation": "DO NOT USE IN PRODUCTION until monotonic progression and first downs issues are resolved and validated"
    }
}


def generate_markdown_report():
    """Generate comprehensive markdown report."""

    report = []

    # Header
    report.append("# NFL Receiving Cumulative Statistics - Test Report")
    report.append("")
    report.append(f"**Test Date**: {report_data['test_date']}")
    report.append(f"**Script Under Test**: {report_data['script_under_test']}")
    report.append(f"**Test Framework**: pytest with custom NFL testing framework")
    report.append("")
    report.append("---")
    report.append("")

    # Executive Summary
    report.append("## Executive Summary")
    report.append("")
    report.append(f"**Overall Pass Rate**: {report_data['pass_rate']:.1f}% ({report_data['tests_passed']}/{report_data['tests_run']} tests)")
    report.append(f"**Production Readiness**: **{report_data['production_readiness']['status']}**")
    report.append("")

    if report_data['production_readiness']['status'] == "NOT READY":
        report.append("⚠️ **CRITICAL**: This script has {0} blocking issues and should NOT be used in production until resolved.".format(
            report_data['production_readiness']['critical_issues']))
        report.append("")

    report.append("### Test Categories")
    report.append("")
    report.append("| Category | Tests | Passed | Failed | Pass Rate |")
    report.append("|----------|-------|--------|--------|-----------|")
    report.append(f"| Critical Tests | {report_data['critical_tests']['total']} | "
                 f"{report_data['critical_tests']['passed']} | "
                 f"{report_data['critical_tests']['failed']} | "
                 f"{report_data['critical_tests']['pass_rate']:.1f}% |")
    report.append(f"| Edge Case Tests | {report_data['edge_case_tests']['total']} | "
                 f"{report_data['edge_case_tests']['passed']} | "
                 f"{report_data['edge_case_tests']['failed']} | "
                 f"{report_data['edge_case_tests']['pass_rate']:.1f}% |")
    report.append("")
    report.append("---")
    report.append("")

    # Critical Findings
    report.append("## Critical Findings")
    report.append("")
    for i, finding in enumerate(report_data['critical_findings'], 1):
        report.append(f"### {i}. {finding['issue']} [{finding['severity']}]")
        report.append("")
        report.append(f"**Description**: {finding['description']}")
        report.append("")
        report.append(f"**Affected**: {finding['affected']}")
        report.append("")
        report.append(f"**Likely Cause**: {finding['likely_cause']}")
        report.append("")
        report.append(f"**Recommendation**: {finding['recommendation']}")
        report.append("")

    report.append("---")
    report.append("")

    # Detailed Test Results
    report.append("## Detailed Test Results")
    report.append("")

    report.append("### Ground Truth Validation")
    report.append("")
    for test, result in report_data['test_results']['ground_truth'].items():
        report.append(f"- **{test}**: {result}")
    report.append("")

    report.append("### Calculation Logic Tests")
    report.append("")
    for test, result in report_data['test_results']['calculation_logic'].items():
        report.append(f"- **{test}**: {result}")
    report.append("")

    report.append("### Data Integrity Tests")
    report.append("")
    for test, result in report_data['test_results']['data_integrity'].items():
        report.append(f"- **{test}**: {result}")
    report.append("")

    report.append("### Edge Case Tests")
    report.append("")
    for test, result in report_data['test_results']['edge_cases'].items():
        report.append(f"- **{test}**: {result}")
    report.append("")

    report.append("---")
    report.append("")

    # Positive Findings
    report.append("## Positive Findings")
    report.append("")
    report.append("The following aspects of the implementation are **CORRECT**:")
    report.append("")
    for finding in report_data['positive_findings']:
        report.append(f"{finding}")
    report.append("")
    report.append("---")
    report.append("")

    # Recommendations
    report.append("## Recommendations")
    report.append("")

    report.append("### Immediate Action Required")
    report.append("")
    for rec in report_data['recommendations']['immediate_action']:
        report.append(f"{rec}")
    report.append("")

    report.append("### Before Production Use")
    report.append("")
    for rec in report_data['recommendations']['before_production']:
        report.append(f"{rec}")
    report.append("")

    report.append("### Test Improvements")
    report.append("")
    for rec in report_data['recommendations']['test_improvements']:
        report.append(f"{rec}")
    report.append("")

    report.append("---")
    report.append("")

    # Production Readiness
    report.append("## Production Readiness Assessment")
    report.append("")
    report.append(f"**Status**: **{report_data['production_readiness']['status']}**")
    report.append("")
    report.append(f"**Pass Rate**: {report_data['production_readiness']['actual_pass_rate']}% "
                 f"(Threshold: {report_data['production_readiness']['pass_rate_threshold']})")
    report.append("")
    report.append(f"**Critical Issues**: {report_data['production_readiness']['critical_issues']}")
    report.append("")
    report.append("**Blocking Issues**:")
    for issue in report_data['production_readiness']['blocking_issues']:
        report.append(f"- {issue}")
    report.append("")
    report.append(f"**Recommendation**: {report_data['production_readiness']['recommendation']}")
    report.append("")

    report.append("---")
    report.append("")

    # Appendix
    report.append("## Appendix")
    report.append("")
    report.append("### Test Environment")
    report.append("- Python 3.13.5")
    report.append("- pytest 8.4.2")
    report.append("- pandas (latest)")
    report.append("- pyarrow (latest)")
    report.append("")
    report.append("### Test Files")
    report.append("- `test_receiving_stats.py` - Main test suite (29 tests)")
    report.append("- `test_framework.py` - Reusable testing framework")
    report.append("- `conftest.py` - pytest configuration and fixtures")
    report.append("- `script_analysis.md` - Detailed implementation analysis")
    report.append("")
    report.append("### Data Sources")
    report.append("- Input: `C:/Users/nocap/Desktop/code/NFL_ml/parquet_files/raw/plyr_gm/plyr_gm_rec`")
    report.append("- Output: `C:/Users/nocap/Desktop/code/NFL_ml/parquet_files/raw/plyr_szn/plyr_rec`")
    report.append("- Seasons Tested: 2022, 2023, 2024")
    report.append("- Weeks Tested: 1-17")
    report.append("")

    report.append("---")
    report.append("")
    report.append("**Report Generated**: {0}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    report.append("")

    return "\n".join(report)


if __name__ == "__main__":
    # Generate report
    report_content = generate_markdown_report()

    # Save to file
    output_file = Path(__file__).parent / "test_results" / "test_report.md"
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report_content)

    print(f"Test report generated: {output_file}")
    print("")
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Pass Rate: {report_data['pass_rate']:.1f}%")
    print(f"Status: {report_data['production_readiness']['status']}")
    print("=" * 80)

    # Also save as JSON
    json_file = Path(__file__).parent / "test_results" / "test_results.json"
    with open(json_file, 'w') as f:
        json.dump(report_data, f, indent=2)

    print(f"JSON results saved: {json_file}")
