from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from config import ValidatorConfig
from src.image_loader import SUPPORTED_EXTENSIONS
from src.validator import PersonPhotoValidator


def iter_images(path: Path):
    if path.is_file():
        yield path
        return

    for item in sorted(path.rglob("*")):
        if item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield item


def print_structured_report(result: dict) -> None:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    flag = result.get("flag", "unknown")
    if flag == "acceptable":
        flag_str = f"{GREEN}{BOLD}ACCEPTABLE{RESET}"
        reason_icon = f"{GREEN}✓{RESET}"
    elif flag == "rejected":
        flag_str = f"{RED}{BOLD}REJECTED{RESET}"
        reason_icon = f"{RED}✗{RESET}"
    else:
        flag_str = f"{YELLOW}{BOLD}MANUAL VERIFICATION{RESET}"
        reason_icon = f"{YELLOW}⚠{RESET}"

    print(f"\n{BLUE}{'=' * 60}{RESET}")
    print(f"{BOLD}📷 IMAGE VALIDATION REPORT{RESET}")
    print(f"{BLUE}{'=' * 60}{RESET}")
    print(f"{BOLD}File:{RESET} {result.get('image_path')}")
    print(f"{BOLD}Status:{RESET} {flag_str}")
    
    reasons = result.get("reasons", [])
    if reasons:
        print(f"\n{BOLD}Details:{RESET}")
        for r in reasons:
            print(f"  {reason_icon} {r}")
            
    warnings = result.get("warnings", [])
    if warnings:
        print(f"\n{YELLOW}{BOLD}Warnings:{RESET}")
        for w in warnings:
            print(f"  ⚠ {w}")

    scores = result.get("scores", {})
    if scores:
        print(f"\n{BOLD}Metrics & Scores:{RESET}")
        print(f"  • Resolution:      {scores.get('image_width')}x{scores.get('image_height')} px")
        print(f"  • Blur Score:      {scores.get('blur_score'):.2f} (higher is sharper)")
        print(f"  • Brightness:      {scores.get('brightness'):.2f}")
        print(f"  • Contrast:        {scores.get('contrast'):.2f}")
        print(f"  • Face Count:      {scores.get('face_count')}")
        if scores.get("face_confidence") is not None:
            print(f"  • Face Confidence: {scores.get('face_confidence') * 100:.1f}%")
        if scores.get("body_status") is not None:
            print(f"  • Body Status:     {scores.get('body_status')} (Confidence: {scores.get('body_confidence', 0) * 100:.1f}%)")
        if scores.get("ai_suspicion_score") is not None:
            print(f"  • AI Suspicion Score: {scores.get('ai_suspicion_score') * 100:.1f}%")
    print(f"{BLUE}{'=' * 60}{RESET}\n")


def print_summary_table(results: list[dict]) -> None:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    print(f"\n{BLUE}{'=' * 90}{RESET}")
    print(f"{BOLD}📊 VALIDATION SUMMARY TABLE{RESET}")
    print(f"{BLUE}{'=' * 90}{RESET}")
    print(f"{BOLD}{'Image Path':<50} | {'Status':<20} | {'Reasons'}{RESET}")
    print(f"{'-' * 90}")
    for r in results:
        flag = r.get("flag", "unknown")
        if flag == "acceptable":
            flag_str = f"{GREEN}ACCEPTABLE{RESET}"
        elif flag == "rejected":
            flag_str = f"{RED}REJECTED{RESET}"
        else:
            flag_str = f"{YELLOW}MANUAL VERI{RESET}"
            
        path = r.get("image_path", "")
        if len(path) > 48:
            path = "..." + path[-45:]
            
        reasons_str = ", ".join(r.get("reasons", []))
        if len(reasons_str) > 30:
            reasons_str = reasons_str[:27] + "..."
            
        print(f"{path:<50} | {flag_str:<29} | {reasons_str}")
    print(f"{BLUE}{'=' * 90}{RESET}\n")


def write_csv(results: list[dict], output_path: Path) -> None:
    if not results:
        return

    headers = [
        "image_path",
        "flag",
        "reasons",
        "warnings",
        "face_count",
        "body_status",
        "body_confidence",
        "face_confidence",
        "ai_suspicion_score",
        "blur_score",
        "brightness",
        "contrast",
        "image_width",
        "image_height",
        "face_height_ratio",
        "face_width_ratio",
        "space_below_face_ratio",
        "face_center_x_ratio",
        "face_center_y_ratio"
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            row = {
                "image_path": r.get("image_path"),
                "flag": r.get("flag"),
                "reasons": "; ".join(r.get("reasons", [])),
                "warnings": "; ".join(r.get("warnings", [])),
            }
            scores = r.get("scores", {})
            for k, v in scores.items():
                row[k] = v
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate person photos using SCRFD + MediaPipe Pose")
    parser.add_argument("input", help="Image file or folder path")
    parser.add_argument("--model", default=None, help="Path to SCRFD 2.5G ONNX model")
    parser.add_argument("--output", default="outputs/results.json", help="Output results path (JSON or CSV)")
    parser.add_argument("--pretty", action="store_true", help="Print formatted JSON")
    args = parser.parse_args()

    cfg = ValidatorConfig()
    if args.model:
        cfg = ValidatorConfig(scrfd_model_path=args.model)

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    validator = PersonPhotoValidator(cfg)
    results = []
    is_single_file = input_path.is_file()

    try:
        for image_path in iter_images(input_path):
            try:
                result = validator.validate(image_path)
                results.append(result)
                if not is_single_file:
                    print(f"Processed: {image_path} -> {result['flag']}")
            except Exception as exc:
                error_result = {
                    "image_path": str(image_path),
                    "flag": "manual_verification",
                    "reasons": ["Processing failed"],
                    "error": str(exc),
                }
                results.append(error_result)
                if not is_single_file:
                    print(f"Processed: {image_path} -> ERROR: {exc}")
    finally:
        validator.close()

    if not results:
        print("No supported images found to validate.")
        return

    # Print structured report(s)
    if is_single_file:
        print_structured_report(results[0])
    else:
        print_summary_table(results)

    # Save to output file (JSON or CSV)
    if output_path.suffix.lower() == ".csv":
        write_csv(results, output_path)
        print(f"Results successfully saved to CSV: {output_path}")
    else:
        payload = results[0] if is_single_file and len(results) == 1 else results
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Results successfully saved to JSON: {output_path}")

    if args.pretty:
        if output_path.suffix.lower() == ".csv":
            # Just print the CSV content
            print(output_path.read_text(encoding="utf-8"))
        else:
            print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
