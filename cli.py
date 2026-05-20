#!/usr/bin/env python3
"""
PhishGuard CLI — Phishing Email Detection
Usage:
    python cli.py                      # interactive mode
    python cli.py "email text here"    # single email
    python cli.py -f emails.txt        # batch from file
    python cli.py --train              # retrain model
"""

import sys, os, argparse, json

R = '\033[91m'; G = '\033[92m'; Y = '\033[93m'
B = '\033[94m'; P = '\033[95m'; C = '\033[96m'
W = '\033[97m'; D = '\033[90m'; X = '\033[0m'; BOLD = '\033[1m'

BANNER = f"""
{R}{BOLD} ██████╗ ██╗  ██╗██╗███████╗██╗  ██╗
{R} ██╔══██╗██║  ██║██║██╔════╝██║  ██║
{R} ██████╔╝███████║██║███████╗███████║
{R} ██╔═══╝ ██╔══██║██║╚════██║██╔══██║
{R} ██║     ██║  ██║██║███████║██║  ██║
{R} ╚═╝     ╚═╝  ╚═╝╚═╝╚══════╝╚═╝  ╚═╝{X}
{D}         G U A R D  —  AI Phishing Detector v2.0{X}
"""

def sev_color(s):
    return {
        'critical': R, 'high': Y, 'medium': C, 'low': B, 'safe': G
    }.get(s, W)

def print_result(r):
    icon  = "🚨" if r['is_phishing'] else "✅"
    color = R if r['is_phishing'] else G
    print(f"\n{color}{BOLD}{'─'*55}{X}")
    print(f"{icon}  {color}{BOLD}{r['prediction'].upper()}{X} EMAIL DETECTED")
    print(f"{color}{'─'*55}{X}")
    print(f"  Confidence  : {BOLD}{r['confidence']}%{X} ({r['confidence_level']})")
    print(f"  Risk Score  : {color}{r['risk_score']}/100{X}")
    print(f"  Phishing    : {R}{r['phishing_prob']}%{X}")
    print(f"  Safe        : {G}{r['safe_prob']}%{X}")

    print(f"\n  {W}{BOLD}Threat Indicators:{X}")
    for ind in r['indicators']:
        c = sev_color(ind['severity'])
        print(f"  {ind['icon']} {c}{ind['type']}{X} — {D}{ind['detail']}{X}")

    print()


def run_cli():
    parser = argparse.ArgumentParser(description='PhishGuard — AI Phishing Email Detector')
    parser.add_argument('email', nargs='?', help='Email text to analyze')
    parser.add_argument('-f', '--file', help='File with emails (one per line, separated by ---)')
    parser.add_argument('--train', action='store_true', help='Retrain the model')
    parser.add_argument('-o', '--output', help='Save results to JSON file')
    args = parser.parse_args()

    print(BANNER)

    # Import model
    from model import PhishingDetector, get_training_data

    # Load or train model
    model_path = "phishing_model.pkl"
    if args.train or not os.path.exists(model_path):
        print(f"{Y}[*] Training model...{X}")
        detector = PhishingDetector()
        emails, labels = get_training_data()
        metrics = detector.train(emails, labels)
        detector.save(model_path)
        print(f"{G}[+] Model trained! Accuracy: {metrics['accuracy']}%{X}\n")
    else:
        print(f"{G}[+] Loading model from {model_path}{X}\n")
        detector = PhishingDetector.load(model_path)

    results = []

    # Single email from args
    if args.email:
        r = detector.predict(args.email)
        print_result(r)
        results.append(r)

    # Batch from file
    elif args.file:
        try:
            with open(args.file) as f:
                content = f.read()
            emails = [e.strip() for e in content.split('---') if e.strip()]
            print(f"{C}[*] Analyzing {len(emails)} emails...{X}\n")
            phish = 0
            for i, email in enumerate(emails):
                r = detector.predict(email)
                icon  = "🚨" if r['is_phishing'] else "✅"
                color = R if r['is_phishing'] else G
                preview = email[:60].replace('\n',' ')
                print(f"  [{i+1:2}] {icon} {color}{r['prediction']:8}{X} {r['confidence']:5.1f}% | {preview}...")
                if r['is_phishing']: phish += 1
                results.append(r)

            print(f"\n  {W}Summary:{X} {R}{phish} phishing{X} / {G}{len(emails)-phish} safe{X} out of {len(emails)} emails")

        except FileNotFoundError:
            print(f"{R}[!] File not found: {args.file}{X}")
            sys.exit(1)

    # Interactive mode
    else:
        print(f"{C}Interactive Mode — type or paste email, then press Enter twice to analyze.{X}")
        print(f"{D}Type 'quit' to exit, 'stats' to see model metrics.{X}\n")
        while True:
            print(f"{W}Enter email content (empty line to submit):{X}")
            lines = []
            while True:
                try:
                    line = input()
                except EOFError:
                    break
                if line.lower() == 'quit':
                    print(f"\n{G}Goodbye!{X}")
                    sys.exit(0)
                if line.lower() == 'stats':
                    m = detector.metrics
                    print(f"\n{W}Model Metrics:{X}")
                    print(f"  Accuracy  : {G}{m['accuracy']}%{X}")
                    print(f"  Precision : {B}{m['precision']}%{X}")
                    print(f"  Recall    : {P}{m['recall']}%{X}")
                    print(f"  F1 Score  : {Y}{m['f1_score']}%{X}")
                    print(f"  ROC AUC   : {C}{m['roc_auc']}%{X}\n")
                    lines = []
                    break
                if line == '' and lines:
                    break
                lines.append(line)

            if lines:
                email = '\n'.join(lines)
                r = detector.predict(email)
                print_result(r)
                results.append(r)

    # Save to JSON if requested
    if args.output and results:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"{G}[+] Results saved to {args.output}{X}")


if __name__ == "__main__":
    run_cli()
