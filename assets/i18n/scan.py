import ast
import json
import os
import sys
from pathlib import Path
from collections import OrderedDict, defaultdict
import argparse


def extract_i18n_strings(node):
    """
    Extract i18n strings from AST nodes
    
    Args:
        node: AST node to process
        
    Returns:
        List of extracted i18n strings
    """
    i18n_strings = []

    # Check for direct i18n() calls
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "i18n"
    ):
        for arg in node.args:
            if isinstance(arg, ast.Str):
                i18n_strings.append(arg.s)
            elif isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                i18n_strings.append(arg.value)

    # Recursively process child nodes
    for child_node in ast.iter_child_nodes(node):
        i18n_strings.extend(extract_i18n_strings(child_node))

    return i18n_strings


def process_file(file_path, verbose=False):
    """
    Process a single Python file to extract i18n strings
    
    Args:
        file_path: Path to the Python file
        verbose: Whether to print processing info
        
    Returns:
        List of i18n strings found in the file
    """
    try:
        with open(file_path, "r", encoding="utf8") as file:
            code = file.read()
            if "I18nAuto" in code or "i18n(" in code:
                try:
                    tree = ast.parse(code)
                    i18n_strings = extract_i18n_strings(tree)
                    if verbose:
                        print(f"  {file_path}: {len(i18n_strings)} strings")
                    return i18n_strings
                except SyntaxError as e:
                    print(f"  Syntax error in {file_path}: {e}")
                    return []
    except Exception as e:
        print(f"  Error reading {file_path}: {e}")
        return []
    return []


def scan_directory(directory=".", file_patterns=None, verbose=False):
    """
    Scan directory for i18n strings
    
    Args:
        directory: Directory to scan
        file_patterns: List of file patterns to include
        verbose: Whether to print detailed info
        
    Returns:
        Set of unique i18n strings
    """
    if file_patterns is None:
        file_patterns = ["**/*.py", "**/*.ts", "**/*.js", "**/*.jsx"]
    
    # Use a set to store unique strings
    code_keys = set()
    file_counts = defaultdict(int)
    
    for pattern in file_patterns:
        files = Path(directory).rglob(pattern)
        for py_file in files:
            if py_file.is_file():
                strings = process_file(py_file, verbose)
                if strings:
                    file_counts[str(py_file)] = len(strings)
                    code_keys.update(strings)
    
    if verbose:
        print(f"\nProcessed {len(file_counts)} files")
        total_strings = sum(file_counts.values())
        print(f"Total unique strings: {len(code_keys)}")
        print(f"Total string instances: {total_strings}")
    
    return code_keys, file_counts


def compare_with_languages(language_dir="languages", standard_lang="en_US"):
    """
    Compare found strings with existing language files
    
    Args:
        language_dir: Directory containing language files
        standard_lang: Standard language file to use as reference
        
    Returns:
        Dictionary with comparison results
    """
    lang_path = Path(language_dir)
    if not lang_path.exists():
        return {"error": f"Language directory {language_dir} not found"}
    
    standard_file = lang_path / f"{standard_lang}.json"
    if not standard_file.exists():
        return {"error": f"Standard language file {standard_file} not found"}
    
    try:
        with open(standard_file, "r", encoding="utf-8") as file:
            standard_data = json.load(file, object_pairs_hook=OrderedDict)
        standard_keys = set(standard_data.keys())
    except Exception as e:
        return {"error": f"Error reading standard file: {e}"}
    
    # Find other language files
    other_lang_files = []
    for lang_file in lang_path.glob("*.json"):
        if lang_file.name != f"{standard_lang}.json":
            try:
                with open(lang_file, "r", encoding="utf-8") as file:
                    lang_data = json.load(file, object_pairs_hook=OrderedDict)
                other_lang_files.append({
                    "file": lang_file,
                    "keys": set(lang_data.keys()),
                    "data": lang_data
                })
            except Exception as e:
                print(f"Error reading {lang_file}: {e}")
    
    return {
        "standard_keys": standard_keys,
        "other_languages": other_lang_files
    }


def generate_missing_translations(code_keys, language_dir="languages", output_file=None):
    """
    Generate missing translations for all language files
    
    Args:
        code_keys: Set of all found i18n strings
        language_dir: Directory containing language files
        output_file: Optional output file for the updated standard language
    """
    lang_path = Path(language_dir)
    if not lang_path.exists():
        print(f"Language directory {language_dir} not found")
        return
    
    # Process each language file
    for lang_file in lang_path.glob("*.json"):
        print(f"\nProcessing {lang_file.name}...")
        
        try:
            with open(lang_file, "r", encoding="utf-8") as file:
                lang_data = json.load(file, object_pairs_hook=OrderedDict)
            
            lang_keys = set(lang_data.keys())
            missing_keys = code_keys - lang_keys
            unused_keys = lang_keys - code_keys
            
            print(f"  Current keys: {len(lang_keys)}")
            print(f"  Missing keys: {len(missing_keys)}")
            print(f"  Unused keys: {len(unused_keys)}")
            
            if missing_keys:
                print("  Adding missing keys...")
                for key in sorted(missing_keys):
                    lang_data[key] = key  # Add with default value (same as key)
                print(f"  Added {len(missing_keys)} missing keys")
            
            if unused_keys:
                print(f"  Found {len(unused_keys)} unused keys (keeping for compatibility)")
            
            # Write updated file
            with open(lang_file, "w", encoding="utf-8") as file:
                json.dump(lang_data, file, ensure_ascii=False, indent=4, sort_keys=True)
                file.write("\n")
            
            print(f"  Updated {lang_file.name}")
            
        except Exception as e:
            print(f"  Error processing {lang_file.name}: {e}")
    
    # Optionally update the main standard file
    if output_file:
        code_keys_dict = OrderedDict((s, s) for s in sorted(code_keys))
        with open(output_file, "w", encoding="utf-8") as file:
            json.dump(code_keys_dict, file, ensure_ascii=False, indent=4, sort_keys=True)
            file.write("\n")
        print(f"\nUpdated standard file: {output_file}")


def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description="Scan and update i18n strings")
    parser.add_argument("--scan-dir", default=".", help="Directory to scan (default: current directory)")
    parser.add_argument("--patterns", nargs="+", default=["**/*.py", "**/*.ts", "**/*.js", "**/*.jsx"],
                       help="File patterns to scan")
    parser.add_argument("--languages-dir", default="assets/i18n/languages", 
                       help="Languages directory (default: assets/i18n/languages)")
    parser.add_argument("--standard-lang", default="en_US", help="Standard language (default: en_US)")
    parser.add_argument("--output-standard", help="Output file for updated standard language")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--generate-missing", action="store_true", 
                       help="Generate missing translations in all language files")
    
    args = parser.parse_args()
    
    # Change to scan directory
    original_cwd = os.getcwd()
    if args.scan_dir != ".":
        os.chdir(args.scan_dir)
    
    try:
        # Scan for i18n strings
        print("Scanning for i18n strings...")
        code_keys, file_counts = scan_directory(
            directory=args.scan_dir if args.scan_dir != "." else ".",
            file_patterns=args.patterns,
            verbose=args.verbose
        )
        
        if not code_keys:
            print("No i18n strings found!")
            return
        
        # Compare with existing languages
        print("\nComparing with existing language files...")
        lang_comparison = compare_with_languages(
            language_dir=args.languages_dir,
            standard_lang=args.standard_lang
        )
        
        if "error" in lang_comparison:
            print(f"Language comparison error: {lang_comparison['error']}")
            return
        
        standard_keys = lang_comparison["standard_keys"]
        other_languages = lang_comparison["other_languages"]
        
        # Calculate differences
        missing_keys = code_keys - standard_keys
        unused_keys = standard_keys - code_keys
        
        print(f"\nResults:")
        print(f"  Found in code: {len(code_keys)}")
        print(f"  In standard language: {len(standard_keys)}")
        print(f"  Missing from standard: {len(missing_keys)}")
        print(f"  Unused in standard: {len(unused_keys)}")
        
        if args.verbose and missing_keys:
            print(f"\nMissing keys ({len(missing_keys)}):")
            for key in sorted(missing_keys)[:10]:  # Show first 10
                print(f"  {key}")
            if len(missing_keys) > 10:
                print(f"  ... and {len(missing_keys) - 10} more")
        
        # Generate missing translations if requested
        if args.generate_missing:
            print("\nGenerating missing translations...")
            generate_missing_translations(
                code_keys=code_keys,
                language_dir=args.languages_dir,
                output_file=args.output_standard
            )
    
    finally:
        # Restore original working directory
        os.chdir(original_cwd)


if __name__ == "__main__":
    main()
