#!/usr/bin/env python3
"""
Fix plots.py indentation issue
==============================
The functions 'plot_EPN_scenarios_plotly' and 'plot_EPN_scenarios_plotly_combined'
are incorrectly indented (nested inside plot_EPN function).

This script fixes the indentation by removing 4 spaces from those functions.

Usage:
    python fix_plots_indentation.py
    
    # or with custom paths
    python fix_plots_indentation.py --input plots.py --output plots_fixed.py
"""

import argparse
import re
import os


def fix_plots_file(input_path: str, output_path: str = None) -> bool:
    """
    Fix the indentation issue in plots.py
    
    Args:
        input_path: Path to the original plots.py
        output_path: Path for the fixed file (default: overwrite input)
    
    Returns:
        True if successful, False otherwise
    """
    if output_path is None:
        output_path = input_path
    
    # Read the file
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split into lines
    lines = content.split('\n')
    
    # Find the problematic sections
    # Looking for lines that start with "    def plot_EPN_scenarios_plotly"
    
    fixed_lines = []
    in_nested_function = False
    nested_start_patterns = [
        '    def plot_EPN_scenarios_plotly(',
        '    def plot_EPN_scenarios_plotly_combined('
    ]
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check if we're entering a nested function that needs fixing
        for pattern in nested_start_patterns:
            if line.startswith(pattern):
                print(f"Found nested function at line {i+1}: {line[:50]}...")
                in_nested_function = True
                # Remove 4-space indent from this line
                fixed_lines.append(line[4:] if line.startswith('    ') else line)
                i += 1
                
                # Continue fixing until we hit another function at module level
                # (a line starting with 'def ' without indent)
                while i < len(lines):
                    current_line = lines[i]
                    
                    # Check if we've reached the next module-level function
                    if current_line.startswith('def ') and not current_line.startswith('    '):
                        in_nested_function = False
                        break
                    
                    # Check if we've hit the next incorrectly nested function
                    next_nested = False
                    for p in nested_start_patterns:
                        if current_line.startswith(p):
                            next_nested = True
                            break
                    
                    if next_nested:
                        # Let the outer loop handle it
                        in_nested_function = False
                        break
                    
                    # Remove 4-space indent if line has it
                    if current_line.startswith('    '):
                        fixed_lines.append(current_line[4:])
                    else:
                        fixed_lines.append(current_line)
                    
                    i += 1
                
                break  # Don't process this line again
        else:
            # Normal line, keep as-is
            fixed_lines.append(line)
            i += 1
    
    # Write the fixed content
    fixed_content = '\n'.join(fixed_lines)
    
    # Create backup if overwriting
    if output_path == input_path:
        backup_path = input_path + '.backup'
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Backup saved to: {backup_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(fixed_content)
    
    print(f"Fixed file saved to: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Fix plots.py indentation issue'
    )
    parser.add_argument(
        '--input', '-i',
        default='plots.py',
        help='Input file path (default: plots.py)'
    )
    parser.add_argument(
        '--output', '-o',
        default=None,
        help='Output file path (default: overwrite input)'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"ERROR: File not found: {args.input}")
        return 1
    
    print(f"Fixing indentation in: {args.input}")
    success = fix_plots_file(args.input, args.output)
    
    if success:
        print("\n✅ Fix complete!")
        print("\nPlease verify the changes:")
        print("  1. Open plots.py")
        print("  2. Check that 'plot_EPN_scenarios_plotly' is at module level (no indent)")
        print("  3. Check that 'plot_EPN_scenarios_plotly_combined' is at module level")
        print("  4. Run: python -c 'import plots' to verify no syntax errors")
        return 0
    else:
        print("\n❌ Fix failed!")
        return 1


if __name__ == '__main__':
    exit(main())
