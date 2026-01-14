"""
语法检查脚本
"""
import sys
import py_compile

files_to_check = [
    'F:/v3/scripts/train.py',
    'F:/v3/scripts/debug_vit_params.py',
    'F:/v3/scripts/test_freeze_strategy.py',
]

print("Checking Python syntax...")
all_passed = True

for filepath in files_to_check:
    try:
        py_compile.compile(filepath, doraise=True)
        print(f"✓ {filepath}")
    except py_compile.PyCompileError as e:
        print(f"✗ {filepath}")
        print(f"  Error: {e}")
        all_passed = False

if all_passed:
    print("\n✓ All syntax checks passed!")
else:
    print("\n✗ Some files have syntax errors!")
    sys.exit(1)
