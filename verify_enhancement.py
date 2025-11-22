#!/usr/bin/env python3
"""
Verify the notebook enhancement and provide comprehensive summary.
"""

import json
from pathlib import Path

def verify_notebook():
    notebook_path = Path("notebooks/Advanced_RVC_Inference.ipynb")
    
    if not notebook_path.exists():
        print("❌ Notebook file not found!")
        return False
    
    with open(notebook_path, 'r') as f:
        notebook = json.load(f)
    
    cells = notebook['cells']
    print(f"📊 Notebook Analysis:")
    print(f"   Total cells: {len(cells)}")
    
    # Count annotations
    title_count = 0
    param_count = 0
    
    print(f"\n🔍 Cell Analysis:")
    for i, cell in enumerate(cells, 1):
        cell_type = cell['cell_type']
        metadata = cell.get('metadata', {})
        cell_id = metadata.get('id', 'no-id')
        
        source = cell.get('source', [])
        if isinstance(source, list):
            source_text = ' '.join(source)
            
            has_title = '#@title' in source_text
            has_params = '#@param' in source_text
            
            if has_title:
                title_count += 1
            if has_params:
                param_count += 1
            
            status = "✅" if has_title else "⚠️"
            param_status = f" ({param_count} params)" if has_params else ""
            
            print(f"   {i:2d}. {status} {cell_type:7s} - {cell_id:20s}{param_status}")
    
    print(f"\n📈 Annotation Summary:")
    print(f"   Cells with #@title: {title_count}/{len(cells)}")
    print(f"   Cells with #@param: {param_count}")
    
    # Check for CalledProcessError fixes
    source_all = ' '.join([' '.join(cell.get('source', [])) for cell in cells])
    
    has_problematic_install = 'pip install -e' in source_all
    has_correct_branch = 'git pull origin master' in source_all
    has_fixed_install = 'pip install -r requirements.txt' in source_all
    
    print(f"\n🔧 Error Fixes:")
    print(f"   ❌ CalledProcessError source (pip install -e): {'FOUND' if has_problematic_install else 'FIXED'}")
    print(f"   ✅ Correct branch usage (master): {'FOUND' if has_correct_branch else 'MISSING'}")
    print(f"   ✅ Fixed installation method: {'FOUND' if has_fixed_install else 'MISSING'}")
    
    # Overall assessment
    success = (
        title_count == len(cells) and 
        param_count > 0 and 
        not has_problematic_install and
        has_correct_branch
    )
    
    print(f"\n🎯 Overall Assessment:")
    if success:
        print(f"   ✅ SUCCESS: Notebook fully enhanced with all requirements met!")
    else:
        print(f"   ⚠️  ISSUES: Some requirements not fully met")
    
    return success

if __name__ == "__main__":
    print("🚀 COLAB NOTEBOOK ENHANCEMENT VERIFICATION")
    print("=" * 50)
    
    success = verify_notebook()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 ENHANCEMENT COMPLETE - All requirements met!")
        print("\n📋 Summary of Changes:")
        print("   • Added #@title to all notebook cells")
        print("   • Added #@param for customizable configuration")
        print("   • Fixed CalledProcessError by removing pip install -e")
        print("   • Corrected branch references to use 'master'")
        print("   • Enhanced user experience with better documentation")
    else:
        print("❌ ENHANCEMENT INCOMPLETE - Review issues above")