#!/usr/bin/env python3
"""
检查项目依赖库是否安装的脚本
"""
import sys
import importlib
import subprocess
import pkg_resources

def check_library_installation():
    """检查项目所需的所有库是否已安装"""
    
    # 从requirements.txt读取依赖
    required_from_file = []
    try:
        with open('requirements.txt', 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # 提取库名（去掉版本要求）
                    lib_name = line.split('>=')[0].split('==')[0].split('<')[0].split('>')[0]
                    required_from_file.append(lib_name)
    except FileNotFoundError:
        print("⚠️  requirements.txt 文件未找到")
    
    # 项目中实际使用的库（从import语句分析）
    used_libraries = [
        # 标准库中的第三方库
        'numpy', 'scipy', 'pandas', 'matplotlib', 'networkx',
        # 机器学习库
        'torch', 'torch_geometric', 'sklearn', 'gymnasium', 
        # 工具库
        'tqdm', 'tensorboard', 'psutil', 'optuna',
        # 图处理库
        'metis', 'PyMetis', 'seaborn'
    ]
    
    # 合并库列表
    all_required = list(set(required_from_file + used_libraries))
    
    print('🔍 正在检查Python库安装状态...\n')
    
    missing_libs = []
    installed_libs = []
    
    for lib in all_required:
        try:
            # 处理特殊的库名映射
            import_name = lib
            if lib == 'sklearn':
                import_name = 'sklearn'
            elif lib == 'torch_geometric':
                import_name = 'torch_geometric'
            elif lib in ['PyMetis', 'metis']:
                import_name = 'metis'  # PyMetis 导入时使用 metis
            elif lib == 'gymnasium':
                import_name = 'gymnasium'
                
            # 尝试导入
            module = importlib.import_module(import_name)
            
            # 获取版本信息
            try:
                if lib == 'torch_geometric':
                    import torch_geometric
                    version = torch_geometric.__version__
                elif lib in ['PyMetis', 'metis']:
                    # PyMetis 可能没有 __version__ 属性
                    version = 'installed'
                else:
                    version = getattr(module, '__version__', 'unknown')
                installed_libs.append(f'{lib}: {version}')
            except:
                installed_libs.append(f'{lib}: installed')
                
        except ImportError as e:
            missing_libs.append(lib)
    
    # 显示结果
    print('✅ 已安装的库:')
    for lib in sorted(installed_libs):
        print(f'  ✓ {lib}')
    
    if missing_libs:
        print(f'\n❌ 缺失的库 ({len(missing_libs)} 个):')
        for lib in sorted(missing_libs):
            print(f'  ✗ {lib}')
        
        print(f'\n🔧 建议的安装命令:')
        print(f'pip install {" ".join(missing_libs)}')
        
        # 特殊库的安装说明
        special_installs = []
        for lib in missing_libs:
            if lib == 'torch':
                special_installs.append("# PyTorch (GPU版本):")
                special_installs.append("pip install torch==2.0.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118")
            elif lib == 'torch_geometric':
                special_installs.append("# PyTorch Geometric:")
                special_installs.append("pip install torch-geometric")
            elif lib in ['PyMetis', 'metis']:
                special_installs.append("# PyMetis (图分割库):")
                special_installs.append("pip install PyMetis")
        
        if special_installs:
            print(f'\n📦 特殊库安装说明:')
            for cmd in special_installs:
                print(cmd)
                
        return False
    else:
        print(f'\n🎉 所有必需的库都已安装！')
        return True

def check_import_errors():
    """通过尝试导入检查是否有错误"""
    print('\n🧪 测试关键模块导入...')
    
    test_imports = [
        ('numpy', 'import numpy as np'),
        ('torch', 'import torch'),
        ('networkx', 'import networkx as nx'),
        ('matplotlib', 'import matplotlib.pyplot as plt'),
        ('pandas', 'import pandas as pd'),
        ('sklearn', 'from sklearn.cluster import KMeans'),
        ('gymnasium', 'import gymnasium as gym'),
        ('tqdm', 'from tqdm import tqdm'),
    ]
    
    errors = []
    
    for name, import_code in test_imports:
        try:
            exec(import_code)
            print(f'  ✓ {name}')
        except Exception as e:
            print(f'  ✗ {name}: {e}')
            errors.append((name, str(e)))
    
    if errors:
        print(f'\n❌ 发现 {len(errors)} 个导入错误:')
        for name, error in errors:
            print(f'  • {name}: {error}')
        return False
    else:
        print('\n✅ 所有关键模块导入成功！')
        return True

def main():
    """主函数"""
    print("=" * 60)
    print("🔍 Python依赖库检查工具")
    print("=" * 60)
    
    # 检查库安装状态
    deps_ok = check_library_installation()
    
    # 检查导入错误
    imports_ok = check_import_errors()
    
    print("\n" + "=" * 60)
    if deps_ok and imports_ok:
        print("🎉 所有检查通过！你的环境配置正确。")
    else:
        print("⚠️  发现问题，请根据上述建议安装缺失的库。")
        print("\n💡 如果安装完成后仍有问题，可以运行:")
        print("   python check_dependencies.py")
        print("   再次检查")
    print("=" * 60)

if __name__ == "__main__":
    main() 