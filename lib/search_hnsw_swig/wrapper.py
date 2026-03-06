"""
Does all sorts of dark magic in order to build/import c++ bfs
"""
"""
Does all sorts of dark magic in order to build/import c++ bfs
"""
import os
import os.path as osp
import subprocess
import sys

package_abspath = osp.dirname(osp.abspath(__file__))  # 简化路径拼接（等价于原代码）
so_path = osp.join(package_abspath, '_search_hnsw.so')

if not os.path.exists(so_path):
    # 替换 sandbox.run_setup：用 subprocess 执行 setup.py 编译
    workdir = os.getcwd()
    try:
        os.chdir(package_abspath)
        # 执行 setup.py clean + build（替代 sandbox.run_setup）
        subprocess.run(
            [sys.executable, "setup.py", "clean", "build"],
            check=True,  # 执行失败时抛异常
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        # 拷贝编译后的 SO 文件（兼容不同系统的 build/lib 目录）
        build_lib_dir = osp.join(package_abspath, "build")
        for root, _, files in os.walk(build_lib_dir):
            for file in files:
                if file.endswith('.so'):
                    src_so = osp.join(root, file)
                    os.system(f'cp {src_so} {so_path}')
                    break
        assert os.path.exists(so_path), "编译失败：未生成 _search_hnsw.so"
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"编译 C++ 模块失败：{e.stderr}")
    finally:
        os.chdir(workdir)

# 导入编译后的模块（保留原逻辑）
try:
    from . import _search_hnsw as search_hnsw_module
except ImportError as e:
    raise ImportError(f"导入 C++ 模块失败：{e}")

search_hnsw = search_hnsw_module.find_nearest