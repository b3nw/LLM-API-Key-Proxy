# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Mirrowel

import os
import shutil


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(script_dir, "..")
    target_dir = os.path.join(os.path.expanduser("~"), "Desktop", "LLM-proxy-dev")

    os.makedirs(target_dir, exist_ok=True)

    for pkg in ("proxy_app", "rotator_library"):
        src_pkg = os.path.join(src_dir, pkg)
        dst_pkg = os.path.join(target_dir, pkg)
        if os.path.isdir(dst_pkg):
            shutil.rmtree(dst_pkg)
        shutil.copytree(src_pkg, dst_pkg, dirs_exist_ok=True)
        print(f"Copied {pkg} -> {dst_pkg}")

    for env_file in (".env", ".env.example"):
        env_src = os.path.join(src_dir, "..", env_file)
        if os.path.isfile(env_src):
            shutil.copy2(env_src, os.path.join(target_dir, env_file))
            print(f"Copied {env_file}")

    req_src = os.path.join(src_dir, "..", "requirements.txt")
    if os.path.isfile(req_src):
        shutil.copy2(req_src, os.path.join(target_dir, "requirements.txt"))
        print("Copied requirements.txt")

    print(f"Done. Run with: python {target_dir}/proxy_app/main.py")


if __name__ == "__main__":
    main()
