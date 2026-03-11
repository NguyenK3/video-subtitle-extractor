import argparse
import os
from qpt.executor import CreateExecutableModule as CEM
from qpt.modules.cuda import CopyCUDAPackage
from qpt.smart_opt import set_default_pip_source
from qpt.kernel.qinterpreter import PYPI_PIP_SOURCE


def main():
    WORK_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    LAUNCH_PATH = os.path.join(WORK_DIR, 'gui.py')
    SAVE_PATH = os.path.join(os.path.dirname(WORK_DIR), 'vse_out')
    ICON_PATH = os.path.join(WORK_DIR, "design", "vse.ico")

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Build distribution package")
    parser.add_argument(
        "--cuda",
        nargs="?",                # Optional argument value
        const="10.2",             # Default value is 10.2 if only --cuda is specified
        default=None,             # If --cuda is not specified, defaults to None
        help="Include CUDA module, optionally specify version, e.g. --cuda or --cuda=11.8"
    )

    args = parser.parse_args()

    sub_modules = []

    if args.cuda:
        sub_modules.append(CopyCUDAPackage(cuda_version=args.cuda))

    if os.getenv("QPT_Action") == "True":
        set_default_pip_source(PYPI_PIP_SOURCE)

    module = CEM(
        work_dir=WORK_DIR,
        launcher_py_path=LAUNCH_PATH,
        save_path=SAVE_PATH,
        icon=ICON_PATH,
        hidden_terminal=False,
        requirements_file="./requirements.txt",
        sub_modules=sub_modules,
    )

    module.make()


if __name__ == '__main__':
    main()
