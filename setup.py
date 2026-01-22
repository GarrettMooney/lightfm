# coding=utf-8
"""
Minimal setup.py for custom build commands.

All metadata is defined in pyproject.toml. This file only contains:
- Extension module definitions with platform-specific flags
- Custom Cythonize command for template generation
- Custom Clean command for build cleanup
"""
import os
import subprocess
import sys
import textwrap

from setuptools import Command, Extension, setup


def find_libomp():
    """
    Find libomp installation on macOS (via Homebrew).

    Returns a tuple of (include_dir, lib_dir) if found, or (None, None) otherwise.
    """
    if not sys.platform.startswith("darwin"):
        return None, None

    # Check common Homebrew locations
    # Apple Silicon: /opt/homebrew/opt/libomp
    # Intel Mac: /usr/local/opt/libomp
    homebrew_paths = [
        "/opt/homebrew/opt/libomp",  # Apple Silicon
        "/usr/local/opt/libomp",      # Intel Mac
    ]

    for base_path in homebrew_paths:
        include_dir = os.path.join(base_path, "include")
        lib_dir = os.path.join(base_path, "lib")
        if os.path.exists(include_dir) and os.path.exists(lib_dir):
            return include_dir, lib_dir

    # Try to find via brew --prefix
    try:
        result = subprocess.run(
            ["brew", "--prefix", "libomp"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            base_path = result.stdout.strip()
            include_dir = os.path.join(base_path, "include")
            lib_dir = os.path.join(base_path, "lib")
            if os.path.exists(include_dir) and os.path.exists(lib_dir):
                return include_dir, lib_dir
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    return None, None


def define_extensions(use_openmp, libomp_paths=None):
    """Define C extensions with platform-specific compile flags."""
    compile_args = []
    if not os.environ.get("LIGHTFM_NO_CFLAGS"):
        compile_args += ["-ffast-math"]

        if sys.platform.startswith("darwin"):
            compile_args += []
        else:
            compile_args += ["-march=native"]

    if not use_openmp:
        print("Compiling without OpenMP support.")
        return [
            Extension(
                "lightfm._lightfm_fast_no_openmp",
                ["lightfm/_lightfm_fast_no_openmp.c"],
                extra_compile_args=compile_args,
            )
        ]

    # Build OpenMP-enabled extension
    openmp_compile_args = compile_args.copy()
    openmp_link_args = []
    include_dirs = []
    library_dirs = []

    if sys.platform.startswith("darwin") and libomp_paths:
        # macOS with libomp from Homebrew
        include_dir, lib_dir = libomp_paths
        print(f"Compiling with OpenMP support (libomp from {lib_dir}).")
        include_dirs.append(include_dir)
        library_dirs.append(lib_dir)
        openmp_compile_args.extend(["-Xpreprocessor", "-fopenmp"])
        openmp_link_args.extend(["-lomp"])
    else:
        # Linux and other platforms with native OpenMP
        print("Compiling with OpenMP support.")
        openmp_compile_args.append("-fopenmp")
        openmp_link_args.append("-fopenmp")

    return [
        Extension(
            "lightfm._lightfm_fast_openmp",
            ["lightfm/_lightfm_fast_openmp.c"],
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            extra_link_args=openmp_link_args,
            extra_compile_args=openmp_compile_args,
        )
    ]


class Cythonize(Command):
    """
    Compile the extension .pyx files from the template.

    This command generates two variants from _lightfm_fast.pyx.template:
    - _lightfm_fast_no_openmp.pyx (single-threaded)
    - _lightfm_fast_openmp.pyx (multi-threaded with OpenMP)

    Usage: python setup.py cythonize
    """

    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def generate_pyx(self):
        """Generate .pyx files from template with OpenMP/no-OpenMP variants."""
        openmp_import = textwrap.dedent(
            """
             from cython.parallel import parallel, prange
             cimport openmp
        """
        )

        lock_init = textwrap.dedent(
            """
             cdef openmp.omp_lock_t THREAD_LOCK
             openmp.omp_init_lock(&THREAD_LOCK)
        """
        )

        params = (
            (
                "no_openmp",
                dict(
                    openmp_import="",
                    nogil_block="with nogil:",
                    range_block="range",
                    thread_num="0",
                    lock_init="",
                    lock_acquire="",
                    lock_release="",
                ),
            ),
            (
                "openmp",
                dict(
                    openmp_import=openmp_import,
                    nogil_block="with nogil, parallel(num_threads=num_threads):",
                    range_block="prange",
                    thread_num="openmp.omp_get_thread_num()",
                    lock_init=lock_init,
                    lock_acquire="openmp.omp_set_lock(&THREAD_LOCK)",
                    lock_release="openmp.omp_unset_lock(&THREAD_LOCK)",
                ),
            ),
        )

        file_dir = os.path.join(os.path.dirname(__file__), "lightfm")

        with open(os.path.join(file_dir, "_lightfm_fast.pyx.template"), "r") as fl:
            template = fl.read()

        for variant, template_params in params:
            with open(
                os.path.join(file_dir, "_lightfm_fast_{}.pyx".format(variant)), "w"
            ) as fl:
                fl.write(template.format(**template_params))

    def run(self):
        from Cython.Build import cythonize

        self.generate_pyx()

        cythonize(
            [
                Extension(
                    "lightfm._lightfm_fast_no_openmp",
                    ["lightfm/_lightfm_fast_no_openmp.pyx"],
                ),
                Extension(
                    "lightfm._lightfm_fast_openmp",
                    ["lightfm/_lightfm_fast_openmp.pyx"],
                    extra_link_args=["-fopenmp"],
                ),
            ],
            compiler_directives={"language_level": "3"},
        )


class Clean(Command):
    """
    Clean build files.

    Usage: python setup.py clean
    """

    user_options = [("all", None, "(Compatibility with original clean command)")]

    def initialize_options(self):
        self.all = False

    def finalize_options(self):
        pass

    def run(self):
        pth = os.path.dirname(os.path.abspath(__file__))

        subprocess.call(["rm", "-rf", os.path.join(pth, "build")])
        subprocess.call(["rm", "-rf", os.path.join(pth, "lightfm.egg-info")])
        subprocess.call(["find", pth, "-name", "lightfm*.pyc", "-type", "f", "-delete"])
        subprocess.call(["rm", "-f", os.path.join(pth, "lightfm", "_lightfm_fast.so")])


# Determine OpenMP support
libomp_paths = None
if sys.platform.startswith("darwin"):
    # macOS: check for libomp from Homebrew
    libomp_paths = find_libomp()
    if libomp_paths[0] is not None:
        use_openmp = True
    else:
        print(
            "libomp not found. Install with: brew install libomp\n"
            "Building without OpenMP support (single-threaded)."
        )
        use_openmp = False
elif sys.platform.startswith("win"):
    # Windows: OpenMP not yet supported
    use_openmp = False
else:
    # Linux and other Unix-like systems: native OpenMP support
    use_openmp = True

setup(
    cmdclass={"cythonize": Cythonize, "clean": Clean},
    ext_modules=define_extensions(use_openmp, libomp_paths),
)
