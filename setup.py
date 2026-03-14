from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="mrv-lib",
    version="0.1.0",
    author="Kai Zheng",
    author_email="kaizhengnz@gmail.com",
    maintainer="ModelGuard Lab",
    description="Market Regime Validity Library — model risk diagnostics in non-stationary markets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/modelguard-lab/mrv-lib",
    project_urls={
        "Documentation": "https://github.com/modelguard-lab/mrv-lib#readme",
        "Commercial": "https://modelguard.co.nz",
    },
    packages=find_packages(exclude=("tests", "tests.*", "docs", "examples")),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "dev": ["pytest", "pytest-cov", "black", "ruff"],
    },
    entry_points={
        "console_scripts": [
            "mrv-lib = mrv_lib.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
)
