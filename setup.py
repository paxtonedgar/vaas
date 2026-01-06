"""Setup script for VaaS Tax Document Intelligence Pipeline."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="vaas",
    version="0.1.0",
    author="VaaS Team",
    description="Knowledge graph-based RAG system for IRS tax documents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "PyMuPDF>=1.23.0",
        "pdfplumber>=0.10.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "networkx>=3.1",
        "tqdm>=4.66.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "ml": [
            # Phase B: Fine-tuning (requires Python ≤3.12 for torch compatibility)
            "sentence-transformers>=2.2.0",
            "spacy>=3.7.0",
        ],
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "jupyter>=1.0.0",
            "jupyterlab>=4.0.0",
        ],
        "databricks": [
            "pyspark>=3.5.0",
            "delta-spark>=3.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.10",
    ],
)
