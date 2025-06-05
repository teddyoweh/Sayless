from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="sayless",
    version="0.1.0",
    author="Teddy",
    author_email="",  # Add your email
    description="AI-powered Git commit message generator and repository analysis tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/teddy/sayless",  # Update with your repo URL
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Version Control :: Git",
    ],
    python_requires=">=3.7",
    install_requires=[
        "typer>=0.9.0",
        "rich>=10.0.0",
        "openai>=1.0.0",
        "python-dateutil>=2.8.2",
        "numpy>=1.19.0",
        "requests>=2.25.0",
        "tqdm>=4.65.0",
        "pydantic>=2.0.0",
        "httpx>=0.24.0",
    ],
    entry_points={
        "console_scripts": [
            "sayless=sayless.cli.core:app",
            "sl=sayless.cli.core:app",
        ],
    },
    include_package_data=True,
) 