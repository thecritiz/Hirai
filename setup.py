from setuptools import setup, find_packages

setup(
    name="hiring_system",
    version="0.1.0",
    author="Aditya",
    packages=find_packages(),
    install_requires=[
        "langchain>=0.1.0",
        "langgraph>=0.0.20",
        "crewai>=0.28.0",
        "sentence-transformers>=2.2.2",
        "faiss-cpu>=1.7.4",
        "boto3>=1.28.0",
        "botocore>=1.31.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "networkx>=3.0",
        "matplotlib>=3.7.0",
        "pydantic>=2.0.0",
        "python-dotenv>=1.0.0",
        "torch>=2.0.0"
    ],
)
