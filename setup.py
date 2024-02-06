from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

install_requires = [
    "transformers==4.34.0",
    "datasets",
    "scikit-learn",
    "evaluate",
    "accelerate",
    "matplotlib",
    "tqdm",
    "huggingface_hub",
    "wandb",
    "einops",
    "deepspeed"
]

setup(  
    name="metatreelib",
    version="0.1.0",
    author="Yufan Zhuang, Lucas Liu, Chandan Singh, Jingbo Shang, Jianfeng Gao",
    author_email="y5zhuang@ucsd.edu",
    description="PyTorch Implementation for MetaTree: Learning a Decision Tree Algorithm with Transformers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/EvanZhuang/MetaTree",
    packages=find_packages(),
    include_package_data=True,
    python_requires='>=3.7.0',
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)