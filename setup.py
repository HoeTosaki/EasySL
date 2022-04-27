import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="EasySL",
    version = "0.1.0",
    author="HoeWang@THU",
    author_email="1139032564@qq.com",
    description="Easy Save&Load (ESL) provides an automatic data management for Python.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    license="GPLv3",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    # 依赖模块
    install_requires=[
        'numpy',
        'pandas',
        'dgl',
        'networkx',
        'torch',
    ],
    python_requires=">=3",
    url="https://github.com/HoeTosaki/EasySL",
)

