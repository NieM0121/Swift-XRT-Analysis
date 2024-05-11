import setuptools


setuptools.setup(
    name="swift_xrt_analysis",
    version="0.1",
    author="Meng-Nie",
    author_email="niemeng@mail.ynu.edu.cn",
    description="SWIFT/XRT telescope data processing tools.",
    long_description=open("README.md").read(),
    long_description_content_type="Markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

