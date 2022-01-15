from setuptools import setup, find_packages

setup(
    name="screenply",
    version="0.1",
    packages=find_packages(),

    install_requires=["pdfminer.six","pandas","numpy","bs4"],

    package_data={
    },

    author="Geoff Kaufman",
    author_email="geoff.kaufman2@gmail.com",
    description="Extract data from screenplay PDFs",
    url="https://github.com/glentennis/screenply",  
    project_urls={
    },
    classifiers=[
    ]

)